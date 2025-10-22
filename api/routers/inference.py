# 추론/탐지 API 라우터 - yolo_infer_to_gps.py 기반
# YOLO 모델로 피해목 탐지 + GPS 좌표 변환 + CSV 저장

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import os
import tempfile
import shutil
import pandas as pd
import cv2
import rasterio
from rasterio.transform import from_bounds
from rasterio import Affine
from ultralytics import YOLO
import datetime
import json
from pathlib import Path
import sys
import numpy as np
import zipfile

# 프로젝트 루트 경로 추가 (scripts 폴더 접근용)
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from .. import config

router = APIRouter()

# 응답 모델 정의
class DetectionResult(BaseModel):
    filename: str
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float
    confidence: Optional[float] = None
    tm_x: Optional[float] = None  # TM 좌표 X
    tm_y: Optional[float] = None  # TM 좌표 Y

class BatchInferenceResponse(BaseModel):
    success: bool
    message: str
    total_images: int
    total_detections: int
    processed_images: List[str]
    failed_images: List[str]
    csv_file_url: Optional[str] = None
    results_zip_url: Optional[str] = None
    merged_visualization: Optional[str] = None  # 합쳐진 시각화 이미지 파일명

def extract_tile_position(filename: str) -> tuple:
    """타일 파일명에서 위치 정보 추출 (prefix_tile_x_y.tif -> (x, y))"""
    try:
        # 다양한 파일명 형식 지원
        name_without_ext = filename.replace('.tif', '').replace('.tiff', '').replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
        
        # 형식 1: A20250917_tile_2_1 -> ['A20250917', 'tile', '2', '1']
        parts = name_without_ext.split('_')
        
        # tile이 포함된 경우
        if 'tile' in parts:
            tile_idx = parts.index('tile')
            if len(parts) > tile_idx + 2:
                x = int(parts[tile_idx + 1])
                y = int(parts[tile_idx + 2])
                return (x, y)
        
        # 형식 2: prefix_x_y (마지막 두 개가 숫자인 경우)
        if len(parts) >= 3:
            try:
                x = int(parts[-2])
                y = int(parts[-1])
                return (x, y)
            except ValueError:
                pass
        
        # 형식 3: 정규식으로 숫자 패턴 찾기
        import re
        pattern = r'(\d+)_(\d+)(?:\.|$)'
        match = re.search(pattern, filename)
        if match:
            x = int(match.group(1))
            y = int(match.group(2))
            return (x, y)
            
    except (ValueError, IndexError, AttributeError):
        pass
    
    # 파싱 실패시 파일명에 _0_0이 있는지 확인
    if '_0_0' in filename:
        return (0, 0)
    
    print(f"⚠️ 타일 위치 추출 실패: {filename}")
    return (0, 0)  # 기본값


def create_merged_visualization(all_results: List[DetectionResult], output_base: str, 
                               timestamp: str, extract_dir: str, tile_size: int = config.DEFAULT_TILE_SIZE) -> Optional[str]:
    """타일별 탐지 결과를 합쳐서 전체 이미지 시각화 생성"""
    
    if not all_results:
        return None
    
    try:
        print("🖼️ 합쳐진 시각화 이미지 생성 시작...")
        
        # 모든 타일 파일 먼저 수집 (탐지 결과 유무와 관계없이)
        all_tile_files = {}
        tile_positions = {}
        
        # extract_dir에서 모든 타일 파일 찾기 (macOS 숨김 파일 제외)
        for root, dirs, files in os.walk(extract_dir):
            # macOS 시스템 폴더 제외
            if '__MACOSX' in root:
                continue
            for file in files:
                # macOS 숨김 파일 및 시스템 파일 제외
                if file.startswith('.') or file.startswith('._'):
                    continue
                if file.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
                    x, y = extract_tile_position(file)
                    if (x, y) != (0, 0) or '_0_0' in file:  # (0,0)은 실제 좌표이거나 파싱 실패
                        tile_path = os.path.join(root, file)
                        all_tile_files[(x, y)] = tile_path
                        
                        # 탐지 결과가 있는 타일 체크
                        if (x, y) not in tile_positions:
                            tile_positions[(x, y)] = []
        
        # 탐지 결과를 해당 타일 위치에 할당
        for result in all_results:
            x, y = extract_tile_position(result.filename)
            if (x, y) in tile_positions:
                tile_positions[(x, y)].append(result)
        
        if not all_tile_files:
            print("⚠️ 타일 파일을 찾을 수 없습니다.")
            return None
        
        # 실제 존재하는 타일들로부터 격자 범위 계산
        existing_positions = list(all_tile_files.keys())
        min_x = min(pos[0] for pos in existing_positions)
        max_x = max(pos[0] for pos in existing_positions)
        min_y = min(pos[1] for pos in existing_positions)
        max_y = max(pos[1] for pos in existing_positions)
        
        print(f"📊 타일 격자 범위: X({min_x}~{max_x}), Y({min_y}~{max_y})")
        print(f"📊 전체 타일 수: {len(all_tile_files)}개")
        
        # 각 행/열별 실제 크기 계산 (실제 타일 이미지 기반)
        row_heights = {}  # ty -> height
        col_widths = {}   # tx -> width
        
        # 모든 타일의 크기 미리 측정
        print("📏 타일 크기 측정 중...")
        for (tx, ty), tile_path in all_tile_files.items():
            try:
                tile_img = cv2.imread(tile_path)
                if tile_img is not None:
                    h, w = tile_img.shape[:2]
                    
                    # 해당 행/열의 최대 크기 저장
                    if ty not in row_heights:
                        row_heights[ty] = h
                    else:
                        row_heights[ty] = max(row_heights[ty], h)
                    
                    if tx not in col_widths:
                        col_widths[tx] = w
                    else:
                        col_widths[tx] = max(col_widths[tx], w)
                else:
                    print(f"⚠️ 타일 로드 실패: {tile_path}")
            except Exception as e:
                print(f"⚠️ 타일 크기 측정 실패 {tile_path}: {e}")
        
        # 전체 이미지 크기 계산
        total_width = sum(col_widths.get(tx, tile_size) for tx in range(min_x, max_x + 1))
        total_height = sum(row_heights.get(ty, tile_size) for ty in range(min_y, max_y + 1))
        
        print(f"📐 계산된 전체 이미지 크기: {total_width}x{total_height}")
        print(f"📐 행별 높이: {row_heights}")
        print(f"📐 열별 너비: {col_widths}")
        
        # 전체 이미지 생성 (회색 배경으로 시작 - 누락 영역 식별용)
        merged_image = np.full((total_height, total_width, 3), (128, 128, 128), dtype=np.uint8)
        
        # 타일들을 순서대로 합치기
        current_y = 0
        for ty in range(min_y, max_y + 1):
            current_x = 0
            row_height = row_heights.get(ty, tile_size)
            
            for tx in range(min_x, max_x + 1):
                col_width = col_widths.get(tx, tile_size)
                
                if (tx, ty) in all_tile_files:
                    tile_path = all_tile_files[(tx, ty)]
                    try:
                        # 타일 이미지 로드
                        tile_img = cv2.imread(tile_path)
                        if tile_img is not None:
                            h, w = tile_img.shape[:2]
                            
                            # 배치 영역 계산
                            end_x = min(current_x + w, total_width)
                            end_y = min(current_y + h, total_height)
                            
                            actual_w = end_x - current_x
                            actual_h = end_y - current_y
                            
                            if actual_w > 0 and actual_h > 0:
                                merged_image[current_y:end_y, current_x:end_x] = tile_img[:actual_h, :actual_w]
                                print(f"✅ 타일 배치: ({tx},{ty}) at ({current_x},{current_y}) size={actual_w}x{actual_h}")
                        else:
                            print(f"❌ 타일 로드 실패: {tile_path}")
                    except Exception as e:
                        print(f"⚠️ 타일 처리 실패 {tile_path}: {e}")
                else:
                    print(f"⚠️ 타일 누락: ({tx},{ty}) at ({current_x},{current_y})")
                
                current_x += col_width
            
            current_y += row_height
        
        print("🎯 타일 합치기 완료, 탐지 결과 그리기 시작...")
        
        # 탐지 결과 그리기 (개선된 좌표 계산)
        detection_count = 0
        current_y_offset = 0
        
        for ty in range(min_y, max_y + 1):
            current_x_offset = 0
            row_height = row_heights.get(ty, tile_size)
            
            for tx in range(min_x, max_x + 1):
                col_width = col_widths.get(tx, tile_size)
                
                # 해당 타일의 탐지 결과들 그리기
                if (tx, ty) in tile_positions:
                    for detection in tile_positions[(tx, ty)]:
                        # 타일 내 좌표를 전체 이미지 좌표로 변환
                        global_x = current_x_offset + detection.center_x
                        global_y = current_y_offset + detection.center_y
                        
                        # 바운딩 박스 계산
                        x1 = int(global_x - detection.width / 2)
                        y1 = int(global_y - detection.height / 2)
                        x2 = int(global_x + detection.width / 2)
                        y2 = int(global_y + detection.height / 2)
                        
                        # 이미지 경계 내로 제한
                        x1 = max(0, min(x1, total_width))
                        y1 = max(0, min(y1, total_height))
                        x2 = max(0, min(x2, total_width))
                        y2 = max(0, min(y2, total_height))
                        
                        if x2 > x1 and y2 > y1:  # 유효한 바운딩 박스만 그리기
                            # 🎨 신뢰도에 따른 색상 코딩 (신뢰도 텍스트 제거)
                            confidence = detection.confidence if detection.confidence else 0.5
                            if confidence >= 0.7:
                                color = (0, 255, 0)      # 초록색: 높은 신뢰도 (70%+)
                            elif confidence >= 0.4:
                                color = (0, 165, 255)    # 주황색: 중간 신뢰도 (40-70%)
                            else:
                                color = (0, 0, 255)      # 빨간색: 낮은 신뢰도 (40% 미만)
                            
                            # 바운딩 박스 그리기 (신뢰도별 색상)
                            cv2.rectangle(merged_image, (x1, y1), (x2, y2), color, 3)
                            
                            detection_count += 1
                
                current_x_offset += col_width
            
            current_y_offset += row_height
        
        print(f"✅ 탐지 결과 그리기 완료: {detection_count}개 바운딩 박스")
        
        # 합쳐진 시각화 이미지 저장
        merged_filename = f"merged_detection_{timestamp}.jpg"
        merged_path = os.path.join(output_base, merged_filename)
        
        # 큰 이미지인 경우 크기 조정 (메모리 절약 및 파일 크기 최적화)
        if total_width > 8192 or total_height > 8192:
            scale = min(8192 / total_width, 8192 / total_height)
            new_width = int(total_width * scale)
            new_height = int(total_height * scale)
            print(f"🔧 이미지 크기 조정: {total_width}x{total_height} → {new_width}x{new_height}")
            merged_image = cv2.resize(merged_image, (new_width, new_height))
        
        # 이미지 품질 설정으로 저장
        cv2.imwrite(merged_path, merged_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        print(f"✅ 합쳐진 시각화 이미지 생성 완료: {merged_filename}")
        return merged_filename
        
    except Exception as e:
        print(f"⚠️ 합쳐진 시각화 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


# 기본 설정
DEFAULT_WEIGHTS = str(config.DEFAULT_MODEL_PATH)
DEFAULT_OUTPUT_DIR = str(config.API_RESULTS_DIR)

def pixel_to_tm(px: float, py: float, tfw: List[float]) -> tuple:
    """픽셀 좌표를 TM 좌표로 변환
    
    Args:
        px, py: 픽셀 좌표
        tfw: TFW 파라미터 리스트 [A, D, B, E, C, F]
    
    Returns:
        tuple: (tm_x, tm_y) TM 좌표
    """
    A, D, B, E, C, F = tfw
    tm_x = A * px + B * py + C
    tm_y = D * px + E * py + F
    return tm_x, tm_y

def get_tfw_from_tiff(image_path: str) -> Optional[List[float]]:
    """TIFF 파일에서 지리참조 정보 추출하여 TFW 파라미터 생성
    
    Args:
        image_path: TIFF 이미지 파일 경로
        
    Returns:
        List[float]: TFW 파라미터 [A, D, B, E, C, F] 또는 None
    """
    try:
        with rasterio.open(image_path) as src:
            if src.transform:
                # Affine 변환 매트릭스에서 TFW 파라미터 추출
                transform = src.transform
                # Affine 매트릭스: [a, b, c, d, e, f]
                # TFW 형식: [A=a, D=d, B=b, E=e, C=c, F=f]
                tfw_params = [
                    transform.a,  # A: X 픽셀 크기
                    transform.d,  # D: Y 픽셀 크기 (보통 음수)
                    transform.b,  # B: 회전/기울기
                    transform.e,  # E: 회전/기울기
                    transform.c,  # C: 좌상단 X 좌표
                    transform.f   # F: 좌상단 Y 좌표
                ]
                return tfw_params
    except Exception as e:
        print(f"⚠️ 지리참조 정보 추출 실패: {e}")
    return None

def load_tfw_file(tfw_path: str) -> Optional[List[float]]:
    """TFW 파일에서 변환 파라미터 로드
    
    Args:
        tfw_path: TFW 파일 경로
        
    Returns:
        List[float]: TFW 파라미터 [A, D, B, E, C, F] 또는 None
    """
    try:
        with open(tfw_path, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 6:
                return [float(line.strip()) for line in lines[:6]]
    except Exception as e:
        print(f"⚠️ TFW 파일 로드 실패: {e}")
    return None

def draw_bounding_boxes_on_image(image, results):
    """바운딩 박스를 이미지에 그리는 함수 - 소나무 전용 최적화"""
    # 이미지 크기 가져오기
    img_height, img_width = image.shape[:2]
    
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                # 바운딩 박스 좌표 (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # � YOLO 원본 예측 좌표 사용 (정확한 위치 표시)
                # 바운딩 박스 크기 조정 없이 모델이 예측한 정확한 위치 그대로 사용
                new_x1 = int(x1)
                new_y1 = int(y1)
                new_x2 = int(x2)
                new_y2 = int(y2)
                
                # 신뢰도에 따른 색상 조정 (낮은 신뢰도는 주황색으로)
                if confidence >= 0.7:
                    bbox_color = (0, 255, 0)  # 녹색 (높은 신뢰도)
                elif confidence >= 0.4:
                    bbox_color = (0, 165, 255)  # 주황색 (중간 신뢰도)
                else:
                    bbox_color = (0, 0, 255)  # 빨간색 (낮은 신뢰도)
                
                # 최적화된 바운딩 박스 그리기 (더 얇은 선)
                cv2.rectangle(image, (new_x1, new_y1), (new_x2, new_y2), 
                            bbox_color, 2)  # 두께를 2로 줄임
                
                # 🎨 신뢰도 텍스트 제거 (깔끔한 시각화를 위해)
                # 색상만으로 신뢰도를 표현: 초록(높음) → 주황(중간) → 빨강(낮음)

@router.get("/download/{filename}")
async def download_csv_result(filename: str):
    """
    탐지 결과 CSV 파일을 다운로드합니다.
    """
    file_path = os.path.join(DEFAULT_OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )


@router.get("/visualization/{filename}")
async def download_visualization_image(filename: str):
    """
    탐지 결과 시각화 이미지를 다운로드합니다.
    """
    file_path = os.path.join(str(config.API_VISUALIZATION_DIR), filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="시각화 이미지를 찾을 수 없습니다.")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='image/jpeg'
    )


@router.get("/models")
async def list_available_models():
    """
    사용 가능한 YOLO 모델 목록을 반환합니다.
    """
    models_dir = "models"
    results_dir = "results"
    
    available_models = []
    
    # models 디렉토리 스캔
    if os.path.exists(models_dir):
        for root, dirs, files in os.walk(models_dir):
            for file in files:
                if file.endswith('.pt'):
                    model_path = os.path.join(root, file)
                    available_models.append({
                        "name": file,
                        "path": model_path,
                        "size": os.path.getsize(model_path) if os.path.exists(model_path) else 0
                    })
    
    # results 디렉토리의 weights 폴더도 스캔
    if os.path.exists(results_dir):
        for root, dirs, files in os.walk(results_dir):
            if "weights" in root:
                for file in files:
                    if file.endswith('.pt'):
                        model_path = os.path.join(root, file)
                        available_models.append({
                            "name": f"{os.path.basename(os.path.dirname(root))}_{file}",
                            "path": model_path,
                            "size": os.path.getsize(model_path) if os.path.exists(model_path) else 0
                        })
    
    return {
        "available_models": available_models,
        "total_count": len(available_models)
    }


@router.post("/detect", response_model=BatchInferenceResponse)
async def detect_damaged_trees(
    images_zip: UploadFile = File(..., description="추론할 이미지들이 포함된 ZIP 파일"),
    model_path: str = Form(default=DEFAULT_WEIGHTS, description="사용할 YOLO 모델 경로"),
    confidence: float = Form(default=config.DEFAULT_CONFIDENCE, description="탐지 신뢰도 임계값 (0.16)"),
    iou_threshold: float = Form(default=config.DEFAULT_IOU_THRESHOLD, description="IoU 임계값 (중복 탐지 제거용)"),
    save_visualization: bool = Form(default=True, description="탐지 결과 시각화 이미지 저장 여부"),
    output_tm_coordinates: bool = Form(default=True, description="TM 좌표 변환 여부")
):
    """
    🚀 **소나무재선충병 피해목 탐지 API**
    
    ZIP 파일로 업로드된 이미지들(또는 타일들)을 일괄 처리하여 피해목을 탐지합니다.
    
    **🎯 주요 기능:**
    - 🖼️ 대용량 이미지 타일 일괄 처리
    - 🔍 YOLO 모델 기반 피해목 자동 탐지
    - 🗺️ TM 좌표 변환 (지리참조 정보 포함 시)
    - 📊 CSV 결과 파일 생성
    - 🎨 시각화 이미지 생성 (바운딩 박스 포함)
    - 🖼️ 전체 이미지 병합 시각화
    
    **📋 매개변수:**
    - **images_zip**: 추론할 이미지들이 포함된 ZIP 파일 (.zip)
    - **model_path**: 사용할 YOLO 모델 파일 경로 (기본: 최적화된 소나무 모델)
    - **confidence**: 탐지 신뢰도 임계값 (0.0-1.0, 기본값: 0.28)
    - **iou_threshold**: IoU 임계값 (0.0-1.0, 중복 탐지 제거용, 기본값: 0.6)
    - **save_visualization**: 탐지 결과 시각화 이미지 저장 여부 (기본: True)
    - **output_tm_coordinates**: TM 좌표 변환 출력 여부 (기본: True)
    
    **🎯 출력:**
    - 📊 탐지 통계 정보
    - 📁 결과 ZIP 파일 (CSV + 시각화 이미지들)
    - 🖼️ 병합된 전체 이미지 시각화
    - 🗺️ TM 좌표가 포함된 CSV 파일
    """
    
    try:
        # 파일명 검증
        if not images_zip.filename:
            raise HTTPException(status_code=400, detail="업로드된 파일에 파일명이 없습니다.")
        
        # ZIP 파일 확장자 검증
        if not images_zip.filename.lower().endswith('.zip'):
            raise HTTPException(status_code=400, detail="ZIP 파일만 업로드 가능합니다.")
        
        # 모델 경로 검증
        if not os.path.exists(model_path):
            raise HTTPException(status_code=400, detail=f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        # YOLO 모델 로드 (디버깅 정보 추가)
        print(f"🤖 모델 로딩: {model_path}")
        model = YOLO(model_path)
        
        # 모델 정보 출력
        print(f"  📋 모델 정보:")
        print(f"    - 클래스 수: {len(model.names) if hasattr(model, 'names') else 'Unknown'}")
        print(f"    - 클래스 명: {model.names if hasattr(model, 'names') else 'Unknown'}")
        print(f"    - 모델 크기: {os.path.getsize(model_path) / 1024 / 1024:.1f}MB")
        
        # 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # ZIP 파일 저장 및 압축 해제
            zip_path = os.path.join(temp_dir, images_zip.filename)
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(images_zip.file, f)
            
            # ZIP 파일 압축 해제
            extract_dir = os.path.join(temp_dir, "extracted")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # 이미지 파일 찾기 (macOS 숨김 파일 제외)
            image_files = []
            for root, dirs, files in os.walk(extract_dir):
                # macOS 시스템 폴더 제외
                if '__MACOSX' in root:
                    continue
                for file in files:
                    # macOS 숨김 파일 및 시스템 파일 제외
                    if file.startswith('.') or file.startswith('._'):
                        continue
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                        image_files.append(os.path.join(root, file))
            
            if not image_files:
                raise HTTPException(status_code=400, detail="ZIP 파일에서 이미지 파일을 찾을 수 없습니다.")
            
            # 결과 저장용 디렉토리 생성
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base = os.path.join(DEFAULT_OUTPUT_DIR, f"batch_inference_{timestamp}")
            
            # 기본 출력 디렉토리가 없으면 생성
            os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
            os.makedirs(output_base, exist_ok=True)
            
            viz_dir = None
            if save_visualization:
                viz_dir = os.path.join(output_base, "visualizations")
                os.makedirs(viz_dir, exist_ok=True)
            
            # 배치 추론 실행
            all_results = []
            processed_images = []
            failed_images = []
            
            print(f"📊 배치 추론 시작: {len(image_files)}개 이미지 처리 예정")
            
            for idx, image_path in enumerate(image_files, 1):
                try:
                    image_name = os.path.basename(image_path)
                    print(f"🔍 처리 중 ({idx}/{len(image_files)}): {image_name}")
                    
                    # 🎯 촘촘한 피해목 특화 최적화된 3단계 추론
                    print(f"  🔍 촘촘한 피해목 특화 추론 시작...")
                    
                    # 1단계: 표준 탐지 (일반적인 피해목)
                    print(f"    📊 640px 표준: conf=0.15, iou=0.35")
                    results_640 = model(image_path, imgsz=640, conf=0.15, iou=0.35)
                    
                    # 2단계: 고해상도 중밀도 (작은 피해목)
                    print(f"    📊 832px 고해상도: conf=0.13, iou=0.2")  
                    results_832 = model(image_path, imgsz=832, conf=0.13, iou=0.2)
                    
                    # 3단계: 초고해상도 초밀집 (촘촘한 피해목 전용) ⭐ 더욱 강화!
                    print(f"    📊 1024px 초밀집: conf=0.10, iou=0.05 (극한 밀집)")
                    results_1024 = model(image_path, imgsz=1024, conf=0.10, iou=0.05)
                    
                    # 기존 변수명 호환성 유지 (결과 통합)
                    primary_results = results_640
                    dense_results = results_832  
                    fine_results = results_1024  # 촘촘한 지역용
                    
                    # TFW 정보 추출 (TM 좌표 변환용)
                    tfw_params = None
                    if output_tm_coordinates:
                        tfw_params = get_tfw_from_tiff(image_path)
                        if not tfw_params:
                            # 동일 디렉토리에서 TFW 파일 찾기
                            tfw_file_path = image_path.replace('.tif', '.tfw').replace('.tiff', '.tfw')
                            if os.path.exists(tfw_file_path):
                                tfw_params = load_tfw_file(tfw_file_path)
                    
                    # YOLOv11s 내장 NMS만 사용 (추가 NMS 불필요)
                    
                    # 🧩 3단계 결과 통합 및 지능형 중복 제거
                    all_detections_raw = []
                    
                    # 1단계 결과 수집 (높은 우선순위)
                    stage1_count = 0
                    for result in primary_results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                                conf = float(box.conf[0].cpu().numpy())
                                all_detections_raw.append({
                                    'center_x': center_x, 'center_y': center_y, 'conf': conf,
                                    'box': box, 'stage': 1, 'priority': 3
                                })
                                stage1_count += 1
                    
                    # 2단계 결과 수집 (중간 우선순위)
                    stage2_count = 0
                    for result in dense_results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                                conf = float(box.conf[0].cpu().numpy())
                                all_detections_raw.append({
                                    'center_x': center_x, 'center_y': center_y, 'conf': conf,
                                    'box': box, 'stage': 2, 'priority': 2
                                })
                                stage2_count += 1
                    
                    # 3단계 결과 수집 (낮은 우선순위)
                    stage3_count = 0
                    for result in fine_results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                                conf = float(box.conf[0].cpu().numpy())
                                all_detections_raw.append({
                                    'center_x': center_x, 'center_y': center_y, 'conf': conf,
                                    'box': box, 'stage': 3, 'priority': 1
                                })
                                stage3_count += 1
                    
                    print(f"  📊 단계별 원시 탐지: 1단계={stage1_count}개, 2단계={stage2_count}개, 3단계={stage3_count}개")
                    
                    # 🎯 지능형 중복 제거 (우선순위 + 거리 기반)
                    # 높은 우선순위(stage1) > 높은 신뢰도 > 낮은 stage 순으로 정렬
                    all_detections_raw.sort(key=lambda x: (x['priority'], x['conf']), reverse=True)
                    
                    filtered_detections = []
                    MIN_DISTANCE = 15  # 15픽셀 이내는 중복 (기존 25에서 줄임)
                    
                    for detection in all_detections_raw:
                        is_duplicate = False
                        for existing in filtered_detections:
                            dist = ((detection['center_x'] - existing['center_x']) ** 2 + 
                                   (detection['center_y'] - existing['center_y']) ** 2) ** 0.5
                            if dist < MIN_DISTANCE:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            filtered_detections.append(detection)
                    
                    print(f"  🧹 중복 제거 후: {len(filtered_detections)}개 (거리 {MIN_DISTANCE}px 기준)")
                    
                    # 결과 처리 및 밀도 기반 동적 바운딩박스 크기 최적화
                    image_results = []
                    all_boxes = [(det['center_x'], det['center_y'], det['conf']) for det in filtered_detections]
                        # 각 탐지에 대해 밀도 기반 크기 조정
                    for i, detection_info in enumerate(filtered_detections):
                        center_x = detection_info['center_x']
                        center_y = detection_info['center_y']
                        box = detection_info['box']
                        stage = detection_info['stage']
                        
                        conf_value = detection_info['conf']
                        print(f"    탐지 {i+1}: 신뢰도 {conf_value:.4f} (단계{stage})")
                        
                        # 원본 바운딩 박스 정보
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        orig_width = x2 - x1
                        orig_height = y2 - y1
                        
                        # 🌲 주변 밀도 기반 동적 크기 조정
                        SEARCH_RADIUS = 80  # 80픽셀 반경 내 밀도 확인
                        nearby_count = 0
                        
                        # 현재 탐지 주변의 다른 탐지들 개수 세기
                        for other_x, other_y, _ in all_boxes:
                            if other_x != center_x or other_y != center_y:  # 자기 자신 제외
                                distance = ((center_x - other_x) ** 2 + (center_y - other_y) ** 2) ** 0.5
                                if distance <= SEARCH_RADIUS:
                                    nearby_count += 1
                        
                        # 🎯 실용적 접근: YOLO 예측 크기 그대로 사용 (시각화만 개선)
                        # 실제 탐지 성능은 멀티스케일 해상도가 담당
                        
                        # 밀도 레벨 계산 (시각화용)
                        if nearby_count >= 5:
                            density_level = "매우촘촘"
                        elif nearby_count >= 3:
                            density_level = "촘촘"
                        elif nearby_count >= 1:
                            density_level = "보통"
                        else:
                            density_level = "외딴"
                        
                        print(f"      🎯 주변 밀도: {nearby_count}개 → 밀도 레벨: {density_level}")
                        print(f"      📏 YOLO 예측 크기: {orig_width:.1f}x{orig_height:.1f}px")
                        
                        # 🌲 소나무 피해목 실용적 크기: 밀도 기반 적응적 제한
                        if density_level == "많이 밀집된 곳":
                            MAX_SIZE = 20   # 촘촘한 지역은 작게
                            TARGET_SIZE = 16
                        elif density_level == "적당히 밀집된 곳":
                            MAX_SIZE = 30   # 적당히 작게
                            TARGET_SIZE = 22
                        elif density_level == "보통":
                            MAX_SIZE = 40   # 보통 크기
                            TARGET_SIZE = 28
                        else:  # 외딴
                            MAX_SIZE = 50   # 외딴 지역은 크게 (명확 표시)
                            TARGET_SIZE = 35
                            
                        MIN_SIZE = 8    # 최소 크기 보장
                        
                        # 극단적인 경우만 제한, 나머지는 YOLO 예측 그대로 사용
                        if orig_width > MAX_SIZE or orig_height > MAX_SIZE:
                            # 너무 큰 경우만 제한 (비율 유지하며 축소)
                            scale_factor = min(MAX_SIZE / orig_width, MAX_SIZE / orig_height)
                            new_width = orig_width * scale_factor
                            new_height = orig_height * scale_factor
                            print(f"      📐 대형목 크기 조정: {orig_width:.1f}x{orig_height:.1f} → {new_width:.1f}x{new_height:.1f}px")
                        elif orig_width < MIN_SIZE or orig_height < MIN_SIZE:
                            # 너무 작은 경우만 최소 크기 보장
                            new_width = max(orig_width, MIN_SIZE)
                            new_height = max(orig_height, MIN_SIZE)
                            print(f"      📐 최소 크기 보장: {orig_width:.1f}x{orig_height:.1f} → {new_width:.1f}x{new_height:.1f}px")
                        else:
                            # TARGET_SIZE 기준으로 적절히 조정
                            avg_size = (orig_width + orig_height) / 2
                            if avg_size > TARGET_SIZE * 1.2:  # 20% 이상 크면 조정
                                scale = TARGET_SIZE / avg_size
                                new_width = orig_width * scale
                                new_height = orig_height * scale
                                print(f"      📐 적정 크기 조정: {orig_width:.1f}x{orig_height:.1f} → {new_width:.1f}x{new_height:.1f}px")
                            else:
                                # 적정 범위는 원본 사용
                                new_width = orig_width
                                new_height = orig_height
                                print(f"      ✅ 적정 크기: {new_width:.1f}x{new_height:.1f}px")
                        
                        conf = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # TM 좌표 변환
                        tm_x, tm_y = None, None
                        if tfw_params:
                            tm_x, tm_y = pixel_to_tm(center_x, center_y, tfw_params)
                        
                        detection = DetectionResult(
                            filename=image_name,
                            class_id=class_id,
                            center_x=center_x,
                            center_y=center_y,
                            width=new_width,  # 밀도 기반 조정된 크기
                            height=new_height,  # 밀도 기반 조정된 크기
                            confidence=conf,
                            tm_x=tm_x,
                            tm_y=tm_y
                        )
                        
                        image_results.append(detection)
                        all_results.append(detection)
                    
                    processed_images.append(image_name)
                    raw_detections = stage1_count + stage2_count + stage3_count
                    print(f"✅ 완료: {image_name} (원시: {raw_detections}개, 최종: {len(image_results)}개 탐지)")
                    
                    # 시각화 이미지 저장 (동적 바운딩박스 크기 반영)
                    if save_visualization and viz_dir and image_results:
                        viz_filename = f"{os.path.splitext(image_name)[0]}_detected_{timestamp}.jpg"
                        viz_path = os.path.join(viz_dir, viz_filename)
                        try:
                            # 원본 이미지 로드
                            image = cv2.imread(image_path)
                            if image is not None:
                                # 탐지 결과 그리기 (동적 크기 반영)
                                for detection in image_results:
                                    x1 = int(detection.center_x - detection.width / 2)
                                    y1 = int(detection.center_y - detection.height / 2)
                                    x2 = int(detection.center_x + detection.width / 2)
                                    y2 = int(detection.center_y + detection.height / 2)
                                    
                                    # 신뢰도에 따른 색상 조정
                                    if detection.confidence >= 0.7:
                                        bbox_color = (0, 255, 0)  # 녹색 (높은 신뢰도)
                                    elif detection.confidence >= 0.4:
                                        bbox_color = (0, 165, 255)  # 주황색 (중간 신뢰도)
                                    else:
                                        bbox_color = (0, 0, 255)  # 빨간색 (낮은 신뢰도)
                                    
                                    # 바운딩 박스 그리기 (크기에 따른 선 두께 조정)
                                    thickness = 2 if detection.width >= 24 else 1
                                    cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, thickness)
                                    
                                    # 신뢰도 텍스트 제거 (시인성 향상을 위해)
                                
                                # 시각화 이미지 저장
                                cv2.imwrite(viz_path, image)
                                print(f"  🖼️ 시각화 저장: {viz_filename}")
                        except Exception as e:
                            print(f"⚠️ 시각화 저장 실패 ({image_name}): {e}")
                    elif save_visualization and viz_dir:
                        # 탐지 결과가 없어도 원본 이미지 저장 (회색 테두리로 표시)
                        original_img = cv2.imread(image_path)
                        if original_img is not None:
                            h, w = original_img.shape[:2]
                            cv2.rectangle(original_img, (0, 0), (w-1, h-1), (128, 128, 128), 10)
                            
                            viz_filename = f"{os.path.splitext(image_name)[0]}_no_detection_{timestamp}.jpg"
                            viz_path = os.path.join(viz_dir, viz_filename)
                            cv2.imwrite(viz_path, original_img)
                            print(f"  🖼️ 탐지없음 시각화 저장: {viz_filename}")
                    
                    # 추론 결과 메모리 해제
                    del primary_results, dense_results, fine_results
                    
                    # 탐지가 매우 적으면 디버그 정보 출력
                    if len(image_results) < 3:
                        print(f"  🔍 탐지 수가 적음 - 극한 디버그")
                        # 극한 낮은 신뢰도로 재시도
                        extreme_results = model(image_path, conf=0.01, iou=0.1)
                        extreme_count = sum(len(r.boxes) if r.boxes is not None else 0 for r in extreme_results)
                        print(f"  🔍 극한 디버그 (conf=0.01, iou=0.1): {extreme_count}개 탐지")
                        del extreme_results
                    
                except Exception as e:
                    failed_images.append(f"{os.path.basename(image_path)}: {str(e)}")
                    print(f"❌ 실패: {os.path.basename(image_path)} - {e}")
            
            # CSV 파일 생성
            csv_file_url = None
            if all_results:
                csv_filename = f"batch_detections_{timestamp}.csv"
                csv_path = os.path.join(output_base, csv_filename)
                
                # 탐지 결과를 DataFrame으로 변환
                results_data = []
                for idx, detection in enumerate(all_results, 1):
                    row = {
                        'no': idx,
                        'filename': detection.filename,
                        'class_id': detection.class_id,
                        'center_x': detection.center_x,
                        'center_y': detection.center_y,
                        'width': detection.width,
                        'height': detection.height,
                        'confidence': detection.confidence
                    }
                    
                    # TM 좌표가 있는 경우 추가
                    if detection.tm_x is not None and detection.tm_y is not None:
                        row['tm_x'] = detection.tm_x
                        row['tm_y'] = detection.tm_y
                    
                    results_data.append(row)
                
                # CSV 저장
                df = pd.DataFrame(results_data)
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                csv_file_url = f"/api/v1/inference/download/{csv_filename}"
            
            # 합쳐진 시각화 이미지 생성
            merged_viz_filename = None
            if save_visualization and all_results:
                merged_viz_filename = create_merged_visualization(
                    all_results, output_base, timestamp, extract_dir, tile_size=config.DEFAULT_TILE_SIZE
                )
            
            # 결과 ZIP 파일 생성
            results_zip_url = None
            if all_results or (save_visualization and viz_dir and os.path.exists(viz_dir) and os.listdir(viz_dir)):
                zip_filename = f"batch_results_{timestamp}.zip"
                zip_path = os.path.join(DEFAULT_OUTPUT_DIR, zip_filename)
                
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # CSV 파일 추가
                    if all_results:
                        csv_path = os.path.join(output_base, f"batch_detections_{timestamp}.csv")
                        if os.path.exists(csv_path):
                            zipf.write(csv_path, f"batch_detections_{timestamp}.csv")
                    
                    # 시각화 이미지들 추가
                    if save_visualization and viz_dir and os.path.exists(viz_dir):
                        for file in os.listdir(viz_dir):
                            file_path = os.path.join(viz_dir, file)
                            if os.path.isfile(file_path):  # 파일인지 확인
                                zipf.write(file_path, f"visualizations/{file}")
                    
                    # 합쳐진 시각화 이미지 추가
                    if merged_viz_filename:
                        merged_path = os.path.join(output_base, merged_viz_filename)
                        if os.path.exists(merged_path):
                            zipf.write(merged_path, merged_viz_filename)
                
                results_zip_url = f"/api/v1/inference/download/{zip_filename}"
            
            return BatchInferenceResponse(
                success=True,
                message=f"배치 추론 완료: {len(processed_images)}개 이미지 처리, {len(all_results)}개 객체 탐지",
                total_images=len(image_files),
                total_detections=len(all_results),
                processed_images=processed_images,
                failed_images=failed_images,
                csv_file_url=csv_file_url,
                results_zip_url=results_zip_url,
                merged_visualization=merged_viz_filename
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"배치 추론 중 오류 발생: {str(e)}")
