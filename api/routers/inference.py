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
                            # 바운딩 박스 그리기 (두꺼운 초록색)
                            cv2.rectangle(merged_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            
                            # 신뢰도 텍스트 추가
                            if detection.confidence:
                                text = f"{detection.confidence:.3f}"
                                # 텍스트 배경 추가 (가독성 향상)
                                font_scale = 0.6
                                font_thickness = 2
                                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                                
                                # 텍스트 위치 조정 (바운딩 박스 위쪽)
                                text_y = max(y1 - 5, text_size[1] + 5)
                                bg_y1 = text_y - text_size[1] - 5
                                bg_y2 = text_y + 5
                                bg_x1 = x1
                                bg_x2 = min(x1 + text_size[0] + 10, total_width)
                                
                                # 배경 그리기
                                cv2.rectangle(merged_image, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 255, 0), -1)
                                # 텍스트 그리기
                                cv2.putText(merged_image, text, (x1 + 5, text_y), 
                                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
                            
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
                
                # 🌲 소나무 피해목 전용 바운딩 박스 크기 최적화
                # 바운딩 박스를 15% 축소하여 더 정확한 영역만 표시
                width = x2 - x1
                height = y2 - y1
                center_x = x1 + width / 2
                center_y = y1 + height / 2
                
                # 🎯 동적 바운딩 박스 크기 조정
                if config.DYNAMIC_BBOX_SIZING:
                    # 신뢰도와 크기에 따른 스케일 팩터 계산
                    confidence_factor = min(1.0, confidence * 2)  # 신뢰도가 높을수록 더 정확
                    
                    # 원본 크기에 따른 조정
                    box_area = width * height
                    img_area = img_width * img_height
                    area_ratio = box_area / img_area if img_area > 0 else 0
                    
                    if area_ratio > 0.1:  # 큰 객체 (10% 이상)
                        scale_factor = 0.75 * confidence_factor  # 더 축소
                    elif area_ratio > 0.01:  # 중간 객체 (1-10%)
                        scale_factor = 0.85 * confidence_factor  # 적당히 축소  
                    else:  # 작은 객체 (1% 미만)
                        scale_factor = 0.95 * confidence_factor  # 약간만 축소
                        
                    new_width = width * scale_factor
                    new_height = height * scale_factor
                else:
                    # 기존 고정 스케일 팩터 (85% 크기로 축소)
                    scale_factor = 0.85
                    new_width = width * scale_factor
                    new_height = height * scale_factor
                
                # 새로운 좌표 계산
                new_x1 = int(center_x - new_width / 2)
                new_y1 = int(center_y - new_height / 2)
                new_x2 = int(center_x + new_width / 2)
                new_y2 = int(center_y + new_height / 2)
                
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
                
                # 더 작은 폰트로 레이블 표시
                label = f"Pine Damage: {confidence:.2f}"
                font_scale = 0.5  # 폰트 크기 축소
                font_thickness = 1  # 폰트 두께 축소
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                           font_scale, font_thickness)[0]
                
                # 텍스트 배경 그리기 (더 작게)
                cv2.rectangle(image, (new_x1, new_y1 - label_size[1] - 5), 
                            (new_x1 + label_size[0], new_y1), 
                            bbox_color, -1)
                
                # 텍스트 그리기 (더 작게)
                cv2.putText(image, label, (new_x1, new_y1 - 3), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                          (255, 255, 255), font_thickness)

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
                    
                    # 🎯 YOLOv11s 내장 FPN 사용한 단일 추론
                    print(f"  � YOLOv11s FPN 추론: conf={confidence}, iou={iou_threshold}")
                    results = model(image_path, conf=confidence, iou=iou_threshold)
                    
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
                    
                    # 결과 처리 (디버깅 정보 추가)
                    image_results = []
                    raw_detections = 0  # 원시 탐지 수
                    
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            raw_detections += len(boxes)
                            print(f"  📊 원시 탐지 수: {len(boxes)}개 (conf >= {confidence})")
                            
                            for i, box in enumerate(boxes):
                                conf_value = float(box.conf[0].cpu().numpy())
                                print(f"    탐지 {i+1}: 신뢰도 {conf_value:.4f}")
                                # 바운딩 박스 정보 추출
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                width = x2 - x1
                                height = y2 - y1
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
                                    width=width,
                                    height=height,
                                    confidence=conf,
                                    tm_x=tm_x,
                                    tm_y=tm_y
                                )
                                
                                image_results.append(detection)
                                all_results.append(detection)
                    
                    # 시각화 이미지 저장
                    if save_visualization and viz_dir and image_results:
                        viz_path = os.path.join(viz_dir, f"detected_{image_name}")
                        try:
                            # 원본 이미지 로드
                            image = cv2.imread(image_path)
                            if image is not None:
                                # 탐지 결과 그리기
                                for detection in image_results:
                                    x1 = int(detection.center_x - detection.width / 2)
                                    y1 = int(detection.center_y - detection.height / 2)
                                    x2 = int(detection.center_x + detection.width / 2)
                                    y2 = int(detection.center_y + detection.height / 2)
                                    
                                    # 바운딩 박스 그리기
                                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                    
                                    # 신뢰도 텍스트 추가
                                    if detection.confidence:
                                        text = f"{detection.confidence:.2f}"
                                        cv2.putText(image, text, (x1, y1-10), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                
                                # 시각화 이미지 저장
                                cv2.imwrite(viz_path, image)
                                # 메모리 해제
                                del image
                        except Exception as e:
                            print(f"⚠️ 시각화 저장 실패 ({image_name}): {e}")
                    
                    # 추론 결과 메모리 해제
                    del results
                    
                    processed_images.append(image_name)
                    print(f"✅ 완료: {image_name} (원시: {raw_detections}개, 최종: {len(image_results)}개 탐지)")
                    
                    # 탐지가 없는 경우 추가 디버깅
                    if raw_detections == 0:
                        print(f"  ⚠️ 탐지 없음: {image_name} - 신뢰도 {confidence} 기준")
                        # 낮은 신뢰도로 재시도
                        debug_results = model(image_path, conf=0.01, iou=iou_threshold)
                        debug_count = sum(len(r.boxes) if r.boxes is not None else 0 for r in debug_results)
                        print(f"  🔍 디버그 (conf=0.01): {debug_count}개 탐지")
                        del debug_results
                    
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
