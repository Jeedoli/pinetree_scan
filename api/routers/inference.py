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

class InferenceResponse(BaseModel):
    success: bool
    message: str
    detected_count: int
    results: List[DetectionResult]
    csv_file_url: Optional[str] = None
    visualization_images: Optional[List[str]] = None  # 시각화 이미지 URL 리스트

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
        # A20250917_tile_2_1.tif -> ['A20250917', 'tile', '2', '1', 'tif']
        parts = filename.replace('.tif', '').replace('.tiff', '').split('_')
        if len(parts) >= 4 and parts[-3] == 'tile':
            x = int(parts[-2])
            y = int(parts[-1])
            return (x, y)
    except (ValueError, IndexError):
        pass
    return (0, 0)


def create_merged_visualization(all_results: List[DetectionResult], output_base: str, 
                               timestamp: str, extract_dir: str, tile_size: int = 1024) -> Optional[str]:
    """타일별 탐지 결과를 합쳐서 전체 이미지 시각화 생성"""
    
    if not all_results:
        return None
    
    try:
        print("🖼️ 합쳐진 시각화 이미지 생성 시작...")
        
        # 타일 위치 정보 수집
        tile_positions = {}
        tile_files = {}  # 타일 파일 경로 저장
        
        for result in all_results:
            x, y = extract_tile_position(result.filename)
            if (x, y) not in tile_positions:
                tile_positions[(x, y)] = []
                # 타일 파일 경로 찾기
                tile_path = None
                for root, dirs, files in os.walk(extract_dir):
                    if result.filename in files:
                        tile_path = os.path.join(root, result.filename)
                        break
                tile_files[(x, y)] = tile_path
            tile_positions[(x, y)].append(result)
        
        if not tile_positions:
            return None
        
        # 전체 이미지 크기 계산
        max_x = max(pos[0] for pos in tile_positions.keys())
        max_y = max(pos[1] for pos in tile_positions.keys())
        
        # 실제 타일들을 로드해서 정확한 크기 계산
        total_width = 0
        total_height = 0
        tile_dimensions = {}  # 각 타일의 실제 크기 저장
        
        # 첫 번째 행의 모든 타일로 전체 너비 계산
        for tx in range(max_x + 1):
            if (tx, 0) in tile_files and tile_files[(tx, 0)]:
                try:
                    tile_img = cv2.imread(tile_files[(tx, 0)])
                    if tile_img is not None:
                        h, w = tile_img.shape[:2]
                        tile_dimensions[(tx, 0)] = (w, h)
                        total_width += w
                    else:
                        tile_dimensions[(tx, 0)] = (tile_size, tile_size)
                        total_width += tile_size
                except:
                    tile_dimensions[(tx, 0)] = (tile_size, tile_size)
                    total_width += tile_size
            else:
                tile_dimensions[(tx, 0)] = (tile_size, tile_size)
                total_width += tile_size
        
        # 첫 번째 열의 모든 타일로 전체 높이 계산
        for ty in range(max_y + 1):
            if (0, ty) in tile_files and tile_files[(0, ty)]:
                try:
                    tile_img = cv2.imread(tile_files[(0, ty)])
                    if tile_img is not None:
                        h, w = tile_img.shape[:2]
                        tile_dimensions[(0, ty)] = (w, h)
                        total_height += h
                    else:
                        tile_dimensions[(0, ty)] = (tile_size, tile_size)
                        total_height += tile_size
                except:
                    tile_dimensions[(0, ty)] = (tile_size, tile_size)
                    total_height += tile_size
            else:
                tile_dimensions[(0, ty)] = (tile_size, tile_size)
                total_height += tile_size
        
        print(f"📐 계산된 전체 이미지 크기: {total_width}x{total_height}")
        
        # 전체 이미지 생성 (검은색 배경으로 시작)
        merged_image = np.zeros((total_height, total_width, 3), dtype=np.uint8)
        
        # 타일들을 순서대로 합치기
        current_y = 0
        for ty in range(max_y + 1):
            current_x = 0
            row_height = 0
            
            for tx in range(max_x + 1):
                tile_path = tile_files.get((tx, ty))
                
                if tile_path and os.path.exists(tile_path):
                    try:
                        # 타일 이미지 로드
                        tile_img = cv2.imread(tile_path)
                        if tile_img is not None:
                            h, w = tile_img.shape[:2]
                            
                            # 타일을 전체 이미지에 배치
                            end_x = min(current_x + w, total_width)
                            end_y = min(current_y + h, total_height)
                            
                            actual_w = end_x - current_x
                            actual_h = end_y - current_y
                            
                            if actual_w > 0 and actual_h > 0:
                                merged_image[current_y:end_y, current_x:end_x] = tile_img[:actual_h, :actual_w]
                            
                            row_height = max(row_height, h)
                            current_x += w
                        else:
                            current_x += tile_size
                            row_height = max(row_height, tile_size)
                    except Exception as e:
                        print(f"⚠️ 타일 로드 실패 {tile_path}: {e}")
                        current_x += tile_size
                        row_height = max(row_height, tile_size)
                else:
                    current_x += tile_size
                    row_height = max(row_height, tile_size)
            
            current_y += row_height
        
        print("🎯 타일 합치기 완료, 탐지 결과 그리기 시작...")
        
        # 탐지 결과 그리기
        current_y_offset = 0
        for ty in range(max_y + 1):
            current_x_offset = 0
            row_height = tile_dimensions.get((0, ty), (tile_size, tile_size))[1]
            
            for tx in range(max_x + 1):
                tile_width = tile_dimensions.get((tx, 0), (tile_size, tile_size))[0]
                
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
                        
                        # 바운딩 박스 그리기 (얇은 초록색)
                        cv2.rectangle(merged_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 신뢰도 텍스트 추가
                        if detection.confidence:
                            text = f"{detection.confidence:.2f}"
                            # 텍스트 배경 추가 (가독성 향상)
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                            cv2.rectangle(merged_image, (x1, y1-25), (x1 + text_size[0] + 10, y1), (0, 255, 0), -1)
                            cv2.putText(merged_image, text, (x1 + 5, y1-8), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                
                current_x_offset += tile_width
            
            current_y_offset += row_height
        
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
                
                # 크기 축소 비율 (85% 크기로 축소)
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

@router.post("/detect", response_model=InferenceResponse)
async def detect_damaged_trees(
    files: List[UploadFile] = File(..., description="탐지할 이미지 파일들 (TIFF, JPG, PNG)"),
    weights: str = Form(default=DEFAULT_WEIGHTS, description="YOLO 모델 가중치 경로"),
    confidence: float = Form(default=0.3, description="신뢰도 임계값 (0.0~1.0) - 소나무 전용 최적화"),
    iou_threshold: float = Form(default=0.6, description="IOU 임계값 (0.0~1.0) - 오탐지 방지"),
    save_csv: bool = Form(default=True, description="CSV 파일로 결과 저장 여부"),
    save_visualization: bool = Form(default=True, description="바운딩 박스가 그려진 시각화 이미지 저장 여부")
):
    """
    업로드된 이미지에서 소나무재선충병 피해목을 탐지합니다.
    
    - **files**: 탐지할 이미지 파일들 (TIFF, JPG, PNG 지원)
    - **weights**: YOLO 모델 가중치 파일 경로
    - **confidence**: 탐지 신뢰도 임계값 (기본: 0.05)
    - **iou_threshold**: IOU 임계값 (기본: 0.25)
    - **save_csv**: CSV 파일로 결과 저장 여부 (기본: True)
    - **save_visualization**: 바운딩 박스가 그려진 시각화 이미지 저장 여부 (기본: True)
    """
    
    try:
        # 모델 파일 존재 확인
        if not os.path.exists(weights):
            raise HTTPException(status_code=404, detail=f"모델 파일을 찾을 수 없습니다: {weights}")
        
        # YOLO 모델 로드
        model = YOLO(weights)
        
        # 결과 저장용 리스트
        all_results = []
        damaged_count = 0
        visualization_urls = []  # 시각화 이미지 URL 저장용
        
        # 세션 타임스탬프 생성 (파일명 일관성을 위해)
        session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # 각 파일 처리
            for file in files:
                # 파일 확장자 검증
                if not file.filename.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
                    raise HTTPException(
                        status_code=400, 
                        detail=f"지원하지 않는 파일 형식입니다: {file.filename}"
                    )
                
                # 임시 파일로 저장
                temp_file_path = os.path.join(temp_dir, file.filename)
                with open(temp_file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # 이미지 전처리 (4채널 -> 3채널 변환)
                img = cv2.imread(temp_file_path, cv2.IMREAD_UNCHANGED)
                if img is not None and len(img.shape) == 3 and img.shape[2] == 4:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    cv2.imwrite(temp_file_path, img_rgb)
                    img = img_rgb
                
                # 🗺️ TM 좌표 변환을 위한 지리참조 정보 추출
                tfw_params = None
                if file.filename.lower().endswith(('.tif', '.tiff')):
                    # 1. TIFF 파일에서 직접 지리참조 정보 추출 시도
                    tfw_params = get_tfw_from_tiff(temp_file_path)
                    
                    # 2. TFW 파일이 있는지 확인 (동일한 이름)
                    if not tfw_params:
                        tfw_path = temp_file_path.replace('.tif', '.tfw').replace('.tiff', '.tfw')
                        if os.path.exists(tfw_path):
                            tfw_params = load_tfw_file(tfw_path)
                
                # YOLO 추론 실행
                yolo_results = model(temp_file_path, conf=confidence, iou=iou_threshold)
                
                # 결과 처리
                for r in yolo_results:
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        conf_score = float(box.conf[0])
                        
                        # 피해목 클래스(class_id=0)만 처리
                        if class_id == 0:
                            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                            
                            # 정규화된 좌표 계산
                            x_center = (xmin + xmax) / 2 / img.shape[1]
                            y_center = (ymin + ymax) / 2 / img.shape[0]
                            width = (xmax - xmin) / img.shape[1]
                            height = (ymax - ymin) / img.shape[0]
                            
                            # 🗺️ TM 좌표 계산 (지리참조 정보가 있는 경우)
                            tm_x, tm_y = None, None
                            if tfw_params:
                                # 픽셀 좌표 (박스 중심점)
                                center_px = (xmin + xmax) / 2
                                center_py = (ymin + ymax) / 2
                                
                                # 픽셀 좌표를 TM 좌표로 변환
                                tm_x, tm_y = pixel_to_tm(center_px, center_py, tfw_params)
                            
                            result = DetectionResult(
                                filename=file.filename,
                                class_id=class_id,
                                center_x=x_center,
                                center_y=y_center,
                                width=width,
                                height=height,
                                confidence=conf_score,
                                tm_x=tm_x,  # TM 좌표 X
                                tm_y=tm_y   # TM 좌표 Y
                            )
                            all_results.append(result)
                            damaged_count += 1
                
                # 시각화 이미지 생성 및 저장
                if save_visualization and yolo_results[0].boxes is not None and len(yolo_results[0].boxes) > 0:
                    # 탐지된 객체가 있는 경우에만 시각화
                    img_copy = img.copy()
                    draw_bounding_boxes_on_image(img_copy, yolo_results)
                    
                    # 시각화 디렉토리 생성
                    vis_dir = str(config.API_VISUALIZATION_DIR)
                    os.makedirs(vis_dir, exist_ok=True)
                    
                    # 시각화 이미지 파일명 생성 (세션 타임스탬프 사용)
                    name_without_ext = os.path.splitext(file.filename)[0]
                    vis_filename = f"{name_without_ext}_detected_{session_timestamp}.jpg"
                    vis_path = os.path.join(vis_dir, vis_filename)
                    
                    # 시각화 이미지 저장
                    cv2.imwrite(vis_path, img_copy)
                    
                    # URL 생성
                    vis_url = f"/api/v1/inference/visualization/{vis_filename}"
                    visualization_urls.append(vis_url)
        
        # CSV 저장 처리
        csv_file_url = None
        if save_csv and all_results:
            # 출력 디렉토리 생성
            os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
            
            # CSV 파일명 생성 (세션 타임스탬프 사용)
            csv_filename = f"detection_results_{session_timestamp}.csv"
            csv_path = os.path.join(DEFAULT_OUTPUT_DIR, csv_filename)
            
            # 🗺️ TM 좌표가 있는 경우 별도 CSV 생성
            tm_results = [r for r in all_results if r.tm_x is not None and r.tm_y is not None]
            
            if tm_results:
                # TM 좌표 CSV 저장 (첨부해주신 형식과 동일)
                tm_csv_filename = f"damaged_trees_tm_coords_{session_timestamp}.csv"
                tm_csv_path = os.path.join(DEFAULT_OUTPUT_DIR, tm_csv_filename)
                
                # TM 좌표 DataFrame 생성 (no, x, y 형식)
                tm_data = []
                for i, result in enumerate(tm_results, 1):
                    tm_data.append({
                        'no': i,
                        'x': round(result.tm_x, 3),  # TM X 좌표 (소수점 3자리)
                        'y': round(result.tm_y, 3)   # TM Y 좌표 (소수점 3자리)
                    })
                
                df_tm = pd.DataFrame(tm_data)
                df_tm.to_csv(tm_csv_path, index=False)
                print(f"🗺️ TM 좌표 CSV 저장: {tm_csv_path}")
                print(f"📊 TM 좌표 변환 성공: {len(tm_results)}개/{len(all_results)}개")
            
            # 기존 전체 정보 CSV 저장 (정규화 좌표 + TM 좌표 모두 포함)
            df_results = pd.DataFrame([result.dict() for result in all_results])
            df_results.to_csv(csv_path, index=False)
            csv_file_url = f"/api/v1/inference/download/{csv_filename}"
        
        # 성공 메시지 생성
        tm_count = len([r for r in all_results if r.tm_x is not None and r.tm_y is not None])
        if tm_count > 0:
            message = f"탐지 완료: {damaged_count}개의 피해목을 발견했습니다. (TM 좌표 변환: {tm_count}개)"
        else:
            message = f"탐지 완료: {damaged_count}개의 피해목을 발견했습니다. (지리참조 정보 없음)"
        
        return InferenceResponse(
            success=True,
            message=message,
            detected_count=damaged_count,
            results=all_results,
            csv_file_url=csv_file_url,
            visualization_images=visualization_urls if visualization_urls else None
        )
        
    except Exception as e:
        import traceback
        error_detail = f"탐지 중 오류 발생: {str(e)}\n상세: {traceback.format_exc()}"
        print(f"API ERROR: {error_detail}")  # 서버 로그에 출력
        raise HTTPException(status_code=500, detail=f"탐지 중 오류 발생: {str(e)}")


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


@router.post("/detect-all", response_model=BatchInferenceResponse)
async def batch_inference(
    images_zip: UploadFile = File(..., description="추론할 이미지들이 포함된 ZIP 파일"),
    model_path: str = Form(default=DEFAULT_WEIGHTS, description="사용할 YOLO 모델 경로"),
    confidence: float = Form(default=0.5, description="탐지 신뢰도 임계값"),
    iou_threshold: float = Form(default=0.8, description="IoU 임계값 (중복 탐지 제거용)"),
    save_visualization: bool = Form(default=True, description="탐지 결과 시각화 이미지 저장 여부"),
    output_tm_coordinates: bool = Form(default=True, description="TM 좌표 변환 여부")
):
    """
    여러 이미지에 대한 배치 추론을 수행합니다.
    ZIP 파일로 업로드된 이미지들을 일괄 처리하여 탐지 결과를 반환합니다.
    
    - **images_zip**: 추론할 이미지들이 포함된 ZIP 파일
    - **model_path**: 사용할 YOLO 모델 파일 경로
    - **confidence**: 탐지 신뢰도 임계값 (0.0-1.0, 기본값: 0.5)
    - **iou_threshold**: IoU 임계값 (0.0-1.0, 중복 탐지 제거용, 기본값: 0.8)
    - **save_visualization**: 탐지 결과 시각화 이미지 저장 여부
    - **output_tm_coordinates**: TM 좌표 변환 출력 여부
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
        
        # YOLO 모델 로드
        model = YOLO(model_path)
        
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
            
            # 이미지 파일 찾기
            image_files = []
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
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
                    
                    # 개별 이미지 추론
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
                    
                    # 결과 처리
                    image_results = []
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
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
                    print(f"✅ 완료: {image_name} ({len(image_results)}개 탐지)")
                    
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
                    all_results, output_base, timestamp, extract_dir, tile_size=1024
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
