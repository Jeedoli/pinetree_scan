# 전처리 API 라우터 - tile_and_label.py 기반  
# 대용량 GeoTIFF 이미지를 타일로 분할 + YOLO 라벨 자동 생성

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
import datetime
import zipfile
import yaml
import random
import sys
import logging
from pathlib import Path
from api import config

router = APIRouter()# Pydantic 모델 정의

# 전처리 API 라우터 - tile_and_label.py 기반
# 대용량 GeoTIFF 이미지를 타일로 분할 + YOLO 라벨 자동 생성

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
import datetime
import zipfile
import yaml
import random
import sys
import logging
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from .. import config

router = APIRouter()

# 응답 모델 정의
class TileInfo(BaseModel):
    tile_name: str
    labels_count: int
    tile_size: tuple
    position: tuple  # (tx, ty)

class InferenceTileInfo(BaseModel):
    tile_name: str
    tile_size: tuple
    position: tuple  # (tx, ty)
    has_georeference: bool

class IntegratedTrainingResponse(BaseModel):
    success: bool
    message: str
    preprocessing_info: Dict
    training_dataset_info: Dict
    download_url: str
    zip_filename: str
    validation_samples_url: Optional[str] = None  # 라벨링 검증 샘플 ZIP URL
    validation_info: Optional[Dict] = None  # 검증 정보
    coordinate_validation: Optional[Dict] = None  # 🔍 GPS 좌표 매핑 검증 정보

class InferenceTilingResponse(BaseModel):
    success: bool
    message: str
    total_tiles: int
    original_size: tuple
    tile_size: int
    tile_info: List[InferenceTileInfo]
    download_url: Optional[str] = None

class MergedDatasetResponse(BaseModel):
    success: bool
    message: str
    merged_dataset_info: Dict
    download_url: str
    zip_filename: str
    validation_samples_url: Optional[str] = None
    source_datasets: List[Dict]  # 원본 ZIP들의 정보

# 기본 설정
DEFAULT_TILE_SIZE = config.DEFAULT_TILE_SIZE
DEFAULT_CLASS_ID = config.DEFAULT_CLASS_ID
DEFAULT_OUTPUT_DIR = Path(config.API_TILES_DIR)  # Path 객체로 변경

# 📦 멀티스케일 바운딩박스 - 설정값은 config.py에서 관리

# 📊 라벨링 검증을 위한 시각화 함수들
def create_labeling_validation_samples(tiles_images_dir: Path, tiles_labels_dir: Path, 
                                      output_dir: Path, num_samples: int = 30) -> List[str]:
    """
    🔍 라벨링 품질 검증을 위한 시각화 샘플 생성
    
    Args:
        tiles_images_dir: 타일 이미지 디렉토리
        tiles_labels_dir: 타일 라벨 디렉토리  
        output_dir: 검증 샘플 출력 디렉토리
        num_samples: 생성할 샘플 수
    
    Returns:
        List[str]: 생성된 검증 샘플 파일명 리스트
    """
    import cv2
    import random
    
    # 검증 샘플 저장 디렉토리 생성
    validation_dir = output_dir / "validation_samples"
    validation_dir.mkdir(parents=True, exist_ok=True)
    
    # 라벨이 있는 타일들 찾기
    image_files = list(tiles_images_dir.glob("*.tif"))
    labeled_tiles = []
    unlabeled_tiles = []
    
    for img_file in image_files:
        label_file = tiles_labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            # 라벨 파일 내용 확인
            with open(label_file, 'r') as f:
                content = f.read().strip()
            if content:  # 라벨이 있는 경우
                labeled_tiles.append((img_file, label_file))
            else:  # 빈 라벨 파일 (negative sample)
                unlabeled_tiles.append((img_file, label_file))
    
    print(f"🔍 라벨링 검증 샘플 생성: {len(labeled_tiles)}개 양성, {len(unlabeled_tiles)}개 음성 타일 발견")
    
    sample_files = []
    
    # 양성 샘플 (라벨이 있는 타일) 시각화
    # 음성 샘플이 없으면 모든 샘플을 양성으로, 있으면 반반으로 배분
    if len(unlabeled_tiles) == 0:
        positive_samples = min(num_samples, len(labeled_tiles))
    else:
        positive_samples = min(num_samples // 2, len(labeled_tiles))
    
    if positive_samples > 0:
        selected_positive = random.sample(labeled_tiles, positive_samples)
        
        for i, (img_file, label_file) in enumerate(selected_positive):
            try:
                # 이미지 로드
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                    
                h, w = img.shape[:2]
                
                # 라벨 파일 읽기
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                # 바운딩박스 그리기
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id, x_center, y_center, width, height = map(float, parts[:5])
                        
                        # YOLO 정규화 좌표를 픽셀 좌표로 변환
                        x_center_px = int(x_center * w)
                        y_center_px = int(y_center * h)
                        width_px = int(width * w)
                        height_px = int(height * h)
                        
                        # 바운딩박스 좌표 계산
                        x1 = max(0, x_center_px - width_px // 2)
                        y1 = max(0, y_center_px - height_px // 2)
                        x2 = min(w, x_center_px + width_px // 2)
                        y2 = min(h, y_center_px + height_px // 2)
                        
                        # 바운딩박스 그리기 (녹색)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # 중심점 표시 (빨간 원)
                        cv2.circle(img, (x_center_px, y_center_px), 3, (0, 0, 255), -1)
                        
                        # 크기 정보 표시
                        cv2.putText(img, f"{width_px}x{height_px}", 
                                  (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 타일 정보 표시
                cv2.putText(img, f"POSITIVE: {img_file.name}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, f"Labels: {len(lines)}", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # 저장
                sample_filename = f"validation_positive_{i+1:02d}_{img_file.stem}.jpg"
                sample_path = validation_dir / sample_filename
                cv2.imwrite(str(sample_path), img)
                sample_files.append(sample_filename)
                
            except Exception as e:
                print(f"⚠️ 양성 샘플 생성 실패 {img_file.name}: {e}")
    
    # 음성 샘플 (라벨이 없는 타일) 시각화  
    negative_samples = min(num_samples - len(sample_files), len(unlabeled_tiles))
    if negative_samples > 0 and unlabeled_tiles:
        selected_negative = random.sample(unlabeled_tiles, negative_samples)
        
        for i, (img_file, label_file) in enumerate(selected_negative):
            try:
                # 이미지 로드
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                # 타일 정보 표시
                cv2.putText(img, f"NEGATIVE: {img_file.name}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img, "No Labels (Background)", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # 저장
                sample_filename = f"validation_negative_{i+1:02d}_{img_file.stem}.jpg"
                sample_path = validation_dir / sample_filename
                cv2.imwrite(str(sample_path), img)
                sample_files.append(sample_filename)
                
            except Exception as e:
                print(f"⚠️ 음성 샘플 생성 실패 {img_file.name}: {e}")
    
    print(f"✅ 라벨링 검증 샘플 {len(sample_files)}개 생성 완료")
    return sample_files

def create_validation_samples_zip(output_base: Path, timestamp: str, sample_files: List[str]) -> str:
    """검증 샘플들을 ZIP 파일로 압축"""
    zip_filename = f"labeling_validation_samples_{timestamp}.zip"
    zip_path = DEFAULT_OUTPUT_DIR / zip_filename
    
    validation_dir = output_base / "validation_samples"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 검증 샘플 이미지들 압축
        for sample_file in sample_files:
            sample_path = validation_dir / sample_file
            if sample_path.exists():
                zipf.write(sample_path, f"validation_samples/{sample_file}")
        
        # README 파일 생성 및 추가
        readme_content = f"""# 라벨링 검증 샘플

생성 시간: {datetime.datetime.now().isoformat()}
총 샘플 수: {len(sample_files)}개

## 파일 설명:
- validation_positive_XX_*.jpg: 피해목 라벨이 있는 타일 (녹색 바운딩박스)
- validation_negative_XX_*.jpg: 피해목이 없는 배경 타일

## 검증 방법:
1. 양성 샘플: 녹색 바운딩박스가 실제 피해목 위치와 일치하는지 확인
2. 음성 샘플: 실제로 피해목이 없는 깨끗한 배경인지 확인
3. 바운딩박스 크기가 적절한지 확인 (너무 크거나 작지 않은지)

## 문제 발견시:
- GPS 좌표 정확도 확인 필요
- 바운딩박스 크기 조정 필요  
- 데이터셋 재생성 권장
"""
        
        readme_path = validation_dir / "README.txt"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        zipf.write(readme_path, "validation_samples/README.txt")
    
    return str(zip_path)

# 헬퍼 함수들
def load_tfw(tfw_path):
    """TFW 파일에서 변환 파라미터 읽기"""
    with open(tfw_path) as f:
        vals = [float(x.strip()) for x in f.readlines()]
    return vals

def tm_to_pixel(x, y, tfw):
    """TM 좌표(x, y)를 이미지 픽셀 좌표(px, py)로 변환"""
    A, D, B, E, C, F = tfw
    px = (x - C) / A
    py = (y - F) / E
    return px, py

# 🎯 멀티스케일 바운딩박스 크기 계산 함수
def calculate_adaptive_bbox_size(target_px, target_py, all_pixel_coords, min_size=16, max_size=128):
    """
    GPS 좌표 밀도와 거리 기반 적응적 바운딩박스 크기 계산
    
    Args:
        target_px, target_py: 대상 픽셀 좌표
        all_pixel_coords: 모든 픽셀 좌표 리스트 [(px, py, row, idx), ...]
        min_size, max_size: 최소/최대 바운딩박스 크기
    
    Returns:
        int: 계산된 바운딩박스 크기
    """
    import math
    
    # 주변 반경 내 좌표 개수로 밀도 계산
    search_radius = 50  # 50픽셀 반경
    nearby_points = 0
    
    for px, py, row, idx in all_pixel_coords:
        distance = math.sqrt((target_px - px)**2 + (target_py - py)**2)
        if distance <= search_radius:
            nearby_points += 1
    
    # 밀도가 높을수록 작은 바운딩박스, 낮을수록 큰 바운딩박스
    if nearby_points <= 1:  # 매우 낮은 밀도 (외딴 피해목)
        bbox_size = max_size  # 128px
    elif nearby_points <= 3:  # 낮은 밀도
        bbox_size = int(max_size * 0.75)  # 96px
    elif nearby_points <= 6:  # 중간 밀도
        bbox_size = int(max_size * 0.5)  # 64px
    elif nearby_points <= 10:  # 높은 밀도
        bbox_size = int(max_size * 0.375)  # 48px
    else:  # 매우 높은 밀도 (밀집된 피해목)
        bbox_size = min_size  # 16px
    
    return bbox_size

def process_tiles_and_labels(image_path, tfw_params, df, output_images, output_labels, 
                           tile_size, class_id, file_prefix):
    """📦 멀티스케일 바운딩박스를 사용한 타일링 및 라벨 생성 (적응적 크기)"""
    tile_info, coordinate_tracking = process_tiles_and_labels_simple(
        image_path, tfw_params, df, output_images, output_labels, 
        tile_size, class_id, file_prefix, use_adaptive_bbox=True
    )
    return tile_info, coordinate_tracking

def process_tiles_and_labels_simple(image_path, tfw_params, df, output_images, output_labels, 
                                   tile_size, class_id, file_prefix, use_adaptive_bbox=True):
    """
    📦 이미지 타일 분할 및 멀티스케일 YOLO 라벨 생성 (GPS 좌표 추적 포함)
    
    Args:
        use_adaptive_bbox: 적응적 바운딩박스 사용 여부 (기본: True)
        
    Returns:
        tuple: (tile_info, coordinate_tracking_info)
    """
    tile_info = []
    
    # 🧹 이전 지도의 캐시된 픽셀 좌표 초기화 (각 지도마다 독립적 처리)
    if hasattr(process_tiles_and_labels_simple, '_pixel_coords'):
        delattr(process_tiles_and_labels_simple, '_pixel_coords')
        print("🔄 이전 지도의 픽셀 좌표 캐시 초기화", flush=True)
    
    # 🔍 GPS 좌표 추적을 위한 변수들
    total_coordinates = len(df)
    processed_coordinates = set()  # 처리된 좌표 인덱스들
    out_of_bounds_coordinates = []  # 이미지 범위 밖 좌표들
    labels_created = 0  # 생성된 총 라벨 수
    
    with rasterio.open(image_path) as src:
        width, height = src.width, src.height
        n_tiles_x = int(np.ceil(width / tile_size))
        n_tiles_y = int(np.ceil(height / tile_size))
        
        total_tiles = n_tiles_x * n_tiles_y
        processed_tiles = 0
        
        print(f"🎯 타일 분할 시작: {width}x{height} → {n_tiles_x}x{n_tiles_y} = {total_tiles}개 타일", flush=True)
        print(f"📊 GPS 포인트 {len(df)}개를 {total_tiles}개 타일에 분배 중...", flush=True)
        
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                x0 = tx * tile_size
                y0 = ty * tile_size
                w = min(tile_size, width - x0)
                h = min(tile_size, height - y0)
                window = Window(x0, y0, w, h)
                
                # RGB 3채널만 추출하여 타일 이미지 생성
                tile_img = src.read([1, 2, 3], window=window)
                tile_name = f"{file_prefix}_{tx}_{ty}.tif"
                
                # 🎯 Multi-Scale 타일 내 bbox 라벨 생성
                lines = []
                
                # 🚀 성능 최적화: 타일 영역에 포함될 가능성이 있는 좌표만 필터링
                # 전체 좌표를 픽셀로 변환 (한 번만 계산) + 추적 정보 포함
                if not hasattr(process_tiles_and_labels_simple, '_pixel_coords'):
                    print("🔄 GPS 좌표를 픽셀 좌표로 변환 중... (최초 1회)", flush=True)
                    pixel_coords = []
                    for idx, row in df.iterrows():
                        px, py = tm_to_pixel(row["x"], row["y"], tfw_params)
                        
                        # 이미지 범위 체크
                        if px < 0 or py < 0 or px >= width or py >= height:
                            out_of_bounds_coordinates.append({
                                'index': idx,
                                'original_coords': (row["x"], row["y"]),
                                'pixel_coords': (px, py),
                                'reason': 'out_of_image_bounds'
                            })
                        else:
                            pixel_coords.append((px, py, row, idx))  # 인덱스 추가
                    
                    process_tiles_and_labels_simple._pixel_coords = pixel_coords
                    print(f"✅ {len(pixel_coords)}개 좌표 변환 완료 (범위 외: {len(out_of_bounds_coordinates)}개)", flush=True)
                
                # 현재 타일 영역에 포함되는 좌표만 처리 (추적 포함)
                for px, py, row, coord_idx in process_tiles_and_labels_simple._pixel_coords:
                    # 타일 내 상대좌표로 변환
                    rel_x = px - x0
                    rel_y = py - y0
                    if 0 <= rel_x < w and 0 <= rel_y < h:
                        x_center = rel_x / w
                        y_center = rel_y / h
                        
                        # 🎯 멀티스케일 적응적 바운딩박스 생성
                        # GPS 좌표 밀도 기반 적응적 바운딩박스 크기 계산
                        adaptive_size = calculate_adaptive_bbox_size(px, py, process_tiles_and_labels_simple._pixel_coords, 
                                                                  min_size=config.ADAPTIVE_BBOX_MIN_SIZE, 
                                                                  max_size=config.ADAPTIVE_BBOX_MAX_SIZE)
                        bw = adaptive_size / w
                        bh = adaptive_size / h
                        lines.append(
                            f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"
                        )
                        
                        # 🔍 처리된 좌표 추적
                        processed_coordinates.add(coord_idx)
                        labels_created += 1
                
                # 라벨이 있는 경우에만 저장
                labels_count = len(lines)
                saved = labels_count > 0
                
                # 진행상황 출력 (10% 단위)
                processed_tiles += 1
                if processed_tiles % max(1, total_tiles // 10) == 0 or processed_tiles == total_tiles:
                    progress_pct = (processed_tiles / total_tiles) * 100
                    tiles_with_labels = sum(1 for t in tile_info if hasattr(t, 'labels_count') and t.labels_count > 0) + (1 if saved else 0)
                    print(f"  📊 진행상황: {processed_tiles}/{total_tiles} ({progress_pct:.1f}%) - 라벨 있는 타일: {tiles_with_labels}개", flush=True)
                
                if saved:
                    # 라벨 파일 저장
                    out_lbl_path = os.path.join(output_labels, tile_name.replace(".tif", ".txt"))
                    with open(out_lbl_path, "w") as f:
                        f.write("\n".join(lines))
                    
                    # 타일 이미지 저장
                    out_img_path = os.path.join(output_images, tile_name)
                    orig_affine = src.transform
                    tile_affine = orig_affine * Affine.translation(x0, y0)
                    with rasterio.open(
                        out_img_path,
                        "w",
                        driver="GTiff",
                        height=h,
                        width=w,
                        count=3,
                        dtype=src.dtypes[0],
                        crs=src.crs,
                        transform=tile_affine,
                    ) as dst:
                        dst.write(tile_img)
                
                # 타일 정보 저장 (TileInfo 모델에 맞게 조정)
                tile_info.append(TileInfo(
                    tile_name=tile_name,
                    labels_count=labels_count,
                    tile_size=(w, h),
                    position=(tx, ty)
                ))
    
    # 🔍 GPS 좌표 추적 통계 계산
    missing_coordinates = len(out_of_bounds_coordinates) if out_of_bounds_coordinates else 0
    total_processed = len(processed_coordinates) if processed_coordinates else 0
    
    # 최종 통계 출력 (GPS 추적 포함)
    tiles_with_labels = sum(1 for t in tile_info if t.labels_count > 0)
    total_labels = sum(t.labels_count for t in tile_info)
    csv_total = len(process_tiles_and_labels_simple._pixel_coords) if hasattr(process_tiles_and_labels_simple, '_pixel_coords') else 0
    success_rate = (total_processed/csv_total * 100) if csv_total > 0 else 0
    
    print(f"✅ 타일 분할 완료: {total_tiles}개 타일 생성, {tiles_with_labels}개 라벨 타일, {total_labels}개 총 라벨", flush=True)
    print(f"🔍 GPS 좌표 매핑: CSV {csv_total}개 → 라벨 {total_processed}개 생성 ({success_rate:.1f}%), 범위외 {missing_coordinates}개", flush=True)
    
    # 좌표 추적 정보 포함하여 반환
    coordinate_tracking = {
        "csv_total": csv_total,
        "labels_created": total_processed,
        "out_of_bounds": missing_coordinates,
        "success_rate": f"{success_rate:.2f}%"
    }
    
    return tile_info, coordinate_tracking

@router.get("/download/{filename}")
async def download_tiles_zip(filename: str):
    """
    생성된 타일과 라벨 ZIP 파일을 다운로드합니다.
    """
    file_path = DEFAULT_OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/zip'
    )


@router.get("/status")
async def preprocessing_status():
    """
    전처리 서비스의 상태 정보를 반환합니다.
    """
    # 최근 생성된 타일 폴더들 조회
    recent_tiles = []
    if DEFAULT_OUTPUT_DIR.exists():
        for item in sorted(os.listdir(DEFAULT_OUTPUT_DIR), reverse=True)[:5]:
            item_path = DEFAULT_OUTPUT_DIR / item
            if item_path.is_dir():
                images_dir = item_path / "images"
                labels_dir = item_path / "labels"
                images_count = len([f for f in os.listdir(images_dir) 
                                 if f.endswith('.tif')]) if images_dir.exists() else 0
                labels_count = len([f for f in os.listdir(labels_dir)
                                 if f.endswith('.txt')]) if labels_dir.exists() else 0
                
                recent_tiles.append({
                    "folder_name": item,
                    "images_count": images_count,
                    "labels_count": labels_count
                })
    
    return {
        "service": "preprocessing",
        "status": "active",
        "recent_tiles": recent_tiles,
        "default_settings": {
            "tile_size": DEFAULT_TILE_SIZE,
            "bbox_mode": f"Multiscale Adaptive {config.ADAPTIVE_BBOX_MIN_SIZE}~{config.ADAPTIVE_BBOX_MAX_SIZE}px",
            "class_id": DEFAULT_CLASS_ID
        }
    }


@router.post("/tile-for-inference", response_model=InferenceTilingResponse)
async def create_inference_tiles(
    image_file: UploadFile = File(..., description="원본 GeoTIFF 이미지"),
    tile_size: int = Form(default=DEFAULT_TILE_SIZE, description="타일 크기 (픽셀)")
):
    """
    추론용 타일 생성: 원본 이미지를 추론하기 위해 타일로 분할합니다.
    원본 이미지만 분할하며, 지리참조 정보를 보존합니다.
    
    - **image_file**: 원본 GeoTIFF 이미지 파일 (.tif/.tiff)
    - **tile_size**: 타일 한 변 크기 (기본: 1024픽셀)
    """
    
    try:
        # 파일 확장자 검증
        if not image_file.filename.lower().endswith(('.tif', '.tiff')):
            raise HTTPException(status_code=400, detail="이미지 파일은 TIFF 형식이어야 합니다.")
        
        # 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # 업로드 파일을 임시 디렉토리에 저장
            image_path = os.path.join(temp_dir, image_file.filename)
            
            with open(image_path, "wb") as f:
                shutil.copyfileobj(image_file.file, f)
            
            # 날짜 기반 접두사 생성
            today = datetime.datetime.now().strftime("%Y%m%d")
            
            # 같은 날짜의 기존 폴더들 확인하여 순차 접두사 결정
            existing_prefixes = []
            if os.path.exists(DEFAULT_OUTPUT_DIR):
                for folder in os.listdir(DEFAULT_OUTPUT_DIR):
                    if folder.startswith(f"inference_tiles_") and today in folder:
                        # inference_tiles_A20250915_ 형태에서 접두사 추출
                        parts = folder.split('_')
                        if len(parts) >= 3 and parts[2].startswith(('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J')):
                            prefix_char = parts[2][0]
                            existing_prefixes.append(prefix_char)
            
            # 다음 순차 접두사 결정 (A, B, C, ... Z 순서)
            next_prefix = 'A'
            for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                if char not in existing_prefixes:
                    next_prefix = char
                    break
            
            # 파일 접두사 생성
            file_prefix = f"{next_prefix}{today}"
            
            # 출력 디렉토리 생성
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base = DEFAULT_OUTPUT_DIR / f"inference_tiles_{file_prefix}_{timestamp}"
            output_images = output_base / "images"
            
            output_images.mkdir(parents=True, exist_ok=True)
            
            # 추론용 타일 분할 실행
            tile_info, original_size = process_inference_tiles(
                image_path, str(output_images), tile_size, file_prefix
            )
            
            # ZIP 파일 생성
            zip_path = create_inference_tiles_zip(output_base, timestamp)
            
            # 통계 계산
            total_tiles = len(tile_info)
            
            return InferenceTilingResponse(
                success=True,
                message=f"추론용 타일 분할 완료: {total_tiles}개 타일 생성",
                total_tiles=total_tiles,
                original_size=original_size,
                tile_size=tile_size,
                tile_info=tile_info,
                download_url=f"/api/v1/preprocessing/download/{os.path.basename(zip_path)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론용 타일 생성 중 오류 발생: {str(e)}")


def process_inference_tiles(
    image_path: str, output_images: str, tile_size: int, file_prefix: str
) -> tuple[List[InferenceTileInfo], tuple]:
    """추론용 타일 분할 메인 로직 (라벨링 없음)"""
    
    tile_info = []
    
    with rasterio.open(image_path) as src:
        width, height = src.width, src.height
        original_size = (width, height)
        
        n_tiles_x = int(np.ceil(width / tile_size))
        n_tiles_y = int(np.ceil(height / tile_size))
        
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                x0 = tx * tile_size
                y0 = ty * tile_size
                w = min(tile_size, width - x0)
                h = min(tile_size, height - y0)
                
                window = Window(x0, y0, w, h)
                
                # RGB 3채널만 추출하여 타일 이미지 생성
                tile_img = src.read([1, 2, 3], window=window)
                tile_name = f"{file_prefix}_tile_{tx}_{ty}.tif"
                
                # 타일 이미지 저장 (지리참조 정보 보존)
                tile_path = os.path.join(output_images, tile_name)
                orig_affine = src.transform
                tile_affine = orig_affine * Affine.translation(x0, y0)
                
                has_georeference = src.crs is not None and src.transform is not None
                
                with rasterio.open(
                    tile_path, 'w',
                    driver='GTiff',
                    height=h, width=w, count=3,
                    dtype=src.dtypes[0],
                    crs=src.crs,
                    transform=tile_affine
                ) as dst:
                    dst.write(tile_img)
                
                # 타일 정보 추가
                tile_info.append(InferenceTileInfo(
                    tile_name=tile_name,
                    tile_size=(w, h),
                    position=(tx, ty),
                    has_georeference=has_georeference
                ))
    
    return tile_info, original_size


def create_inference_tiles_zip(output_base: Path, timestamp: str) -> str:
    """생성된 추론용 타일을 ZIP 파일로 압축"""
    zip_filename = f"inference_tiles_{timestamp}.zip"
    zip_path = DEFAULT_OUTPUT_DIR / zip_filename
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # images 폴더 압축
        images_dir = output_base / "images"
        if images_dir.exists():
            for root, dirs, files in os.walk(images_dir):
                # macOS 시스템 폴더 제외
                if '__MACOSX' in root:
                    continue
                for file in files:
                    # macOS 숨김 파일 및 시스템 파일 제외
                    if file.startswith('.') or file.startswith('._'):
                        continue
                    file_path = os.path.join(root, file)
                    archive_name = os.path.relpath(file_path, output_base)
                    zipf.write(file_path, archive_name)
    
    return str(zip_path)


@router.post("/create-dataset", response_model=IntegratedTrainingResponse)
async def create_dataset(
    image_file: UploadFile = File(..., description="처리할 GeoTIFF 이미지 파일"),
    csv_file: UploadFile = File(..., description="GPS 좌표 CSV 파일 (x, y 또는 longitude, latitude 컬럼)"),
    tfw_file: UploadFile = File(..., description="지리참조를 위한 TFW 파일"),
    file_prefix: str = Form(..., description="생성될 타일 파일명 접두사"),
    tile_size: int = Form(default=config.DEFAULT_TILE_SIZE, description="타일 크기 (픽셀)"),

    class_id: int = Form(default=0, description="YOLO 클래스 ID"),
    train_split: float = Form(default=0.8, description="학습 데이터 비율 (0.0-1.0)"),
    class_names: str = Form(default="damaged_tree", description="클래스 이름 (쉼표로 구분)"),
    max_files: int = Form(default=1000, description="최대 파일 수 제한 (0=무제한)"),
    shuffle_data: bool = Form(default=True, description="데이터 셔플 여부"),
    auto_split: bool = Form(default=True, description="자동 train/val 분할 여부")
):
    """
    🚀 **안정적인 딥러닝 데이터셋 생성 API** (권장)
    
    **이 API는 다음 작업을 한번에 수행합니다:**
    1. 🖼️ GeoTIFF 이미지를 타일로 분할 (모든 영역 포함)
    2. � GPS 좌표 기반 YOLO 라벨 자동 생성 (멀티스케일 적응적)
    3. 🔍 **라벨링 품질 검증 샘플 생성** (NEW!)
    4. ⚖️ Positive/Negative 샘플 균형 데이터셋 생성
    5. 📦 Google Colab 최적화 딥러닝용 ZIP 파일 생성
    6. 📊 Train/Validation 데이터셋 자동 분할
    
    **🎯 멀티스케일 데이터셋 특징:**
    - ✅ **적응적 바운딩박스**: GPS 밀도 기반 16~128px 가변 크기
    - ✅ **실제 크기 반영**: 외딴 피해목(큰 박스) vs 밀집 피해목(작은 박스)
    - ✅ **라벨링 검증**: 바운딩박스 오버레이된 샘플 이미지로 품질 확인 가능
    - ✅ **향상된 탐지율**: 다양한 크기의 피해목 모두 탐지 가능
    - ✅ **Negative 샘플 포함**: 오탐지 방지를 위한 음성 샘플 자동 생성
    
    **📋 매개변수:**
    5. 📋 README 및 사용법 가이드 포함
    
    **📋 매개변수:**
    - **image_file**: 처리할 GeoTIFF 이미지 파일 (.tif/.tiff)
    - **csv_file**: GPS 좌표 CSV 파일 (x,y 또는 longitude,latitude 컬럼 필요)
    - **tfw_file**: 지리참조를 위한 TFW 파일 (.tfw)
    - **file_prefix**: 생성될 타일 파일명 접두사 (예: "A20250919")
    - **tile_size**: 타일 크기 (기본: 1024px, 세밀한 탐지 최적화)
    - **멀티스케일 바운딩박스**: GPS 밀도 기반 자동 크기 조절
      - 외딴 피해목: 128px (큰 바운딩박스)
      - 밀집 지역: 16px (작은 바운딩박스) 
      - 중간 밀도: 48~96px (적응적 조절)
    - **class_id**: YOLO 클래스 ID (기본: 0)
    - **train_split**: 학습 데이터 비율 (기본: 0.8 = 80% 학습, 20% 검증)
    - **class_names**: 클래스 이름 (쉼표로 구분, 기본: "damaged_tree")
    - **max_files**: 최대 파일 수 제한 (기본: 1000개, 0=무제한)
    - **shuffle_data**: 데이터 셔플 여부 (기본: True)
    - **auto_split**: 자동 train/val 분할 여부 (기본: True, YOLO 호환)
    
    **🎯 출력:**
    - ZIP 파일에 포함: images/, labels/, data.yaml, README.txt
    - Google Colab에서 바로 사용 가능한 구조
    - YOLOv8/YOLOv11 호환 형식
    
    **💡 사용 예시:**
    ```python
    # Google Colab에서 데이터셋 사용
    !unzip -q "/content/drive/MyDrive/pinetree_training_dataset_*.zip" -d /content/dataset
    from ultralytics import YOLO
    model = YOLO('yolo11s.pt')
    results = model.train(data='/content/dataset/data.yaml', epochs=200)
    ```
    
    **🔍 라벨링 검증 방법:**
    1. `validation_samples_url`에서 검증 샘플 ZIP 다운로드
    2. 양성 샘플: 녹색 바운딩박스가 실제 피해목과 일치하는지 확인
    3. 음성 샘플: 실제로 피해목이 없는 깨끗한 배경인지 확인
    4. 문제 발견 시 GPS 좌표나 bbox_size 조정 후 재생성
    
    **📋 추가 매개변수:**
    - **class_names**: 클래스 이름 (쉼표로 구분, 기본: "damaged_tree")
    - **max_files**: 최대 파일 수 제한 (기본: 1000개, 0=무제한)
    - **shuffle_data**: 데이터 셔플 여부 (기본: True)
    - **auto_split**: 자동 train/val 분할 여부 (기본: True, YOLO 호환)
    """
    
    try:
        # 파일 확장자 검증
        if not image_file.filename.lower().endswith(tuple(config.ALLOWED_IMAGE_EXTENSIONS)):
            raise HTTPException(
                status_code=400, 
                detail=f"지원하지 않는 이미지 형식: {image_file.filename}"
            )
        
        if not csv_file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="CSV 파일이 필요합니다.")
        
        if not tfw_file.filename.lower().endswith('.tfw'):
            raise HTTPException(status_code=400, detail="TFW 파일이 필요합니다.")
        
        # 출력 디렉토리 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = DEFAULT_OUTPUT_DIR / f"complete_training_{timestamp}"
        output_base.mkdir(parents=True, exist_ok=True)
        
        # 타일 저장 디렉토리
        tiles_images_dir = output_base / "tiles" / "images"
        tiles_labels_dir = output_base / "tiles" / "labels" 
        tiles_images_dir.mkdir(parents=True, exist_ok=True)
        tiles_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # 최종 데이터셋 디렉토리
        dataset_dir = output_base / "training_dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # 임시 파일 저장
        with tempfile.TemporaryDirectory() as temp_dir:
            # 업로드된 파일들 저장
            image_path = os.path.join(temp_dir, image_file.filename)
            csv_path = os.path.join(temp_dir, csv_file.filename)
            tfw_path = os.path.join(temp_dir, tfw_file.filename)
            
            with open(image_path, "wb") as f:
                shutil.copyfileobj(image_file.file, f)
            with open(csv_path, "wb") as f:
                shutil.copyfileobj(csv_file.file, f)
            with open(tfw_path, "wb") as f:
                shutil.copyfileobj(tfw_file.file, f)
            
            # TFW 파라미터 파싱
            tfw_params = []
            with open(tfw_path, 'r') as f:
                tfw_params = [float(line.strip()) for line in f.readlines()]
            
            if len(tfw_params) != 6:
                raise HTTPException(status_code=400, detail="TFW 파일 형식이 올바르지 않습니다.")
            
            # CSV 파일 읽기
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"CSV 파일 읽기 실패: {str(e)}")
            
            # 좌표 컬럼 확인
            x_col, y_col = None, None
            if 'x' in df.columns and 'y' in df.columns:
                x_col, y_col = 'x', 'y'
            elif 'longitude' in df.columns and 'latitude' in df.columns:
                x_col, y_col = 'longitude', 'latitude'
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="CSV 파일에 'x,y' 또는 'longitude,latitude' 컬럼이 필요합니다."
                )
            
            # 좌표 컬럼 이름 통일
            df = df.rename(columns={x_col: 'x', y_col: 'y'})
            
            print(f"📊 GPS 좌표 데이터: {len(df)}개 포인트", flush=True)
            
            # Step 1: 📦 멀티스케일 적응적 타일링 및 라벨 생성
            print(f"🔄 Step 1: 멀티스케일 적응적 타일링 및 라벨 생성 시작...", flush=True)
            print(f"🎯 적응적 바운딩박스: {config.ADAPTIVE_BBOX_MIN_SIZE}~{config.ADAPTIVE_BBOX_MAX_SIZE}px (밀도 기반)", flush=True)
            
            # 멀티스케일 적응적 바운딩박스 처리
            tile_info, coordinate_tracking = process_tiles_and_labels_simple(
                image_path, tfw_params, df, str(tiles_images_dir), str(tiles_labels_dir),
                tile_size, class_id, file_prefix, use_adaptive_bbox=True
            )
            
            tiles_with_labels = sum(1 for t in tile_info if t.labels_count > 0)
            total_labels = sum(t.labels_count for t in tile_info)
            
            # 라벨링 품질 통계 계산
            labels_per_tile = [t.labels_count for t in tile_info if t.labels_count > 0]
            avg_labels_per_tile = sum(labels_per_tile) / len(labels_per_tile) if labels_per_tile else 0
            
            preprocessing_info = {
                "total_tiles": len(tile_info),
                "tiles_with_labels": tiles_with_labels,
                "total_labels": total_labels,
                "tile_size": tile_size,
                "bbox_mode": f"Multiscale Adaptive {config.ADAPTIVE_BBOX_MIN_SIZE}~{config.ADAPTIVE_BBOX_MAX_SIZE}px",
                "file_prefix": file_prefix,
                "labeling_stats": {
                    "avg_labels_per_tile": round(avg_labels_per_tile, 2),
                    "max_labels_per_tile": max(labels_per_tile) if labels_per_tile else 0,
                    "min_labels_per_tile": min(labels_per_tile) if labels_per_tile else 0,
                    "coverage_rate": round(tiles_with_labels / len(tile_info) * 100, 1)
                }
            }
            
            print(f"✅ Step 1 완료: {len(tile_info)}개 타일, {tiles_with_labels}개 라벨 타일, {total_labels}개 총 라벨", flush=True)
            
            # Step 2: 🔍 라벨링 품질 검증 샘플 생성
            print("🔄 Step 2: 라벨링 품질 검증 샘플 생성 시작...", flush=True)
            
            validation_sample_files = create_labeling_validation_samples(
                tiles_images_dir, tiles_labels_dir, output_base, num_samples=30
            )
            
            # 검증 샘플 ZIP 생성
            validation_zip_path = None
            validation_info = {
                "validation_samples_count": len(validation_sample_files),
                "positive_samples": len([f for f in validation_sample_files if "positive" in f]),
                "negative_samples": len([f for f in validation_sample_files if "negative" in f])
            }
            
            if validation_sample_files:
                validation_zip_path = create_validation_samples_zip(output_base, timestamp, validation_sample_files)
                print(f"✅ Step 2 완료: {len(validation_sample_files)}개 검증 샘플 생성", flush=True)
            else:
                print("⚠️ Step 2: 검증 샘플 생성 실패", flush=True)
            
            # Step 3: 딥러닝용 데이터셋 생성  
            print("🔄 Step 3: 딥러닝용 데이터셋 생성 시작...", flush=True)
            
            # 타일 이미지와 라벨 파일 매칭 (모든 이미지 포함)
            image_files = list(tiles_images_dir.glob("*.tif"))
            label_files = list(tiles_labels_dir.glob("*.txt"))
            
            # 모든 이미지에 대해 라벨 파일 매칭 (없으면 빈 라벨로 처리)
            matched_files = []
            positive_samples = 0  # 라벨이 있는 샘플 수
            negative_samples = 0  # 라벨이 없는 샘플 수
            
            for img_file in image_files:
                label_file = tiles_labels_dir / f"{img_file.stem}.txt"
                
                # 라벨 파일이 없으면 빈 라벨 파일 생성
                if not label_file.exists():
                    with open(label_file, 'w') as f:
                        pass  # 빈 파일 생성
                    negative_samples += 1
                else:
                    # 라벨 파일이 있는 경우 내용 확인
                    with open(label_file, 'r') as f:
                        content = f.read().strip()
                    if content:
                        positive_samples += 1
                    else:
                        negative_samples += 1
                
                matched_files.append((img_file, label_file))
            
            print(f"📊 전체 데이터셋: {len(matched_files)}개")
            print(f"📊 양성 샘플 (피해목 있음): {positive_samples}개")
            print(f"📊 음성 샘플 (피해목 없음): {negative_samples}개")
            print(f"📊 클래스 비율: Positive={positive_samples/(positive_samples+negative_samples)*100:.1f}%, Negative={negative_samples/(positive_samples+negative_samples)*100:.1f}%")
            
            if len(matched_files) == 0:
                raise HTTPException(
                    status_code=400, 
                    detail="타일 이미지를 찾을 수 없습니다."
                )
            
            # 데이터 셔플
            if shuffle_data:
                import random
                random.shuffle(matched_files)
            
            # 파일 수 제한
            if max_files > 0:
                matched_files = matched_files[:max_files]
            
            # train/val 분할
            split_idx = int(len(matched_files) * train_split) if auto_split else len(matched_files)
            train_files = matched_files[:split_idx]
            val_files = matched_files[split_idx:] if auto_split else []
            
            # 디렉토리 생성
            for split_name in ['train', 'val'] if auto_split else ['train']:
                (dataset_dir / split_name / 'images').mkdir(parents=True, exist_ok=True)
                (dataset_dir / split_name / 'labels').mkdir(parents=True, exist_ok=True)
            
            # 파일 복사
            for img_file, label_file in train_files:
                shutil.copy2(img_file, dataset_dir / 'train' / 'images')
                shutil.copy2(label_file, dataset_dir / 'train' / 'labels')
            
            if auto_split and val_files:
                for img_file, label_file in val_files:
                    shutil.copy2(img_file, dataset_dir / 'val' / 'images')
                    shutil.copy2(label_file, dataset_dir / 'val' / 'labels')
            
            # data.yaml 생성
            class_list = [name.strip() for name in class_names.split(',')]
            data_yaml = {
                'path': '.',
                'train': 'train/images',
                'val': 'val/images' if auto_split else 'train/images',
                'nc': len(class_list),
                'names': class_list,
                
                # 메타데이터
                'created_date': datetime.datetime.now().isoformat(),
                'source_info': {
                    'image_file': image_file.filename,
                    'csv_file': csv_file.filename,
                    'total_coordinates': len(df),
                    'tile_size': tile_size,
                    'bbox_mode': f"Multiscale Adaptive {config.ADAPTIVE_BBOX_MIN_SIZE}~{config.ADAPTIVE_BBOX_MAX_SIZE}px (Density-based)"
                },
                'dataset_stats': {
                    'total_files': len(matched_files),
                    'train_files': len(train_files),
                    'val_files': len(val_files),
                    'total_labels': total_labels
                }
            }
            
            import yaml
            with open(dataset_dir / 'data.yaml', 'w') as f:
                yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
            
            # Step 4: ZIP 파일 생성
            print("🔄 Step 4: ZIP 파일 생성 시작...", flush=True)
            zip_filename = f"complete_training_dataset_{file_prefix}_{timestamp}.zip"
            zip_path = DEFAULT_OUTPUT_DIR / zip_filename
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 데이터셋 디렉토리 전체를 압축 (macOS 숨김 파일 제외)
                for root, dirs, files in os.walk(dataset_dir):
                    # macOS 시스템 폴더 제외
                    if '__MACOSX' in root:
                        continue
                    for file in files:
                        # macOS 숨김 파일 및 시스템 파일 제외
                        if file.startswith('.') or file.startswith('._'):
                            continue
                        file_path = os.path.join(root, file)
                        archive_name = os.path.relpath(file_path, dataset_dir)
                        zipf.write(file_path, archive_name)
            
            # 최종 정보 구성
            dataset_info = {
                "zip_filename": zip_filename,
                "total_files": len(matched_files),
                "train_files": len(train_files),
                "val_files": len(val_files),
                "total_labels": total_labels,
                "classes": class_list,
                "data_yaml_created": True,
                "file_size_mb": round(zip_path.stat().st_size / (1024 * 1024), 2)
            }
            
            download_url = f"/api/v1/preprocessing/download/{zip_filename}"
            validation_samples_url = None
            
            if validation_zip_path:
                validation_samples_url = f"/api/v1/preprocessing/download/{os.path.basename(validation_zip_path)}"
            
            print(f"✅ 통합 처리 완료!", flush=True)
            print(f"📦 데이터셋 ZIP: {zip_filename} ({dataset_info['file_size_mb']}MB)", flush=True)
            if validation_samples_url:
                print(f"🔍 검증 샘플 ZIP: {os.path.basename(validation_zip_path)}", flush=True)
            
            return IntegratedTrainingResponse(
                success=True,
                message=f"통합 데이터셋 생성 완료! 총 {len(matched_files)}개 파일, {total_labels}개 라벨 (검증 샘플 포함) | GPS 매핑: {coordinate_tracking['labels_created']}개/{coordinate_tracking['csv_total']}개 ({coordinate_tracking['success_rate']})",
                preprocessing_info=preprocessing_info,
                training_dataset_info=dataset_info,
                download_url=download_url,
                zip_filename=zip_filename,
                validation_samples_url=validation_samples_url,
                validation_info=validation_info,
                coordinate_validation=coordinate_tracking  # 🔍 GPS 좌표 추적 정보 추가
            )
            
    except Exception as e:
        print(f"❌ 통합 처리 오류: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"통합 데이터셋 생성 중 오류 발생: {str(e)}")


@router.get("/download/{filename}")
async def download_processed_file(filename: str):
    """처리된 파일 다운로드"""
    file_path = DEFAULT_OUTPUT_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='application/octet-stream'
    )

# 🔗 다중 데이터셋 통합을 위한 헬퍼 함수들
def analyze_dataset_structure(extract_dir: Path, zip_filename: str) -> Dict:
    """ZIP 파일의 데이터셋 구조를 분석"""
    
    # data.yaml 찾기
    data_yaml_path = None
    for root, dirs, files in os.walk(extract_dir):
        if 'data.yaml' in files:
            data_yaml_path = Path(root) / 'data.yaml'
            break
    
    dataset_info = {
        "zip_filename": zip_filename,
        "has_data_yaml": data_yaml_path is not None,
        "train_path": None,
        "val_path": None,
        "images_count": {"train": 0, "val": 0},
        "labels_count": {"train": 0, "val": 0}
    }
    
    if data_yaml_path:
        try:
            with open(data_yaml_path, 'r', encoding='utf-8') as f:
                data_yaml = yaml.safe_load(f)
                dataset_info["classes"] = data_yaml.get("names", ["damaged_tree"])
                dataset_info["nc"] = data_yaml.get("nc", 1)
        except:
            dataset_info["classes"] = ["damaged_tree"]
            dataset_info["nc"] = 1
    
    # train/val 폴더 찾기
    base_dir = data_yaml_path.parent if data_yaml_path else extract_dir
    
    for split in ["train", "val"]:
        images_dir = base_dir / split / "images"
        labels_dir = base_dir / split / "labels"
        
        if images_dir.exists():
            dataset_info[f"{split}_path"] = images_dir.parent
            dataset_info["images_count"][split] = len(list(images_dir.glob("*.tif")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
            if labels_dir.exists():
                dataset_info["labels_count"][split] = len(list(labels_dir.glob("*.txt")))
    
    return dataset_info

def collect_dataset_files(extract_dir: Path, dataset_info: Dict, source_id: int) -> tuple:
    """데이터셋에서 파일들을 수집하고 소스 정보 추가"""
    
    train_files = []
    val_files = []
    
    for split, file_list in [("train", train_files), ("val", val_files)]:
        split_path = dataset_info.get(f"{split}_path")
        if not split_path:
            continue
            
        images_dir = Path(split_path) / "images"
        labels_dir = Path(split_path) / "labels"
        
        if images_dir.exists():
            for img_file in images_dir.glob("*"):
                if img_file.suffix.lower() in ['.tif', '.tiff', '.jpg', '.jpeg', '.png']:
                    label_file = labels_dir / f"{img_file.stem}.txt"
                    
                    # 라벨 파일이 없으면 빈 라벨 파일로 처리
                    if not label_file.exists():
                        label_file = None
                    
                    file_list.append({
                        "image_path": img_file,
                        "label_path": label_file,
                        "source_id": source_id,
                        "source_zip": dataset_info["zip_filename"],
                        "original_split": split
                    })
    
    return train_files, val_files

def create_merged_dataset(all_train_files: List, all_val_files: List, temp_path: Path, 
                         merged_dataset_name: str, train_split: float, shuffle_data: bool,
                         class_names: str, timestamp: str) -> Dict:
    """통합 데이터셋 생성"""
    
    # 모든 파일을 하나로 합치기
    all_files = all_train_files + all_val_files
    
    if shuffle_data:
        import random
        random.shuffle(all_files)
    
    # 새로운 train/val 분할
    split_idx = int(len(all_files) * train_split)
    new_train_files = all_files[:split_idx]
    new_val_files = all_files[split_idx:]
    
    # 통합 데이터셋 디렉토리 생성
    dataset_dir = temp_path / "merged_dataset"
    for split in ["train", "val"]:
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # 파일 복사 및 중복명 처리
    copied_files = {"train": 0, "val": 0}
    filename_counter = {}  # 중복 파일명 카운터
    
    for split, files in [("train", new_train_files), ("val", new_val_files)]:
        for file_info in files:
            # 고유한 파일명 생성
            orig_name = file_info["image_path"].stem
            if orig_name in filename_counter:
                filename_counter[orig_name] += 1
                unique_name = f"{orig_name}_{filename_counter[orig_name]:03d}"
            else:
                filename_counter[orig_name] = 0
                unique_name = orig_name
            
            # 이미지 파일 복사
            src_img = file_info["image_path"]
            dst_img = dataset_dir / split / "images" / f"{unique_name}{src_img.suffix}"
            shutil.copy2(src_img, dst_img)
            
            # 라벨 파일 복사 (없으면 빈 파일 생성)
            dst_label = dataset_dir / split / "labels" / f"{unique_name}.txt"
            if file_info["label_path"] and file_info["label_path"].exists():
                shutil.copy2(file_info["label_path"], dst_label)
            else:
                # 빈 라벨 파일 생성
                with open(dst_label, 'w') as f:
                    pass
            
            copied_files[split] += 1
    
    # data.yaml 생성
    class_list = [name.strip() for name in class_names.split(',')]
    data_yaml = {
        'path': '.',
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(class_list),
        'names': class_list,
        
        # 통합 메타데이터
        'created_date': datetime.datetime.now().isoformat(),
        'merge_info': {
            'source_datasets': len(set(f["source_zip"] for f in all_files)),
            'total_source_files': len(all_files),
            'train_split_ratio': train_split,
            'shuffled': shuffle_data
        },
        'dataset_stats': {
            'total_files': len(all_files),
            'train_files': copied_files["train"],
            'val_files': copied_files["val"]
        }
    }
    
    with open(dataset_dir / 'data.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    # ZIP 파일 생성
    zip_filename = f"merged_dataset_{merged_dataset_name}_{timestamp}.zip"
    zip_path = DEFAULT_OUTPUT_DIR / zip_filename
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dataset_dir):
            # macOS 시스템 파일 제외
            if '__MACOSX' in root:
                continue
            for file in files:
                if file.startswith('.') or file.startswith('._'):
                    continue
                file_path = os.path.join(root, file)
                archive_name = os.path.relpath(file_path, dataset_dir)
                zipf.write(file_path, archive_name)
    
    return {
        "zip_filename": zip_filename,
        "total_files": len(all_files),
        "train_files": copied_files["train"],
        "val_files": copied_files["val"],
        "classes": class_list,
        "file_size_mb": round(zip_path.stat().st_size / (1024 * 1024), 2),
        "duplicate_names_resolved": sum(1 for count in filename_counter.values() if count > 0)
    }

@router.post("/merge-datasets", response_model=MergedDatasetResponse)
async def merge_multiple_datasets(
    dataset_zips: List[UploadFile] = File(..., description="통합할 데이터셋 ZIP 파일들 (여러 개)"),
    merged_dataset_name: str = Form(default="merged_dataset", description="통합 데이터셋 이름"),
    train_split: float = Form(default=0.8, description="학습/검증 데이터 분할 비율 (0.0-1.0)"),
    shuffle_data: bool = Form(default=True, description="데이터 셔플 여부"),
    class_names: str = Form(default="damaged_tree", description="클래스 이름들 (쉼표로 구분)")
):
    """
    🔗 여러 데이터셋 ZIP 파일을 통합하여 하나의 대용량 학습 데이터셋 생성
    
    - 각 ZIP의 train/val 데이터를 모두 통합
    - 파일명 중복 자동 해결 (번호 추가)
    - 새로운 train/val 분할 적용
    - 통합 data.yaml 생성
    """
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # 임시 작업 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            print(f"🔗 다중 데이터셋 통합 시작: {len(dataset_zips)}개 ZIP 파일", flush=True)
            
            # Step 1: 각 ZIP 파일 압축 해제 및 분석
            source_datasets = []
            all_train_files = []  # (image_path, label_path, source_info)
            all_val_files = []
            
            for i, zip_file in enumerate(dataset_zips):
                print(f"🔄 ZIP {i+1}/{len(dataset_zips)} 처리 중: {zip_file.filename}", flush=True)
                
                # ZIP 파일 저장
                zip_path = temp_path / f"dataset_{i}_{zip_file.filename}"
                with open(zip_path, "wb") as f:
                    shutil.copyfileobj(zip_file.file, f)
                
                # ZIP 압축 해제
                extract_dir = temp_path / f"extracted_{i}"
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # 데이터셋 구조 분석
                dataset_info = analyze_dataset_structure(extract_dir, zip_file.filename)
                source_datasets.append(dataset_info)
                
                # 파일 수집
                train_files, val_files = collect_dataset_files(extract_dir, dataset_info, i)
                all_train_files.extend(train_files)
                all_val_files.extend(val_files)
            
            print(f"📊 수집 완료: 학습 {len(all_train_files)}개, 검증 {len(all_val_files)}개 파일", flush=True)
            
            # Step 2: 통합 데이터셋 생성
            merged_info = create_merged_dataset(
                all_train_files, all_val_files, temp_path, merged_dataset_name, 
                train_split, shuffle_data, class_names, timestamp
            )
            
            return MergedDatasetResponse(
                success=True,
                message=f"데이터셋 통합 완료! {len(dataset_zips)}개 ZIP → 통합 데이터셋 생성",
                merged_dataset_info=merged_info,
                download_url=f"/api/v1/preprocessing/download/{merged_info['zip_filename']}",
                zip_filename=merged_info['zip_filename'],
                source_datasets=source_datasets
            )
            
    except Exception as e:
        print(f"❌ 데이터셋 통합 오류: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=f"데이터셋 통합 중 오류 발생: {str(e)}")
