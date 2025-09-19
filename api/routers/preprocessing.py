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

class PreprocessingResponse(BaseModel):
    success: bool
    message: str
    total_tiles: int
    tiles_with_labels: int
    total_labels: int
    tile_info: List[TileInfo]
    download_url: Optional[str] = None

class IntegratedTrainingResponse(BaseModel):
    success: bool
    message: str
    preprocessing_info: Dict
    training_dataset_info: Dict
    download_url: str
    zip_filename: str

class InferenceTilingResponse(BaseModel):
    success: bool
    message: str
    total_tiles: int
    original_size: tuple
    tile_size: int
    tile_info: List[InferenceTileInfo]
    download_url: Optional[str] = None

# 기본 설정
DEFAULT_TILE_SIZE = config.DEFAULT_TILE_SIZE
DEFAULT_BBOX_SIZE = config.DEFAULT_BBOX_SIZE
DEFAULT_CLASS_ID = config.DEFAULT_CLASS_ID
DEFAULT_OUTPUT_DIR = Path(config.API_TILES_DIR)  # Path 객체로 변경

@router.post("/tile_and_label", response_model=PreprocessingResponse)
async def create_tiles_and_labels(
    image_file: UploadFile = File(..., description="원본 GeoTIFF 이미지"),
    tfw_file: UploadFile = File(..., description="좌표 변환 파일 (.tfw)"),
    csv_file: UploadFile = File(..., description="피해목 위치 CSV 파일"),
    tile_size: int = Form(default=DEFAULT_TILE_SIZE, description="타일 크기 (픽셀)"),
    bbox_size: int = Form(default=DEFAULT_BBOX_SIZE, description="바운딩박스 크기 (픽셀)"),
    class_id: int = Form(default=DEFAULT_CLASS_ID, description="YOLO 클래스 ID")
):
    """
    ⚠️ **레거시 API**: 이 API는 하위 호환성을 위해 유지됩니다.
    **새로운 프로젝트는 `/create_complete_training_dataset` API 사용을 권장합니다.**
    
    대용량 GeoTIFF 이미지를 타일로 분할하고 YOLO 라벨을 자동 생성합니다.
    (딥러닝용 ZIP 파일 생성은 별도로 처리해야 함)
    
    **📋 매개변수:**
    - **image_file**: 원본 GeoTIFF 이미지 파일 (.tif/.tiff)
    - **tfw_file**: 좌표 변환 파일 (.tfw)
    - **csv_file**: 피해목 위치가 담긴 CSV 파일 (x, y 컬럼 필요)
    - **tile_size**: 타일 한 변 크기 (기본: 1024픽셀, 세밀한 탐지 최적화)
    - **bbox_size**: 바운딩박스 크기 (기본: 24픽셀, 개별 나무 탐지용)
    - **class_id**: YOLO 클래스 ID (기본: 0)
    
    **🚀 권장 사용법:**
    통합 API `/create_complete_training_dataset`를 사용하면:
    - 타일링 + YOLO 라벨 생성 + 딥러닝용 ZIP 생성을 한 번에 처리
    - Google Colab 최적화된 데이터셋 구조
    - 자동 train/validation 분할
    - README 및 사용법 가이드 포함
    """
    
    try:
        # 파일 확장자 검증
        if not image_file.filename.lower().endswith(('.tif', '.tiff')):
            raise HTTPException(status_code=400, detail="이미지 파일은 TIFF 형식이어야 합니다.")
        
        if not tfw_file.filename.lower().endswith('.tfw'):
            raise HTTPException(status_code=400, detail="좌표 변환 파일은 .tfw 형식이어야 합니다.")
        
        if not csv_file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="피해목 위치 파일은 CSV 형식이어야 합니다.")
        
        # 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # 업로드 파일들을 임시 디렉토리에 저장
            image_path = os.path.join(temp_dir, image_file.filename)
            tfw_path = os.path.join(temp_dir, tfw_file.filename)
            csv_path = os.path.join(temp_dir, csv_file.filename)
            
            # 파일 저장
            with open(image_path, "wb") as f:
                shutil.copyfileobj(image_file.file, f)
            with open(tfw_path, "wb") as f:
                shutil.copyfileobj(tfw_file.file, f)
            with open(csv_path, "wb") as f:
                shutil.copyfileobj(csv_file.file, f)
            
            # TFW 파일 로드
            tfw_params = load_tfw(tfw_path)
            
            # CSV 파일 로드 및 검증
            try:
                df = pd.read_csv(csv_path)
                if 'x' not in df.columns or 'y' not in df.columns:
                    raise HTTPException(
                        status_code=400, 
                        detail="CSV 파일에 'x', 'y' 컬럼이 필요합니다."
                    )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"CSV 파일 읽기 실패: {str(e)}")
            
            # 날짜 기반 접두사 생성
            today = datetime.datetime.now().strftime("%Y%m%d")
            
            # 같은 날짜의 기존 폴더들 확인하여 순차 접두사 결정
            existing_prefixes = []
            if os.path.exists(DEFAULT_OUTPUT_DIR):
                for folder in os.listdir(DEFAULT_OUTPUT_DIR):
                    if folder.startswith(f"tiles_") and today in folder:
                        # tiles_A20250915_ 형태에서 접두사 추출
                        parts = folder.split('_')
                        if len(parts) >= 2 and parts[1].startswith(('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J')):
                            prefix_char = parts[1][0]
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
            output_base = DEFAULT_OUTPUT_DIR / f"tiles_{file_prefix}_{timestamp}"
            output_images = output_base / "images"
            output_labels = output_base / "labels"
            
            output_images.mkdir(parents=True, exist_ok=True)
            output_labels.mkdir(parents=True, exist_ok=True)
            
            # 타일 분할 및 라벨 생성 실행
            tile_info = process_tiles_and_labels(
                image_path, tfw_params, df, 
                str(output_images), str(output_labels),
                tile_size, bbox_size, class_id, file_prefix
            )
            
            # ZIP 파일 생성
            zip_path = create_tiles_zip(output_base, timestamp)
            
            # 통계 계산
            total_tiles = len(tile_info)
            tiles_with_labels = len([t for t in tile_info if t.labels_count > 0])
            total_labels = sum(t.labels_count for t in tile_info)
            
            return PreprocessingResponse(
                success=True,
                message=f"타일 분할 완료: {total_tiles}개 타일 생성, {total_labels}개 라벨 생성",
                total_tiles=total_tiles,
                tiles_with_labels=tiles_with_labels,
                total_labels=total_labels,
                tile_info=tile_info,
                download_url=f"/api/v1/preprocessing/download/{os.path.basename(zip_path)}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"전처리 중 오류 발생: {str(e)}")


def load_tfw(tfw_path: str) -> List[float]:
    """TFW 파일에서 변환 파라미터 읽기"""
    with open(tfw_path, 'r') as f:
        return [float(line.strip()) for line in f.readlines()]


def tm_to_pixel(x: float, y: float, tfw: List[float]) -> tuple:
    """TM 좌표를 이미지 픽셀 좌표로 변환"""
    A, D, B, E, C, F = tfw
    px = (x - C) / A
    py = (y - F) / E
    return px, py


def process_tiles_and_labels(
    image_path: str, tfw_params: List[float], df: pd.DataFrame,
    output_images: str, output_labels: str,
    tile_size: int, bbox_size: int, class_id: int, file_prefix: str
) -> List[TileInfo]:
    """타일 분할 및 라벨 생성 메인 로직"""
    
    tile_info = []
    
    with rasterio.open(image_path) as src:
        width, height = src.width, src.height
        n_tiles_x = int(np.ceil(width / tile_size))
        n_tiles_y = int(np.ceil(height / tile_size))
        total_tiles = n_tiles_x * n_tiles_y
        
        print(f"🎯 이미지 크기: {width}x{height} 픽셀", flush=True)
        print(f"📊 생성할 타일: {n_tiles_x}x{n_tiles_y} = {total_tiles}개", flush=True)
        print(f"🔄 타일링 시작...", flush=True)
        
        processed_tiles = 0
        tiles_with_labels = 0
        
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                processed_tiles += 1
                
                # 진행률 출력 (10% 단위 또는 100타일마다)
                if processed_tiles % max(1, total_tiles // 10) == 0 or processed_tiles % 100 == 0:
                    progress = (processed_tiles / total_tiles) * 100
                    print(f"📈 진행률: {progress:.1f}% ({processed_tiles}/{total_tiles})", flush=True)
                
                x0 = tx * tile_size
                y0 = ty * tile_size
                w = min(tile_size, width - x0)
                h = min(tile_size, height - y0)
                
                window = Window(x0, y0, w, h)
                
                # RGB 3채널만 추출하여 타일 이미지 생성
                tile_img = src.read([1, 2, 3], window=window)
                tile_name = f"{file_prefix}_tile_{tx}_{ty}.tif"
                
                # 타일 내 라벨 생성
                labels = []
                for _, row in df.iterrows():
                    px, py = tm_to_pixel(row['x'], row['y'], tfw_params)
                    
                    # 타일 내 상대좌표로 변환
                    rel_x = px - x0
                    rel_y = py - y0
                    
                    if 0 <= rel_x < w and 0 <= rel_y < h:
                        # YOLO 형식으로 정규화
                        x_center = rel_x / w
                        y_center = rel_y / h
                        bw = bbox_size / w
                        bh = bbox_size / h
                        
                        labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")
                
                # 라벨이 있는 경우에만 저장
                if labels:
                    tiles_with_labels += 1
                    
                    # 라벨 파일 저장
                    label_path = os.path.join(output_labels, tile_name.replace(".tif", ".txt"))
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(labels))
                    
                    # 라벨이 있는 타일 로그 (처음 몇 개만)
                    if tiles_with_labels <= 5:
                        print(f"🏷️ 라벨 타일 생성: {tile_name} ({len(labels)}개 라벨)", flush=True)
                    elif tiles_with_labels % 50 == 0:  # 50개마다 로그
                        print(f"🏷️ 라벨 타일 누적: {tiles_with_labels}개", flush=True)
                    
                    # 타일 이미지 저장
                    tile_path = os.path.join(output_images, tile_name)
                    orig_affine = src.transform
                    tile_affine = orig_affine * Affine.translation(x0, y0)
                    
                    with rasterio.open(
                        tile_path, 'w',
                        driver='GTiff',
                        height=h, width=w, count=3,
                        dtype=src.dtypes[0],
                        crs=src.crs,
                        transform=tile_affine
                    ) as dst:
                        dst.write(tile_img)
                
                # 타일 정보 추가 (라벨 유무 관계없이)
                tile_info.append(TileInfo(
                    tile_name=tile_name,
                    labels_count=len(labels),
                    tile_size=(w, h),
                    position=(tx, ty)
                ))
    
    return tile_info


def create_tiles_zip(output_base: Path, timestamp: str) -> str:
    """생성된 타일과 라벨을 ZIP 파일로 압축"""
    zip_filename = f"tiles_and_labels_{timestamp}.zip"
    zip_path = DEFAULT_OUTPUT_DIR / zip_filename
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # images 폴더 압축
        images_dir = output_base / "images"
        if images_dir.exists():
            for root, dirs, files in os.walk(images_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_base)
                    zipf.write(file_path, arcname)
        
        # labels 폴더 압축
        labels_dir = output_base / "labels"
        if labels_dir.exists():
            for root, dirs, files in os.walk(labels_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_base)
                    zipf.write(file_path, arcname)
    
    return str(zip_path)


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
            "bbox_size": DEFAULT_BBOX_SIZE,
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
                for file in files:
                    file_path = os.path.join(root, file)
                    archive_name = os.path.relpath(file_path, output_base)
                    zipf.write(file_path, archive_name)
    
    return str(zip_path)


@router.post("/create_complete_training_dataset", response_model=IntegratedTrainingResponse)
async def create_complete_training_dataset(
    image_file: UploadFile = File(..., description="처리할 GeoTIFF 이미지 파일"),
    csv_file: UploadFile = File(..., description="GPS 좌표 CSV 파일 (x, y 또는 longitude, latitude 컬럼)"),
    tfw_file: UploadFile = File(..., description="지리참조를 위한 TFW 파일"),
    file_prefix: str = Form(..., description="생성될 타일 파일명 접두사"),
    tile_size: int = Form(default=1024, description="타일 크기 (픽셀)"),
    bbox_size: int = Form(default=config.DEFAULT_BBOX_SIZE, description="바운딩 박스 크기 (픽셀)"),
    class_id: int = Form(default=0, description="YOLO 클래스 ID"),
    train_split: float = Form(default=0.8, description="학습 데이터 비율 (0.0-1.0)"),
    class_names: str = Form(default="damaged_tree", description="클래스 이름 (쉼표로 구분)"),
    max_files: int = Form(default=1000, description="최대 파일 수 제한 (0=무제한)"),
    shuffle_data: bool = Form(default=True, description="데이터 셔플 여부"),
    auto_split: bool = Form(default=True, description="자동 train/val 분할 여부")
):
    """
    🚀 **통합 딥러닝 데이터셋 생성 API** (권장)
    
    **이 API는 다음 작업을 한번에 수행합니다:**
    1. 🖼️ GeoTIFF 이미지를 타일로 분할
    2. 🏷️ GPS 좌표 기반 YOLO 라벨 자동 생성  
    3. 📦 Google Colab 최적화 딥러닝용 ZIP 파일 생성
    4. 📊 Train/Validation 데이터셋 자동 분할
    5. 📋 README 및 사용법 가이드 포함
    
    **📋 매개변수:**
    - **image_file**: 처리할 GeoTIFF 이미지 파일 (.tif/.tiff)
    - **csv_file**: GPS 좌표 CSV 파일 (x,y 또는 longitude,latitude 컬럼 필요)
    - **tfw_file**: 지리참조를 위한 TFW 파일 (.tfw)
    - **file_prefix**: 생성될 타일 파일명 접두사 (예: "A20250919")
    - **tile_size**: 타일 크기 (기본: 1024px, 세밀한 탐지 최적화)
    - **bbox_size**: 바운딩 박스 크기 (기본: 24px, 개별 나무 탐지 최적화)
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
    # Google Colab에서 사용법
    !unzip -q "/content/drive/MyDrive/pinetree_training_dataset_*.zip" -d /content/dataset
    from ultralytics import YOLO
    model = YOLO('yolo11s.pt')
    results = model.train(data='/content/dataset/data.yaml', epochs=200)
    ```
    
    **⚠️ 기존 API와의 차이점:**
    - `/tile_and_label`: 타일링과 라벨링만 수행 (레거시)
    - `/create_complete_training_dataset`: 전체 워크플로우를 한번에 수행 (권장)
    
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
            
            # Step 1: 타일링 및 라벨 생성
            print("🔄 Step 1: 타일링 및 라벨 생성 시작...", flush=True)
            tile_info = process_tiles_and_labels(
                image_path=image_path,
                tfw_params=tfw_params,
                df=df,
                output_images=str(tiles_images_dir),
                output_labels=str(tiles_labels_dir),
                tile_size=tile_size,
                bbox_size=bbox_size,
                class_id=class_id,
                file_prefix=file_prefix
            )
            
            tiles_with_labels = sum(1 for t in tile_info if t.labels_count > 0)
            total_labels = sum(t.labels_count for t in tile_info)
            
            preprocessing_info = {
                "total_tiles": len(tile_info),
                "tiles_with_labels": tiles_with_labels,
                "total_labels": total_labels,
                "tile_size": tile_size,
                "bbox_size": bbox_size,
                "file_prefix": file_prefix
            }
            
            print(f"✅ Step 1 완료: {len(tile_info)}개 타일, {tiles_with_labels}개 라벨 타일, {total_labels}개 총 라벨", flush=True)
            
            # Step 2: 딥러닝용 데이터셋 생성
            print("🔄 Step 2: 딥러닝용 데이터셋 생성 시작...", flush=True)
            
            # 타일 이미지와 라벨 파일 매칭
            image_files = list(tiles_images_dir.glob("*.tif"))
            label_files = list(tiles_labels_dir.glob("*.txt"))
            
            # 매칭되는 파일들만 선택
            matched_files = []
            for img_file in image_files:
                label_file = tiles_labels_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    matched_files.append((img_file, label_file))
            
            print(f"📊 매칭된 파일 쌍: {len(matched_files)}개", flush=True)
            
            if len(matched_files) == 0:
                raise HTTPException(
                    status_code=400, 
                    detail="매칭되는 이미지-라벨 파일 쌍을 찾을 수 없습니다."
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
                    'bbox_size': bbox_size
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
            
            # Step 3: ZIP 파일 생성
            print("🔄 Step 3: ZIP 파일 생성 시작...", flush=True)
            zip_filename = f"complete_training_dataset_{file_prefix}_{timestamp}.zip"
            zip_path = DEFAULT_OUTPUT_DIR / zip_filename
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 데이터셋 디렉토리 전체를 압축
                for root, dirs, files in os.walk(dataset_dir):
                    for file in files:
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
            
            print(f"✅ 통합 처리 완료!", flush=True)
            print(f"📦 ZIP 파일: {zip_filename} ({dataset_info['file_size_mb']}MB)", flush=True)
            
            return IntegratedTrainingResponse(
                success=True,
                message=f"통합 데이터셋 생성 완료! 총 {len(matched_files)}개 파일, {total_labels}개 라벨",
                preprocessing_info=preprocessing_info,
                training_dataset_info=dataset_info,
                download_url=download_url,
                zip_filename=zip_filename
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
