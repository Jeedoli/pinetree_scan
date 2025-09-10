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

class PreprocessingResponse(BaseModel):
    success: bool
    message: str
    total_tiles: int
    tiles_with_labels: int
    total_labels: int
    tile_info: List[TileInfo]
    download_url: Optional[str] = None

# 기본 설정
DEFAULT_TILE_SIZE = config.DEFAULT_TILE_SIZE
DEFAULT_BBOX_SIZE = config.DEFAULT_BBOX_SIZE
DEFAULT_CLASS_ID = config.DEFAULT_CLASS_ID
DEFAULT_OUTPUT_DIR = str(config.API_TILES_DIR)

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
    대용량 GeoTIFF 이미지를 타일로 분할하고 YOLO 라벨을 자동 생성합니다.
    
    - **image_file**: 원본 GeoTIFF 이미지 파일 (.tif/.tiff)
    - **tfw_file**: 좌표 변환 파일 (.tfw)
    - **csv_file**: 피해목 위치가 담긴 CSV 파일 (x, y 컬럼 필요)
    - **tile_size**: 타일 한 변 크기 (기본: 1024픽셀)
    - **bbox_size**: 바운딩박스 크기 (기본: 16픽셀)
    - **class_id**: YOLO 클래스 ID (기본: 0)
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
            
            # 출력 디렉토리 생성
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base = os.path.join(DEFAULT_OUTPUT_DIR, f"tiles_{timestamp}")
            output_images = os.path.join(output_base, "images")
            output_labels = os.path.join(output_base, "labels")
            
            os.makedirs(output_images, exist_ok=True)
            os.makedirs(output_labels, exist_ok=True)
            
            # 타일 분할 및 라벨 생성 실행
            tile_info = process_tiles_and_labels(
                image_path, tfw_params, df, 
                output_images, output_labels,
                tile_size, bbox_size, class_id
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
    tile_size: int, bbox_size: int, class_id: int
) -> List[TileInfo]:
    """타일 분할 및 라벨 생성 메인 로직"""
    
    tile_info = []
    
    with rasterio.open(image_path) as src:
        width, height = src.width, src.height
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
                tile_name = f"tile_{tx}_{ty}.tif"
                
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
                    # 라벨 파일 저장
                    label_path = os.path.join(output_labels, tile_name.replace(".tif", ".txt"))
                    with open(label_path, 'w') as f:
                        f.write('\n'.join(labels))
                    
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


def create_tiles_zip(output_base: str, timestamp: str) -> str:
    """생성된 타일과 라벨을 ZIP 파일로 압축"""
    zip_filename = f"tiles_and_labels_{timestamp}.zip"
    zip_path = os.path.join(DEFAULT_OUTPUT_DIR, zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # images 폴더 압축
        images_dir = os.path.join(output_base, "images")
        if os.path.exists(images_dir):
            for root, dirs, files in os.walk(images_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_base)
                    zipf.write(file_path, arcname)
        
        # labels 폴더 압축
        labels_dir = os.path.join(output_base, "labels")
        if os.path.exists(labels_dir):
            for root, dirs, files in os.walk(labels_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, output_base)
                    zipf.write(file_path, arcname)
    
    return zip_path


@router.get("/download/{filename}")
async def download_tiles_zip(filename: str):
    """
    생성된 타일과 라벨 ZIP 파일을 다운로드합니다.
    """
    file_path = os.path.join(DEFAULT_OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
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
    if os.path.exists(DEFAULT_OUTPUT_DIR):
        for item in sorted(os.listdir(DEFAULT_OUTPUT_DIR), reverse=True)[:5]:
            item_path = os.path.join(DEFAULT_OUTPUT_DIR, item)
            if os.path.isdir(item_path):
                images_count = len([f for f in os.listdir(os.path.join(item_path, "images")) 
                                 if f.endswith('.tif')]) if os.path.exists(os.path.join(item_path, "images")) else 0
                labels_count = len([f for f in os.listdir(os.path.join(item_path, "labels"))
                                 if f.endswith('.txt')]) if os.path.exists(os.path.join(item_path, "labels")) else 0
                
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
