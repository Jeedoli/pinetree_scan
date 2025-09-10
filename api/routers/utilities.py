# 유틸리티 API 라우터 - 부가적 기능들
# verify_gps_to_pixel.py, csv_to_yolo_label.py, roboflow_csv_to_yolo.py 기반

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import tempfile
import shutil
import pandas as pd
import numpy as np
import cv2
import rasterio
import pyproj
import datetime
from pathlib import Path
import sys

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from .. import config

router = APIRouter()

# 응답 모델 정의
class GPSVerificationResult(BaseModel):
    latitude: float
    longitude: float
    pixel_x: int
    pixel_y: int
    is_within_bounds: bool

class GPSVerificationResponse(BaseModel):
    success: bool
    message: str
    image_info: Dict[str, Any]
    verification_results: List[GPSVerificationResult]
    output_image_url: Optional[str] = None

class YOLOLabelResult(BaseModel):
    original_coordinates: Dict[str, float]
    yolo_format: str
    is_within_image: bool

class YOLOConversionResponse(BaseModel):
    success: bool
    message: str
    total_objects: int
    converted_objects: int
    label_results: List[YOLOLabelResult]
    download_url: Optional[str] = None

class RoboflowConversionResponse(BaseModel):
    success: bool
    message: str
    processed_images: int
    total_labels: int
    conversion_summary: Dict[str, int]
    download_url: Optional[str] = None

# 기본 설정
DEFAULT_UTILS_DIR = config.API_OUTPUT_BASE / "utilities"

@router.post("/verify_gps_to_pixel", response_model=GPSVerificationResponse)
async def verify_gps_to_pixel_coordinates(
    image_file: UploadFile = File(..., description="검증할 이미지 파일 (GeoTIFF 권장)"),
    csv_file: UploadFile = File(..., description="GPS 좌표가 담긴 CSV 파일 (latitude, longitude 컬럼 필요)"),
    box_size: int = Form(default=20, description="표시할 박스 크기 (픽셀)")
):
    """
    GPS 좌표를 이미지 픽셀 좌표로 변환하여 검증합니다.
    원본 스크립트: verify_gps_to_pixel.py
    
    - **image_file**: 검증할 이미지 파일 (GeoTIFF 형식 권장)
    - **csv_file**: GPS 좌표 CSV (latitude, longitude 컬럼 필요)  
    - **box_size**: 표시할 검증 박스 크기 (기본: 20픽셀)
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
        
        # 출력 디렉토리 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_UTILS_DIR / f"gps_verification_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 임시 디렉토리에서 파일 처리
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # 파일 저장
            image_path = os.path.join(temp_dir, image_file.filename)
            csv_path = os.path.join(temp_dir, csv_file.filename)
            
            with open(image_path, "wb") as f:
                shutil.copyfileobj(image_file.file, f)
            with open(csv_path, "wb") as f:
                shutil.copyfileobj(csv_file.file, f)
            
            # CSV 파일 검증 및 로드
            try:
                df = pd.read_csv(csv_path)
                required_cols = ['latitude', 'longitude']
                if not all(col in df.columns for col in required_cols):
                    raise HTTPException(
                        status_code=400,
                        detail=f"CSV 파일에 필수 컬럼이 없습니다: {required_cols}"
                    )
            except pd.errors.EmptyDataError:
                raise HTTPException(status_code=400, detail="CSV 파일이 비어있습니다.")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"CSV 파일 읽기 실패: {str(e)}")
            
            # 이미지 로드 및 좌표 변환
            img = cv2.imread(image_path)
            if img is None:
                raise HTTPException(status_code=400, detail="이미지 파일을 읽을 수 없습니다.")
            
            verification_results = []
            img_height, img_width = img.shape[:2]
            
            # 이미지 정보 수집
            image_info = {
                "filename": image_file.filename,
                "width": img_width,
                "height": img_height,
                "channels": img.shape[2] if len(img.shape) > 2 else 1
            }
            
            # GPS 좌표 처리
            with rasterio.open(image_path) as src:
                # 이미지에 CRS 정보가 있는지 확인
                has_crs = src.crs is not None
                image_info["has_crs"] = has_crs
                image_info["crs"] = str(src.crs) if has_crs else None
                
                for _, row in df.iterrows():
                    lat, lon = float(row['latitude']), float(row['longitude'])
                    
                    try:
                        if has_crs and not src.crs.to_string().startswith("EPSG:4326"):
                            # CRS 변환 필요한 경우
                            transformer = pyproj.Transformer.from_crs(
                                "EPSG:4326", src.crs, always_xy=True
                            )
                            x_img, y_img = transformer.transform(lon, lat)
                            col, row_idx = src.index(x_img, y_img)
                        else:
                            # 직접 변환
                            col, row_idx = src.index(lon, lat)
                        
                        # 이미지 경계 내 확인
                        is_within = (0 <= col < img_width and 0 <= row_idx < img_height)
                        
                        verification_results.append(GPSVerificationResult(
                            latitude=lat,
                            longitude=lon,
                            pixel_x=int(col),
                            pixel_y=int(row_idx),
                            is_within_bounds=is_within
                        ))
                        
                        # 박스 그리기 (경계 내에 있는 경우만)
                        if is_within:
                            x1 = max(0, int(col) - box_size // 2)
                            y1 = max(0, int(row_idx) - box_size // 2)
                            x2 = min(img_width, int(col) + box_size // 2)
                            y2 = min(img_height, int(row_idx) + box_size // 2)
                            
                            # 빨간 박스 그리기
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            
                            # 좌표 텍스트 추가
                            cv2.putText(
                                img, f"({lat:.6f}, {lon:.6f})", 
                                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, (0, 0, 255), 1
                            )
                        
                    except Exception as e:
                        print(f"좌표 변환 실패 ({lat}, {lon}): {str(e)}")
                        continue
            
            # 결과 이미지 저장
            output_filename = f"gps_verification_{os.path.splitext(image_file.filename)[0]}.png"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), img)
            
            # 통계 계산
            within_bounds_count = sum(1 for r in verification_results if r.is_within_bounds)
            
            return GPSVerificationResponse(
                success=True,
                message=f"GPS 좌표 검증 완료: {len(verification_results)}개 좌표 중 {within_bounds_count}개가 이미지 범위 내에 있습니다.",
                image_info=image_info,
                verification_results=verification_results,
                output_image_url=f"/api/v1/utilities/download/verification/{output_filename}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GPS 검증 중 오류 발생: {str(e)}")


@router.post("/csv_to_yolo_labels", response_model=YOLOConversionResponse)
async def convert_csv_to_yolo_labels(
    csv_file: UploadFile = File(..., description="TM 좌표 CSV 파일 (x, y 컬럼 필요)"),
    tfw_file: UploadFile = File(..., description="좌표 변환 파일 (.tfw)"),
    image_width: int = Form(..., description="이미지 너비 (픽셀)"),
    image_height: int = Form(..., description="이미지 높이 (픽셀)"),
    bbox_size: int = Form(default=16, description="바운딩박스 크기 (픽셀)"),
    class_id: int = Form(default=0, description="YOLO 클래스 ID")
):
    """
    CSV의 TM 좌표를 YOLO 라벨 형식으로 변환합니다.
    원본 스크립트: csv_to_yolo_label.py
    
    - **csv_file**: TM 좌표가 담긴 CSV 파일 (x, y 컬럼 필요)
    - **tfw_file**: 좌표 변환을 위한 TFW 파일
    - **image_width/height**: 대상 이미지 크기
    - **bbox_size**: 바운딩박스 크기 (기본: 16픽셀)
    - **class_id**: YOLO 클래스 ID (기본: 0)
    """
    
    try:
        # 파일 확장자 검증
        if not csv_file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="CSV 파일이 필요합니다.")
        
        if not tfw_file.filename.lower().endswith('.tfw'):
            raise HTTPException(status_code=400, detail="TFW 파일이 필요합니다.")
        
        # 출력 디렉토리 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_UTILS_DIR / f"yolo_conversion_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 임시 디렉토리에서 파일 처리
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # 파일 저장
            csv_path = os.path.join(temp_dir, csv_file.filename)
            tfw_path = os.path.join(temp_dir, tfw_file.filename)
            
            with open(csv_path, "wb") as f:
                shutil.copyfileobj(csv_file.file, f)
            with open(tfw_path, "wb") as f:
                shutil.copyfileobj(tfw_file.file, f)
            
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
            
            # TFW 파일 로드
            try:
                with open(tfw_path, 'r') as f:
                    tfw_params = [float(line.strip()) for line in f.readlines()]
                if len(tfw_params) != 6:
                    raise HTTPException(status_code=400, detail="잘못된 TFW 파일 형식입니다.")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"TFW 파일 읽기 실패: {str(e)}")
            
            # YOLO 라벨 변환
            label_results = []
            yolo_lines = []
            
            for _, row in df.iterrows():
                x_tm, y_tm = float(row['x']), float(row['y'])
                
                # TM 좌표를 픽셀 좌표로 변환
                px, py = tm_to_pixel(x_tm, y_tm, tfw_params)
                
                # YOLO 형식으로 정규화
                x_center = px / image_width
                y_center = py / image_height
                width_norm = bbox_size / image_width
                height_norm = bbox_size / image_height
                
                # 이미지 경계 내 확인
                is_within = (0 <= px < image_width and 0 <= py < image_height)
                
                yolo_format = f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
                
                label_results.append(YOLOLabelResult(
                    original_coordinates={"x": x_tm, "y": y_tm, "px": px, "py": py},
                    yolo_format=yolo_format,
                    is_within_image=is_within
                ))
                
                if is_within:
                    yolo_lines.append(yolo_format)
            
            # YOLO 라벨 파일 저장
            output_filename = f"converted_labels_{timestamp}.txt"
            output_path = output_dir / output_filename
            
            with open(output_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            return YOLOConversionResponse(
                success=True,
                message=f"YOLO 라벨 변환 완료: {len(df)}개 좌표 중 {len(yolo_lines)}개가 이미지 범위 내에서 변환됨",
                total_objects=len(df),
                converted_objects=len(yolo_lines),
                label_results=label_results,
                download_url=f"/api/v1/utilities/download/yolo/{output_filename}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO 변환 중 오류 발생: {str(e)}")


@router.post("/roboflow_to_yolo", response_model=RoboflowConversionResponse)
async def convert_roboflow_to_yolo(
    images: List[UploadFile] = File(..., description="이미지 파일들"),
    labels_csv: UploadFile = File(..., description="Roboflow 라벨 CSV (filename, label, xmin, ymin, xmax, ymax)"),
    create_zip: bool = Form(default=True, description="결과를 ZIP으로 압축할지 여부")
):
    """
    Roboflow 라벨 CSV를 YOLO 형식으로 변환합니다.
    원본 스크립트: roboflow_csv_to_yolo.py
    
    - **images**: 라벨링된 이미지 파일들
    - **labels_csv**: Roboflow 형식의 라벨 CSV (filename, label, xmin, ymin, xmax, ymax)
    - **create_zip**: 결과를 ZIP으로 압축할지 여부
    """
    
    try:
        # CSV 파일 검증
        if not labels_csv.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="라벨 CSV 파일이 필요합니다.")
        
        # 출력 디렉토리 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_UTILS_DIR / f"roboflow_conversion_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 임시 디렉토리에서 파일 처리
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # 이미지 파일들 저장
            image_paths = {}
            for img_file in images:
                if img_file.filename.lower().endswith(tuple(config.ALLOWED_IMAGE_EXTENSIONS)):
                    img_path = os.path.join(temp_dir, img_file.filename)
                    with open(img_path, "wb") as f:
                        shutil.copyfileobj(img_file.file, f)
                    image_paths[img_file.filename] = img_path
            
            # CSV 파일 저장 및 로드
            csv_path = os.path.join(temp_dir, labels_csv.filename)
            with open(csv_path, "wb") as f:
                shutil.copyfileobj(labels_csv.file, f)
            
            try:
                df = pd.read_csv(csv_path)
                required_cols = ['filename', 'label', 'xmin', 'ymin', 'xmax', 'ymax']
                if not all(col in df.columns for col in required_cols):
                    raise HTTPException(
                        status_code=400,
                        detail=f"CSV 파일에 필수 컬럼이 없습니다: {required_cols}"
                    )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"CSV 파일 읽기 실패: {str(e)}")
            
            # 변환 통계
            conversion_summary = {}
            total_labels = 0
            processed_images = 0
            
            # 각 이미지별 라벨 변환
            for filename in df['filename'].unique():
                if filename not in image_paths:
                    continue
                
                img_path = image_paths[filename]
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                image_labels = df[df['filename'] == filename]
                
                yolo_lines = []
                for _, row in image_labels.iterrows():
                    class_id = int(row['label'])
                    xmin, ymin, xmax, ymax = map(float, [
                        row['xmin'], row['ymin'], row['xmax'], row['ymax']
                    ])
                    
                    # YOLO 형식으로 변환 (정규화된 중심좌표와 크기)
                    x_center = ((xmin + xmax) / 2) / w
                    y_center = ((ymin + ymax) / 2) / h
                    width_norm = (xmax - xmin) / w
                    height_norm = (ymax - ymin) / h
                    
                    yolo_lines.append(f"{class_id} {x_center} {y_center} {width_norm} {height_norm}")
                    total_labels += 1
                
                # YOLO 라벨 파일 저장
                label_filename = os.path.splitext(filename)[0] + '.txt'
                label_path = output_dir / label_filename
                
                with open(label_path, 'w') as f:
                    f.write('\n'.join(yolo_lines))
                
                conversion_summary[filename] = len(yolo_lines)
                processed_images += 1
            
            # ZIP 파일 생성 (옵션)
            download_url = None
            if create_zip and processed_images > 0:
                import zipfile
                zip_filename = f"roboflow_to_yolo_{timestamp}.zip"
                zip_path = output_dir.parent / zip_filename
                
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for txt_file in output_dir.glob("*.txt"):
                        zipf.write(txt_file, txt_file.name)
                
                download_url = f"/api/v1/utilities/download/roboflow/{zip_filename}"
            
            return RoboflowConversionResponse(
                success=True,
                message=f"Roboflow → YOLO 변환 완료: {processed_images}개 이미지, {total_labels}개 라벨",
                processed_images=processed_images,
                total_labels=total_labels,
                conversion_summary=conversion_summary,
                download_url=download_url
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Roboflow 변환 중 오류 발생: {str(e)}")


def tm_to_pixel(x: float, y: float, tfw: List[float]) -> tuple:
    """TM 좌표를 이미지 픽셀 좌표로 변환"""
    A, D, B, E, C, F = tfw
    px = (x - C) / A
    py = (y - F) / E
    return px, py


# 다운로드 엔드포인트들
@router.get("/download/verification/{filename}")
async def download_verification_result(filename: str):
    """GPS 검증 결과 이미지 다운로드"""
    # 최신 검증 결과 폴더에서 파일 찾기
    verification_dirs = [d for d in DEFAULT_UTILS_DIR.iterdir() 
                        if d.is_dir() and d.name.startswith("gps_verification_")]
    
    for verification_dir in sorted(verification_dirs, reverse=True):
        file_path = verification_dir / filename
        if file_path.exists():
            return FileResponse(
                path=str(file_path),
                filename=filename,
                media_type='image/png'
            )
    
    raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")


@router.get("/download/yolo/{filename}")
async def download_yolo_labels(filename: str):
    """YOLO 라벨 파일 다운로드"""
    # 최신 변환 결과 폴더에서 파일 찾기
    conversion_dirs = [d for d in DEFAULT_UTILS_DIR.iterdir()
                      if d.is_dir() and d.name.startswith("yolo_conversion_")]
    
    for conversion_dir in sorted(conversion_dirs, reverse=True):
        file_path = conversion_dir / filename
        if file_path.exists():
            return FileResponse(
                path=str(file_path),
                filename=filename,
                media_type='text/plain'
            )
    
    raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")


@router.get("/download/roboflow/{filename}")
async def download_roboflow_conversion(filename: str):
    """Roboflow 변환 결과 ZIP 다운로드"""
    file_path = DEFAULT_UTILS_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='application/zip'
    )


@router.get("/status")
async def utilities_status():
    """유틸리티 서비스 상태 정보"""
    
    # 최근 작업 내역 조회
    recent_operations = []
    
    if DEFAULT_UTILS_DIR.exists():
        for operation_dir in sorted(DEFAULT_UTILS_DIR.iterdir(), reverse=True)[:10]:
            if operation_dir.is_dir():
                file_count = len(list(operation_dir.glob("*")))
                recent_operations.append({
                    "operation": operation_dir.name,
                    "timestamp": operation_dir.stat().st_mtime,
                    "file_count": file_count
                })
    
    return {
        "service": "utilities",
        "status": "active",
        "available_tools": [
            "GPS 좌표 검증 (verify_gps_to_pixel)",
            "CSV → YOLO 라벨 변환 (csv_to_yolo_labels)",
            "Roboflow → YOLO 변환 (roboflow_to_yolo)"
        ],
        "recent_operations": recent_operations,
        "supported_formats": {
            "images": list(config.ALLOWED_IMAGE_EXTENSIONS),
            "coordinates": [".csv", ".tfw"],
            "output": [".png", ".txt", ".zip"]
        }
    }
