# 시각화 API 라우터 - mark_inference_boxes.py 기반
# 추론 결과를 이미지에 시각화 (바운딩박스 마킹)

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import tempfile
import shutil
import pandas as pd
import cv2
import rasterio
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
class VisualizationResult(BaseModel):
    filename: str
    boxes_drawn: int
    output_filename: str

class VisualizationResponse(BaseModel):
    success: bool
    message: str
    processed_images: int
    results: List[VisualizationResult]
    download_url: Optional[str] = None

# 기본 설정
DEFAULT_OUTPUT_DIR = str(config.API_VISUALIZATION_DIR)
DEFAULT_BOX_COLOR = config.DEFAULT_BOX_COLOR_BGR
DEFAULT_BOX_THICKNESS = config.DEFAULT_BOX_THICKNESS

@router.post("/mark_boxes", response_model=VisualizationResponse)
async def mark_inference_boxes(
    images: List[UploadFile] = File(..., description="시각화할 이미지 파일들"),
    csv_file: UploadFile = File(..., description="추론 결과 CSV 파일"),
    box_color_r: int = Form(default=0, description="박스 색상 R값 (0-255)"),
    box_color_g: int = Form(default=0, description="박스 색상 G값 (0-255)"), 
    box_color_b: int = Form(default=255, description="박스 색상 B값 (0-255)"),
    box_thickness: int = Form(default=2, description="박스 테두리 두께"),
    create_zip: bool = Form(default=True, description="결과를 ZIP으로 압축할지 여부")
):
    """
    이미지에 추론 결과 바운딩박스를 마킹하여 시각화합니다.
    
    - **images**: 시각화할 이미지 파일들 (TIFF, JPG, PNG)
    - **csv_file**: 추론 결과가 담긴 CSV 파일
    - **box_color_r/g/b**: 바운딩박스 색상 RGB 값 (기본: 빨간색)
    - **box_thickness**: 박스 테두리 두께 (기본: 2)
    - **create_zip**: 결과를 ZIP으로 압축할지 여부 (기본: True)
    """
    
    try:
        # CSV 파일 확장자 검증
        if not csv_file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="CSV 파일이 필요합니다.")
        
        # 임시 디렉토리 생성
        with tempfile.TemporaryDirectory() as temp_dir:
            
            # CSV 파일 저장 및 로드
            csv_path = os.path.join(temp_dir, csv_file.filename)
            with open(csv_path, "wb") as f:
                shutil.copyfileobj(csv_file.file, f)
            
            # CSV 파일 읽기 및 검증
            try:
                df = pd.read_csv(csv_path)
                required_columns = ['filename']
                if not all(col in df.columns for col in required_columns):
                    raise HTTPException(
                        status_code=400,
                        detail=f"CSV 파일에 필수 컬럼이 없습니다. 필요: {required_columns}"
                    )
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"CSV 파일 읽기 실패: {str(e)}")
            
            # 출력 디렉토리 생성
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            output_dir = os.path.join(DEFAULT_OUTPUT_DIR, f"marked_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            
            # 박스 색상 설정 (BGR 형식)
            box_color = (box_color_b, box_color_g, box_color_r)
            
            # 각 이미지 처리
            results = []
            for image_file in images:
                # 파일 확장자 검증
                if not image_file.filename.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
                    continue
                
                # 임시 파일로 저장
                temp_image_path = os.path.join(temp_dir, image_file.filename)
                with open(temp_image_path, "wb") as f:
                    shutil.copyfileobj(image_file.file, f)
                
                # 해당 이미지의 추론 결과 필터링
                df_img = df[df['filename'] == image_file.filename]
                
                if df_img.empty:
                    # 결과 없는 이미지는 원본 그대로 저장
                    output_filename = f"{os.path.splitext(image_file.filename)[0]}_no_detection.png"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # 원본 이미지 복사
                    img = load_image_safely(temp_image_path)
                    if img is not None:
                        cv2.imwrite(output_path, img)
                        
                        results.append(VisualizationResult(
                            filename=image_file.filename,
                            boxes_drawn=0,
                            output_filename=output_filename
                        ))
                    continue
                
                # 이미지 로드 및 박스 마킹
                boxes_drawn = mark_boxes_on_image(
                    temp_image_path, df_img, output_dir, 
                    box_color, box_thickness
                )
                
                if boxes_drawn >= 0:  # 성공적으로 처리된 경우
                    output_filename = f"{os.path.splitext(image_file.filename)[0]}_marked.png"
                    
                    results.append(VisualizationResult(
                        filename=image_file.filename,
                        boxes_drawn=boxes_drawn,
                        output_filename=output_filename
                    ))
            
            # ZIP 파일 생성 (옵션)
            download_url = None
            if create_zip and results:
                zip_path = create_visualization_zip(output_dir, timestamp)
                download_url = f"/api/v1/visualization/download/{os.path.basename(zip_path)}"
            
            return VisualizationResponse(
                success=True,
                message=f"시각화 완료: {len(results)}개 이미지 처리됨",
                processed_images=len(results),
                results=results,
                download_url=download_url
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"시각화 중 오류 발생: {str(e)}")


def load_image_safely(image_path: str):
    """이미지를 안전하게 로드하고 형식 변환"""
    try:
        # rasterio로 먼저 시도 (TIFF 파일용)
        if image_path.lower().endswith(('.tif', '.tiff')):
            with rasterio.open(image_path) as src:
                img = src.read()
                img = img.transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
                if img.shape[2] > 3:
                    img = img[:, :, :3]  # 4채널 -> 3채널
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return img
        else:
            # OpenCV로 일반 이미지 로드
            img = cv2.imread(image_path)
            return img
    except:
        # 최후 수단으로 OpenCV 시도
        return cv2.imread(image_path)


def mark_boxes_on_image(
    image_path: str, df_img: pd.DataFrame, output_dir: str,
    box_color: tuple, box_thickness: int
) -> int:
    """이미지에 바운딩박스를 마킹"""
    
    try:
        # 이미지 로드
        img = load_image_safely(image_path)
        if img is None:
            return -1
        
        boxes_drawn = 0
        
        # 각 결과에 대해 박스 그리기
        for _, row in df_img.iterrows():
            
            # 좌표 정보 추출 (여러 형식 지원)
            if all(k in row for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                # 절대 좌표 형식
                xmin, ymin, xmax, ymax = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                
            elif all(k in row for k in ['x', 'y']):
                # 중심점 형식 (고정 크기 박스)
                x, y = int(row['x']), int(row['y'])
                w, h = 20, 20  # 기본 박스 크기
                xmin, ymin, xmax, ymax = x - w//2, y - h//2, x + w//2, y + h//2
                
            elif all(k in row for k in ['center_x', 'center_y', 'width', 'height']):
                # YOLO 정규화 형식
                center_x, center_y = float(row['center_x']), float(row['center_y'])
                width, height = float(row['width']), float(row['height'])
                
                image_height, image_width = img.shape[:2]
                
                xmin = int((center_x - width / 2) * image_width)
                ymin = int((center_y - height / 2) * image_height)
                xmax = int((center_x + width / 2) * image_width)
                ymax = int((center_y + height / 2) * image_height)
                
            else:
                continue  # 좌표 정보 없으면 건너뛰기
            
            # 이미지 경계 내 확인
            image_height, image_width = img.shape[:2]
            if (0 <= xmin < image_width and 0 <= ymin < image_height and
                0 <= xmax < image_width and 0 <= ymax < image_height):
                
                # 박스 그리기
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), box_color, box_thickness)
                boxes_drawn += 1
        
        # 결과 이미지 저장
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_filename = f"{base_name}_marked.png"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, img)
        
        return boxes_drawn
        
    except Exception as e:
        print(f"이미지 처리 중 오류: {str(e)}")
        return -1


def create_visualization_zip(output_dir: str, timestamp: str) -> str:
    """시각화 결과를 ZIP 파일로 압축"""
    zip_filename = f"marked_images_{timestamp}.zip"
    zip_path = os.path.join(DEFAULT_OUTPUT_DIR, zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, file)
    
    return zip_path


@router.get("/download/{filename}")
async def download_visualization_result(filename: str):
    """
    시각화 결과 ZIP 파일을 다운로드합니다.
    """
    file_path = os.path.join(DEFAULT_OUTPUT_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/zip'
    )


@router.post("/single_image")
async def mark_single_image_auto_detect(
    image: UploadFile = File(..., description="자동 탐지 후 시각화할 이미지"),
    box_color_r: int = Form(default=0, description="박스 색상 R값"),
    box_color_g: int = Form(default=255, description="박스 색상 G값"), 
    box_color_b: int = Form(default=0, description="박스 색상 B값"),
    confidence_threshold: float = Form(default=config.DEFAULT_CONFIDENCE, description="신뢰도 임계값"),
    iou_threshold: float = Form(default=config.DEFAULT_IOU_THRESHOLD, description="IoU 임계값 (중복 탐지 제거)")
):
    """
    이미지를 업로드하면 자동으로 YOLO 모델로 탐지하고 바운딩박스를 마킹하여 반환합니다.
    """
    from ultralytics import YOLO
    
    try:
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            
            # 이미지 저장
            temp_input = tmp_file.name.replace('.png', '_input.tmp')
            with open(temp_input, "wb") as f:
                shutil.copyfileobj(image.file, f)
            
            # 이미지 로드
            img = load_image_safely(temp_input)
            if img is None:
                raise HTTPException(status_code=400, detail="이미지를 로드할 수 없습니다.")
            
            # YOLO 모델 로드 및 추론
            model = YOLO(str(config.DEFAULT_MODEL_PATH))
            results = model(temp_input, conf=confidence_threshold, iou=iou_threshold, verbose=False)
            
            box_color = (box_color_b, box_color_g, box_color_r)
            detection_count = 0
            
            # 추론 결과가 있으면 바운딩박스 그리기
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    detection_count = len(boxes)
                    
                    # 각 박스에 대해 그리기
                    for i in range(len(boxes)):
                        # YOLO 형식 (xyxy)을 픽셀 좌표로 변환
                        bbox = boxes.xyxy[i].cpu().numpy().astype(int)
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = model.names[class_id] if hasattr(model, 'names') else f'class_{class_id}'
                        
                        # 바운딩 박스 그리기
                        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, 2)
                        
                        # 라벨 텍스트 (클래스명 + 신뢰도)
                        label = f"{class_name}: {confidence:.2f}"
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        
                        # 라벨 배경 그리기
                        cv2.rectangle(img, 
                                    (bbox[0], bbox[1] - label_size[1] - 10), 
                                    (bbox[0] + label_size[0], bbox[1]), 
                                    box_color, -1)
                        
                        # 라벨 텍스트 그리기
                        cv2.putText(img, label, 
                                  (bbox[0], bbox[1] - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                  (255, 255, 255), 1)
            
            # 결과 이미지 저장
            cv2.imwrite(tmp_file.name, img)
            
            # 임시 입력 파일 삭제
            os.unlink(temp_input)
            
            # 응답 헤더에 탐지 정보 추가
            headers = {
                "X-Detection-Count": str(detection_count),
                "X-Confidence-Threshold": str(confidence_threshold),
                "X-Box-Color": f"RGB({box_color_r},{box_color_g},{box_color_b})"
            }
            
            return FileResponse(
                path=tmp_file.name,
                filename=f"detected_{detection_count}_objects_{image.filename}",
                media_type='image/png',
                headers=headers
            )
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 처리 중 오류: {str(e)}")


def mark_boxes_on_single_image(img, df_results: pd.DataFrame, box_color: tuple):
    """단일 이미지에 박스 마킹 (인플레이스 수정)"""
    
    for _, row in df_results.iterrows():
        
        if all(k in row for k in ['center_x', 'center_y', 'width', 'height']):
            # YOLO 정규화 형식
            center_x, center_y = float(row['center_x']), float(row['center_y'])
            width, height = float(row['width']), float(row['height'])
            
            image_height, image_width = img.shape[:2]
            
            xmin = int((center_x - width / 2) * image_width)
            ymin = int((center_y - height / 2) * image_height)
            xmax = int((center_x + width / 2) * image_width)
            ymax = int((center_y + height / 2) * image_height)
            
            # 이미지 경계 내 확인
            if (0 <= xmin < image_width and 0 <= ymin < image_height and
                0 <= xmax < image_width and 0 <= ymax < image_height):
                
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), box_color, 2)


@router.get("/status")
async def visualization_status():
    """
    시각화 서비스의 상태 정보를 반환합니다.
    """
    # 최근 생성된 시각화 폴더들 조회
    recent_visualizations = []
    if os.path.exists(DEFAULT_OUTPUT_DIR):
        for item in sorted(os.listdir(DEFAULT_OUTPUT_DIR), reverse=True)[:5]:
            item_path = os.path.join(DEFAULT_OUTPUT_DIR, item)
            if os.path.isdir(item_path):
                image_count = len([f for f in os.listdir(item_path) 
                                 if f.endswith(('.png', '.jpg', '.jpeg'))])
                recent_visualizations.append({
                    "folder_name": item,
                    "image_count": image_count
                })
    
    return {
        "service": "visualization",
        "status": "active",
        "recent_visualizations": recent_visualizations,
        "supported_formats": ["TIFF", "JPG", "JPEG", "PNG"],
        "default_settings": {
            "box_color": DEFAULT_BOX_COLOR,
            "box_thickness": DEFAULT_BOX_THICKNESS
        }
    }
