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
from ultralytics import YOLO
import datetime
import json
from pathlib import Path
import sys

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

class InferenceResponse(BaseModel):
    success: bool
    message: str
    detected_count: int
    results: List[DetectionResult]
    csv_file_url: Optional[str] = None
    visualization_images: Optional[List[str]] = None  # 시각화 이미지 URL 리스트

# 기본 설정
DEFAULT_WEIGHTS = str(config.DEFAULT_MODEL_PATH)
DEFAULT_OUTPUT_DIR = str(config.API_RESULTS_DIR)

def draw_bounding_boxes_on_image(image, results, output_path: str):
    """바운딩 박스를 이미지에 그리고 저장하는 함수 - 소나무 전용 최적화"""
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
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
                            
                            result = DetectionResult(
                                filename=file.filename,
                                class_id=class_id,
                                center_x=x_center,
                                center_y=y_center,
                                width=width,
                                height=height,
                                confidence=conf_score
                            )
                            all_results.append(result)
                            damaged_count += 1
                
                # 시각화 이미지 생성 및 저장
                if save_visualization:
                    # 해당 파일의 탐지 결과만 필터링
                    file_results = [r for r in all_results if r.filename == file.filename]
                    
                    if file_results:  # 탐지된 객체가 있는 경우에만 시각화
                        # 바운딩 박스가 그려진 이미지 생성
                        annotated_img = draw_detection_boxes(img, file_results, file.filename)
                        
                        # 시각화 디렉토리 생성
                        vis_dir = str(config.API_VISUALIZATION_DIR)
                        os.makedirs(vis_dir, exist_ok=True)
                        
                        # 시각화 이미지 파일명 생성 (세션 타임스탬프 사용)
                        name_without_ext = os.path.splitext(file.filename)[0]
                        vis_filename = f"{name_without_ext}_detected_{session_timestamp}.jpg"
                        vis_path = os.path.join(vis_dir, vis_filename)
                        
                        # 시각화 이미지 저장
                        cv2.imwrite(vis_path, annotated_img)
                        
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
            
            # DataFrame으로 변환 후 저장
            df_results = pd.DataFrame([result.dict() for result in all_results])
            df_results.to_csv(csv_path, index=False)
            csv_file_url = f"/api/v1/inference/download/{csv_filename}"
        
        return InferenceResponse(
            success=True,
            message=f"탐지 완료: {damaged_count}개의 피해목을 발견했습니다.",
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
