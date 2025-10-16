# 소나무 피해목 탐지 API 사용 가이드

## 새로 추가된 배치 처리 워크플로우

### 1. 추론용 타일화 API

**엔드포인트**: `POST /api/v1/preprocessing/tile-for-inference`

원본 GeoTIFF 이미지를 추론용으로 타일화합니다. 라벨링 없이 이미지만 분할하며, 지리참조 정보를 보존합니다.

**요청 파라미터**:
- `image_file`: 원본 GeoTIFF 이미지 파일 (.tif/.tiff)
- `tile_size`: 타일 크기 (기본값: 1024픽셀)

**응답**:
```json
{
  "success": true,
  "message": "추론용 타일 분할 완료: 100개 타일 생성",
  "total_tiles": 100,
  "original_size": [4096, 4096],
  "tile_size": 1024,
  "tile_info": [
    {
      "tile_name": "A20250115_tile_0_0.tif",
      "tile_size": [1024, 1024],
      "position": [0, 0],
      "has_georeference": true
    }
  ],
  "download_url": "/api/v1/preprocessing/download/inference_tiles_20250115_123456.zip"
}
```

### 2. 배치 추론 API

**엔드포인트**: `POST /api/v1/inference/detect-all`

여러 이미지에 대한 배치 추론을 수행합니다. ZIP 파일로 업로드된 타일 이미지들을 일괄 처리합니다.

**요청 파라미터**:
- `images_zip`: 추론할 이미지들이 포함된 ZIP 파일
- `model_path`: YOLO 모델 경로 (기본값: yolov8n.pt)
- `confidence`: 탐지 신뢰도 임계값 (기본값: 0.5)
- `iou_threshold`: IoU 임계값 (기본값: 0.8, 중복 탐지 제거용)
- `save_visualization`: 시각화 이미지 저장 여부 (기본값: true)
- `output_tm_coordinates`: TM 좌표 변환 여부 (기본값: true)

**응답**:
```json
{
  "success": true,
  "message": "배치 추론 완료: 95개 이미지 처리, 47개 객체 탐지",
  "total_images": 100,
  "total_detections": 47,
  "processed_images": ["A20250115_tile_0_0.tif", "A20250115_tile_0_1.tif"],
  "failed_images": ["A20250115_tile_0_5.tif: 파일 손상"],
  "csv_file_url": "/api/v1/inference/download/batch_detections_20250115_123456.csv",
  "results_zip_url": "/api/v1/inference/download/batch_results_20250115_123456.zip",
  "merged_visualization": "merged_detection_20250115_123456.jpg"
}
```

## 완전한 워크플로우 예시

### 1단계: 원본 이미지 타일화
```bash
curl -X POST "http://localhost:8000/api/v1/preprocessing/tile-for-inference" \
  -F "image_file=@sample01.tif" \
  -F "tile_size=1024"
```

### 2단계: 타일 ZIP 파일 다운로드
```bash
wget "http://localhost:8000/api/v1/preprocessing/download/inference_tiles_20250115_123456.zip"
```

### 3단계: 배치 추론 실행
```bash
curl -X POST "http://localhost:8000/api/v1/inference/detect-all" \
  -F "images_zip=@inference_tiles_20250115_123456.zip" \
  -F "model_path=results/pinetree-damage-single-final2/weights/best.pt" \
  -F "confidence=0.16" \
  -F "iou_threshold=0.45" \
  -F "save_visualization=true" \
  -F "output_tm_coordinates=true"
```

### 4단계: 결과 다운로드
- **CSV 파일**: 탐지된 객체들의 좌표 정보 (TM 좌표 포함)
- **ZIP 파일**: 개별 타일 시각화 이미지들과 CSV 파일 포함
- **합쳐진 시각화**: 🆕 **실제 타일 이미지들을 합쳐서 원본 크기로 복원한 전체 탐지 결과 이미지** (배경 이미지 포함)
- ZIP 파일: 시각화 이미지들과 CSV 파일 포함

## TM 좌표 출력 형식

배치 추론 결과 CSV는 다음 컬럼들을 포함합니다:

```csv
no,filename,class_id,center_x,center_y,width,height,confidence,tm_x,tm_y
1,A20250115_tile_0_0.tif,0,512.3,678.9,89.2,125.7,0.87,302145.78,538912.34
2,A20250115_tile_0_1.tif,0,234.1,456.2,76.8,98.3,0.91,301987.23,539456.12
```

- `tm_x`, `tm_y`: TM 좌표계 기준 실제 지리 좌표
- 원본 이미지의 지리참조 정보(TFW 파일 또는 GeoTIFF 메타데이터)를 기반으로 자동 변환

## 기존 API 엔드포인트

### 단일 이미지 추론
- `POST /api/v1/inference/detect`: 단일 이미지 탐지
- `GET /api/v1/inference/download/{filename}`: 결과 파일 다운로드

### 훈련용 전처리
- `POST /api/v1/preprocessing/tile-and-label`: 훈련용 타일화 + 라벨링
- `GET /api/v1/preprocessing/download/{filename}`: 전처리 결과 다운로드

### 시스템 정보
- `GET /api/v1/inference/status`: 추론 서비스 상태
- `GET /api/v1/preprocessing/status`: 전처리 서비스 상태
- `GET /api/v1/inference/models`: 사용 가능한 모델 목록

## 참고사항

1. **파일 형식**: GeoTIFF(.tif, .tiff) 권장
2. **메모리 사용량**: 타일 크기와 배치 크기에 따라 조절
3. **처리 시간**: 이미지 개수와 모델 복잡도에 비례
4. **좌표계**: TM 좌표 출력을 위해 원본 이미지에 지리참조 정보 필요
5. **에러 처리**: 개별 이미지 처리 실패시에도 나머지 이미지는 계속 처리

## 임계값 설정 가이드

### 배치 추론 API 기본값
- **confidence**: 0.5 (높은 정확도를 위한 신뢰도)
- **iou_threshold**: 0.8 (엄격한 중복 탐지 제거)

### 값 조정 가이드
- **confidence 높이기 (0.6~0.8)**: 더 확실한 탐지만 선택, 놓치는 객체 증가
- **confidence 낮추기 (0.3~0.4)**: 더 많은 탐지, 오탐지 가능성 증가
- **iou_threshold 높이기 (0.9)**: 중복 탐지를 더 허용, 비슷한 위치의 다중 탐지
- **iou_threshold 낮추기 (0.5~0.6)**: 중복 탐지를 더 엄격하게 제거
