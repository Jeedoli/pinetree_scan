# 🌲 Pinetree Scan: 소나무재선충병 피해목 자동 탐지 파이프라인

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/YOLOv8-ultralytics-yellowgreen?logo=github" />
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/rasterio-%23007396.svg?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/OS-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey" />
</p>

> **드론/항공 GeoTIFF 이미지에서 소나무재선충병 피해목을 딥러닝(YOLO)으로 자동 탐지하고, 피해목 위치를 위경도(GPS)로 변환·CSV로 저장하는 공간정보 기반 자동화 파이프라인. REST API를 통해 스웨거 테스트 가능**

---

## 🧰 기술스택

| 구분 | 내용 |
|------|------|
| 언어 | Python 3.10+ |
| 딥러닝 | Ultralytics YOLOv8, PyTorch |
| API 서버 | FastAPI, Uvicorn |
| 공간정보 | rasterio, affine, pyproj |
| 데이터 처리 | pandas, numpy, OpenCV |
| 의존성 관리 | Poetry |
| 기타 | pydantic, python-multipart |

---

---

## 🚀 프로젝트 한눈에 보기

- **목표**: 공간정보(GeoTIFF, tfw, prj)와 피해목 TM좌표(csv)만으로, 소나무재선충병 피해목을 YOLO로 자동 탐지 및 위치(GPS) 변환
- **데이터**: 고도 500미터 이상 드론으로 촬영한 산림 이미지
- **제공 형태**: 
  - **스탠드얼론 스크립트**: 로컬 실행용 Python 스크립트들
  - **REST API 서버**: FastAPI 기반 웹 서비스 (대용량 파일 지원, 최대 2GB)
- **주요 기능**:
  - 대용량 GeoTIFF → 타일 분할(자동)
  - 피해목 TM좌표 → 픽셀 변환 → YOLO 라벨 자동 생성
  - YOLO 학습/추론/CSV 결과 자동화
  - 탐지 결과를 위경도(GPS)로 변환해 CSV로 저장
  - GPS 좌표 검증 및 다양한 라벨 형식 변환 유틸리티
- **성능**: mAP 0.633으로 실용적 수준의 탐지 정확도 달성
- **API 문서**: Swagger UI 및 ReDoc 자동 생성

---

## 🗂️ 프로젝트 구조
```
pinetree_scan/
├── api/                           # 🚀 FastAPI 서버
│   ├── main.py                    # API 메인 애플리케이션
│   ├── config.py                  # 설정 및 환경 변수
│   ├── routers/                   # API 엔드포인트 라우터들
│   │   ├── inference.py           # 추론/탐지 API
│   │   ├── preprocessing.py       # 전처리 API
│   │   ├── visualization.py       # 시각화 API
│   │   └── utilities.py           # 유틸리티 API
│   └── README.md                  # API 사용 가이드
├── scripts/                       # 🔧 원본 스크립트들
│   ├── yolo_infer_to_gps.py      # YOLO 추론+GPS 변환
│   ├── tile_and_label.py         # 타일 분할+라벨 생성
│   ├── mark_inference_boxes.py   # 결과 시각화
│   ├── verify_gps_to_pixel.py    # GPS 좌표 검증
│   ├── csv_to_yolo_label.py      # CSV→YOLO 변환
│   └── roboflow_csv_to_yolo.py   # Roboflow→YOLO 변환
├── data/
│   ├── api_outputs/               # 🆕 API 결과 저장소
│   │   ├── inference/             # 추론 결과 CSV들
│   │   ├── tiles/                 # 타일 분할 결과들
│   │   ├── visualizations/        # 시각화 결과들
│   │   └── utilities/             # 유틸리티 결과들
│   ├── training_images/           # 원본 GeoTIFF, TFW, CSV
│   └── data.yaml                  # YOLO 데이터셋 설정
├── models/                        # 학습된 YOLO 모델들
│   └── colab_yolo/best.pt        # 메인 탐지 모델 (mAP: 0.633)
├── results/                       # 학습 결과들
└── pyproject.toml                 # Poetry 의존성 관리
```

---

## � 빠른 시작

### API 서버 실행 (권장)

```bash
# 1. 의존성 설치
poetry install

# 2. API 서버 실행
cd /path/to/pinetree_scan
poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 3. API 문서 확인
# 브라우저에서 http://localhost:8000/docs 접속
```

### 스탠드얼론 스크립트 실행

```bash
# 기존 방식 (로컬 실행)
poetry run python scripts/yolo_infer_to_gps.py --weights models/colab_yolo/best.pt --source data/images
```

---

## 🌐 FastAPI REST API

### API 서버 특징
- **대용량 파일 지원**: 최대 2GB 업로드 (GeoTIFF 타일화 지원)
- **자동 문서 생성**: Swagger UI & ReDoc
- **타임스탬프 파일명**: 결과 파일 중복 방지
- **ZIP 압축 지원**: 대량 결과 파일 압축 다운로드
- **실시간 상태 확인**: 각 서비스별 상태 API

### 🎯 API 엔드포인트 (총 17개)

#### **추론/탐지 API** (`/api/v1/inference`)
```bash
POST /api/v1/inference/detect           # 이미지 → YOLO 탐지 → GPS 변환
GET  /api/v1/inference/download/{file}  # 결과 CSV 다운로드
GET  /api/v1/inference/models           # 사용 가능한 모델 목록
```

#### **전처리 API** (`/api/v1/preprocessing`)  
```bash
POST /api/v1/preprocessing/tile_and_label    # GeoTIFF → 타일분할 + 라벨생성
GET  /api/v1/preprocessing/download/{file}   # 타일 결과 ZIP 다운로드
GET  /api/v1/preprocessing/status            # 전처리 서비스 상태
```

#### **시각화 API** (`/api/v1/visualization`)
```bash
POST /api/v1/visualization/mark_boxes        # 추론 결과 바운딩박스 마킹
POST /api/v1/visualization/single_image      # 단일 이미지 실시간 마킹
GET  /api/v1/visualization/download/{file}   # 시각화 결과 다운로드
GET  /api/v1/visualization/status            # 시각화 서비스 상태
```

#### **유틸리티 API** (`/api/v1/utilities`)
```bash
POST /api/v1/utilities/verify_gps_to_pixel   # GPS 좌표 검증
POST /api/v1/utilities/csv_to_yolo_labels    # CSV → YOLO 라벨 변환  
POST /api/v1/utilities/roboflow_to_yolo      # Roboflow → YOLO 변환
GET  /api/v1/utilities/download/{type}/{file} # 유틸리티 결과 다운로드
GET  /api/v1/utilities/status                # 유틸리티 서비스 상태
```

### 📋 API 사용 예시

#### cURL로 피해목 탐지
```bash
curl -X POST "http://localhost:8000/api/v1/inference/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@sample_image.tif" \
  -F "confidence=0.05" \
  -F "save_csv=true"
```

#### Python requests로 타일 분할
```python
import requests

url = "http://localhost:8000/api/v1/preprocessing/tile_and_label"
files = {
    "image_file": open("sample.tif", "rb"),
    "tfw_file": open("sample.tfw", "rb"), 
    "csv_file": open("sample.csv", "rb")
}
data = {"tile_size": 1024, "bbox_size": 16}

response = requests.post(url, files=files, data=data)
result = response.json()

# 결과 다운로드
if result.get("download_url"):
    download_url = f"http://localhost:8000{result['download_url']}"
    zip_response = requests.get(download_url)
    with open("tiles_result.zip", "wb") as f:
        f.write(zip_response.content)
```

### 🏠 API 접속 주소
- **메인**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **헬스체크**: http://localhost:8000/health

---

## �🛠️ 스크립트 기반 파이프라인 (기존 방식)

### 1. 타일 분할 및 라벨 자동 생성
- `scripts/tile_and_label.py` 실행
- 원본 tif를 타일(1024/4096px)로 분할, 각 타일별 YOLO 라벨(txt) 자동 생성
- **rasterio** 라이브러리 사용 (Pillow 아님)

### 2. YOLO 학습
- data.yaml에서 `train: tiles/images`, `val: tiles/images`로 지정
- `yolo detect train ...` 명령으로 학습
- **모델**: YOLOv8s 사용
- **학습 환경**: Google Colab 환경에서 진행
- **최종 성능**: mAP 0.633 달성 (고도 500m+ 드론 이미지 기준으로 실용적 수준)
- 모델 파일(`best.pt`, `last.pt`)은 `results/프로젝트명/weights/`에 저장

### 3. 피해목 탐지 및 GPS 변환
- `scripts/yolo_infer_to_gps.py` 실행
- Google Colab에서 학습된 YOLOv8s 모델(`best.pt`, mAP: 0.633)로 타일/원본 이미지 추론
- 탐지된 피해목의 중심 픽셀좌표를 위경도(GPS)로 변환
- **결과 CSV는 실행 시점의 년월일시간이 포함된 파일명으로 자동 저장**
  - 예: `sample01_gps_20250718_153012.csv`

---

## 📊 결과 예시

| 파일/폴더 | 설명 |
|-----------|------|
| data/training_images/sample01.tif | 원본 GeoTIFF 이미지 |
| data/training_images/sample01.tfw | 월드파일(좌표계) |
| data/training_images/sample01.csv | 피해목 중심 TM좌표(csv) |
| data/tiles/images/ | 타일 이미지(자동 생성) |
| data/tiles/labels/ | 타일별 YOLO 라벨(txt, 자동 생성) |
| results/프로젝트명/weights/best.pt | 학습된 YOLOv8s 모델 (Google Colab, mAP: 0.633) |
| data/results/sample01_gps_YYYYMMDD_HHMMSS.csv | 탐지 결과(GPS 좌표, 자동 생성) |

---


## 실행 예시

---

## YOLO 학습 로그 해석 (지표 설명)

| 항목         | 의미                                                         | 예시 값         |
|--------------|--------------------------------------------------------------|-----------------|
| **Epoch**    | 현재 학습 반복(에폭) 번호 / 전체 에폭 수                      | 1/50, 2/50 등   |
| **GPU_mem**  | 사용된 GPU 메모리(GB) (CPU만 사용 시 0G)                      | 0G              |
| **box_loss** | 바운딩 박스 회귀(위치) 손실값 (작을수록 좋음)                 | 0.5232, 4.735   |
| **cls_loss** | 클래스 분류 손실값 (작을수록 좋음)                            | 101.3, 4.806    |
| **dfl_loss** | Distribution Focal Loss (YOLOv8의 박스 품질 손실, 작을수록 좋음) | 0.09157, 0.6954 |
| **Instances**| 해당 에폭에서 처리된 객체(라벨) 수                            | 164, 2081 등    |
| **Size**     | 입력 이미지 크기(보통 640)                                    | 640             |

#### 평가 지표 (각 에폭 후 출력)

| 항목         | 의미                                                         | 예시 값         |
|--------------|--------------------------------------------------------------|-----------------|
| **Class**    | 평가 대상 클래스(여기선 all=전체)                             | all             |
| **Images**   | 평가에 사용된 이미지 수                                       | 40              |
| **Instances**| 평가에 사용된 전체 객체(라벨) 수                              | 16602           |
| **Box(P)**   | 박스 정밀도(Precision, 1에 가까울수록 좋음)                   | 0, 0.00025      |
| **R**        | 박스 재현율(Recall, 1에 가까울수록 좋음)                      | 0, 0.000181     |
| **mAP50**    | mAP@0.5, 평균정확도(0.5 IOU 기준, 1에 가까울수록 좋음)        | 0, 0.000126     |
| **mAP50-95** | mAP@[.5:.95], 더 엄격한 평균정확도(1에 가까울수록 좋음)        | 0, 5.45e-05     |

> - box_loss, cls_loss, dfl_loss: 낮을수록 좋음(학습이 잘 되고 있다는 신호)
> - Box(P), R, mAP50, mAP50-95: 0에서 점차 증가하면 정상(1에 가까울수록 성능이 좋음)
> - 초기에는 모두 0이 나와도 정상 (에폭이 쌓이면 점차 올라감)


### 타일 분할 및 라벨 생성
```bash
python3 scripts/tile_and_label.py
```

### YOLO 학습
```bash
yolo detect train data=data/data.yaml model=yolov8s.pt epochs=50 imgsz=640 project=results name=pinetree-damage-tiles4096
```

### 피해목 탐지 및 GPS 변환
```bash
python3 scripts/yolo_infer_to_gps.py --weights results/프로젝트명/weights/best.pt --source data/tiles/images
# 결과: data/results/sample01_gps_YYYYMMDD_HHMMSS.csv
```

---

## 📜 스크립트 상세 설명

> **참고**: 모든 스크립트는 FastAPI로 대체되었지만, 스탠드얼론 실행 및 참고용으로 유지됩니다.

### 🎯 핵심 스크립트 (API로 대체됨)

#### 1. `yolo_infer_to_gps.py` → `/api/v1/inference/detect`
- **기능**: YOLO 추론 결과를 CSV로 저장하며, YOLO 형식 좌표를 위경도(GPS)로 변환
- **API 장점**: 다중 파일 업로드, 실시간 결과 확인, 웹 인터페이스 지원
- **스크립트 사용법**:
  ```bash
  poetry run python scripts/yolo_infer_to_gps.py --weights models/colab_yolo/best.pt --source data/images
  ```

#### 2. `tile_and_label.py` → `/api/v1/preprocessing/tile_and_label`
- **기능**: 대용량 GeoTIFF 이미지를 타일로 분할하고, TM 좌표를 YOLO 라벨로 변환
- **API 장점**: 2GB 대용량 파일 지원, ZIP 자동 압축, 진행 상태 확인
- **스크립트 사용법**:
  ```bash
  poetry run python scripts/tile_and_label.py
  # (소스 코드 내에서 경로 수정 필요)
  ```

#### 3. `mark_inference_boxes.py` → `/api/v1/visualization/mark_boxes`
- **기능**: YOLO 추론 결과를 이미지에 시각화하여 저장
- **API 장점**: 색상 커스터마이징, 실시간 단일 이미지 처리, 자동 ZIP 생성
- **스크립트 사용법**:
  ```bash
  poetry run python scripts/mark_inference_boxes.py --tiles_dir data/tiles/images --csv data/results.csv
  ```

### 🔧 유틸리티 스크립트 (API로 통합됨)

#### 4. `verify_gps_to_pixel.py` → `/api/v1/utilities/verify_gps_to_pixel`
- **기능**: GPS 좌표를 이미지 픽셀 좌표로 변환하여 검증
- **API 장점**: 웹에서 즉시 결과 확인, 자동 검증 리포트 생성

#### 5. `csv_to_yolo_label.py` → `/api/v1/utilities/csv_to_yolo_labels`  
- **기능**: CSV의 TM 좌표를 YOLO 라벨 형식으로 변환
- **API 장점**: 실시간 변환 결과 확인, 자동 오류 검증

#### 6. `roboflow_csv_to_yolo.py` → `/api/v1/utilities/roboflow_to_yolo`
- **기능**: Roboflow 라벨 CSV를 YOLO 형식으로 변환
- **API 장점**: 다중 이미지 일괄 처리, 변환 통계 제공

### 📊 시각화 결과 예시
아래는 추론 결과를 시각화한 이미지 예시입니다.
<img width="1024" height="1024" alt="sample01_8_7_marked" src="https://github.com/user-attachments/assets/1507422b-e1ad-4811-ab2f-41350473894d" />

  

---

## � 시스템 요구사항

### 기본 환경
- **Python**: 3.8 이상 (3.9+ 권장)
- **메모리**: 8GB 이상 (대용량 파일 처리용)
- **저장공간**: 10GB 이상 여유 공간
- **운영체제**: macOS, Linux, Windows 10/11

### API 서버 (FastAPI)
- **파일 업로드**: 최대 2GB 지원
- **동시 요청**: 멀티프로세싱 지원
- **브라우저**: 모던 브라우저 (Chrome, Firefox, Safari 등)
- **포트**: 8000 (기본값, 변경 가능)

### 모델 파일
- **필수**: `models/colab_yolo/best.pt` (YOLO 가중치)
- **지원 형식**: YOLOv8 `.pt` 파일
- **모델 크기**: 일반적으로 50MB~200MB

---

## �💡 FAQ & 참고

### 일반적인 질문
- **Q. API와 스크립트 중 어떤 것을 사용해야 하나요?**
  - **API**: 웹 인터페이스, 대용량 파일, 실시간 처리가 필요한 경우 (권장)
  - **스크립트**: 자동화, 배치 처리, 커스텀 파이프라인이 필요한 경우

- **Q. 결과 CSV는 어떻게 생성되나요?**
  - API: `data/api_outputs/` 폴더에 타임스탬프가 포함된 파일로 저장
  - 스크립트: 기존 방식대로 지정된 출력 경로에 저장

- **Q. 모델 파일은 어디에 저장되나요?**
  - `models/colab_yolo/best.pt`에 위치 (API에서 자동 감지)
  - 다른 경로의 모델 사용 시 설정 파일 수정 필요

### 기술적인 질문
- **Q. API 서버 성능은 어떻게 최적화하나요?**
  - 충분한 메모리 확보 (8GB+)
  - SSD 사용 권장
  - GPU 사용 시 CUDA 호환 환경 구성

- **Q. 대용량 파일 처리 시 주의사항은?**
  - 2GB 이하 파일만 업로드 가능
  - 네트워크 안정성 확보
  - 충분한 저장 공간 확보 (원본의 3배 이상)

- **Q. CSV 데이터가 중첩 저장되나요?**
  - 아니요. 각 처리마다 새로운 파일로 생성됩니다.
  - 타임스탬프로 파일명을 구분합니다.

---

## 📚 참고자료
- [ultralytics YOLO 공식문서](https://docs.ultralytics.com/)
- [rasterio 공식문서](https://rasterio.readthedocs.io/)
