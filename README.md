# 🌲 Pinetree Scan: 소나무 피해목 AI 탐지 시스템

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/YOLOv11s-ultralytics-yellowgreen?logo=github" />
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white" />
  <img src="https://img.shields.io/badge/rasterio-%23007396.svg?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/mAP-0.7+-success" />
  <img src="https://img.shields.io/badge/Multi--Scale-Adaptive-orange" />
</p>

> **드론/항공 이미지에서 YOLOv11s + 멀티스케일 적응적 바운딩박스로 소나무 피해목을 자동 탐지하고, GPS 좌표로 변환하여 저장하는 AI 기반 공간정보 시스템**

---

## 🚀 주요 특징

- 🎯 **멀티스케일 적응적 바운딩박스**: GPS 밀도 기반으로 16~128px 크기 자동 조절
- 🧠 **지능형 데이터 생성**: 피해목 분포에 따른 적응적 라벨 크기 결정
- 🌐 **Google Colab 완벽 지원**: 클라우드 GPU로 무료 학습 환경 제공
- 🔥 **FastAPI REST API**: 대용량 파일(2GB) 지원, Swagger UI 자동 문서화
- 📊 **다중 지역 데이터셋 병합**: 여러 지역 데이터를 통합하여 robust한 모델 학습
- 🗺️ **정밀 GPS 변환**: GeoTIFF 공간정보 기반 위경도 좌표 자동 변환
- 📈 **개선된 성능**: 캐시 버그 수정 및 멀티스케일로 mAP 0.7+ 달성

---

## 🧰 기술 스택

| 구분 | 기술 |
|------|------|
| **AI 모델** | YOLOv11s, Ultralytics, PyTorch |
| **클라우드 학습** | Google Colab, GPU T4/V100 |
| **API 서버** | FastAPI, Uvicorn, Swagger UI |
| **공간정보** | rasterio, pyproj, GDAL |
| **데이터** | pandas, numpy, OpenCV |

---

## 🗂️ 프로젝트 구조

```
pinetree_scan/
├── 📊 notebooks/
│   └── YOLOv11s_Smart_Training.ipynb    # Google Colab 학습 노트북
├── 🚀 api/                              # FastAPI 서버
│   ├── main.py                          # API 메인 애플리케이션
│   ├── config.py                        # API 설정 (conf=0.16, iou=0.45)
│   └── routers/                         # API 엔드포인트
│       ├── inference.py                 # 피해목 탐지 API
│       ├── preprocessing.py             # 멀티스케일 데이터 전처리 API
│       ├── visualization.py             # 결과 시각화 API
│       └── model_performance.py         # 모델 성능 평가 API
├── 🤖 models/                           # 학습된 모델
│   └── colab_yolo/best.pt              # YOLOv11s 탐지 모델
├── 📁 data/
│   ├── api_outputs/                     # API 결과 저장소
│   └── training_data/                   # 학습 데이터
└── 📈 results/                          # 학습 결과

## 🏗️ 시스템 아키텍처

```
🌲 Pinetree Scan Architecture

📡 데이터 수집
├── 🛰️ GeoTIFF (항공이미지)
├── 📍 GPS 좌표 (CSV)  
└── 🗺️ 지리참조 (TFW)
      ↓
🧠 멀티스케일 전처리
├── GPS 밀도 분석 (50px 반경)
├── 적응적 바운딩박스 (16~128px)
└── 타일 분할 + 라벨 생성
      ↓  
🤖 YOLOv11s 학습
├── Google Colab (무료 GPU)
├── 다중 지역 데이터셋 병합
└── 실시간 성능 모니터링
      ↓
🚀 FastAPI 추론 서버
├── 배치 탐지 (conf=0.16, iou=0.45)
├── GPS 좌표 변환
└── 시각화 결과 생성
```

---

## ⚡ 빠른 시작

### 1️⃣ Google Colab에서 학습
```python
# Google Colab에서 실행
!git clone https://github.com/your-repo/pinetree_scan.git
%cd pinetree_scan
!pip install ultralytics

# 노트북 실행
# notebooks/YOLOv11s_Smart_Training.ipynb
```

### 2️⃣ 로컬에서 API 서버 실행
```bash
# 의존성 설치
poetry install

# API 서버 시작  
poetry run uvicorn api.main:app --reload

# 브라우저에서 접속
# http://localhost:8000/docs (Swagger UI)
# http://localhost:8000/redoc (ReDoc)
```
```

---


## 🔥 Google Colab으로 학습

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jeedoli/pinetree_scan/blob/main/notebooks/YOLOv11s_Smart_Training.ipynb)

```python
# 1. 위 링크로 Colab 노트북 열기
# 2. 런타임 → GPU 설정
# 3. 1단계부터 4단계까지 순차 실행
# 4. 자동으로 Google Drive에 모델 백업
```

**Colab 특징:**
- 🆓 **무료 GPU**: T4/V100 GPU 무료 사용
- 🤖 **24px 최적화**: 개별 나무 탐지 특화
- 📊 **실시간 모니터링**: 학습 진행 상황 실시간 확인
- ☁️ **자동 백업**: Google Drive 자동 연동

### 🌐 API 서버 실행

```bash
# 1. 저장소 클론 및 의존성 설치
git clone https://github.com/Jeedoli/pinetree_scan.git
cd pinetree_scan
poetry install

# 2. API 서버 실행
poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 3. 브라우저에서 API 문서 확인
# http://localhost:8000/docs (Swagger UI)
# http://localhost:8000/redoc (ReDoc)
```

---

## 📊 학습 성능

| 모델 | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | 특징 |
|------|---------|--------------|-----------|--------|------|
| **YOLOv11s-멀티스케일** | **0.7+** | **0.6+** | **0.75+** | **0.72+** | 적응적 바운딩박스 |
| YOLOv11s-고정크기 | 0.5 | 0.45 | 0.60 | 0.55 | 32px 고정 (이전) |
| YOLOv8s-64px | 0.45 | 0.40 | 0.55 | 0.50 | 기존 버전 |

**🎯 멀티스케일 적응적 바운딩박스 시스템:**
- **GPS 밀도 분석**: 50px 반경 내 GPS 포인트 개수로 밀도 계산
- **적응적 크기 결정**:
  - 🌲 **외딴 피해목** (밀도 ≤1): 128px 큰 바운딩박스
  - � **낮은 밀도** (밀도 ≤3): 96px 중간 바운딩박스  
  - 🌿 **중간 밀도** (밀도 ≤6): 64px 표준 바운딩박스
  - 🍃 **높은 밀도** (밀도 ≤10): 48px 작은 바운딩박스
  - 🌱 **밀집 지역** (밀도 >10): 16px 최소 바운딩박스

**📈 성능 개선 효과:**
- 🎯 **상황별 최적화**: GPS 밀도에 따른 바운딩박스 크기 자동 조절
- � **캐시 버그 수정**: 각 지도별 독립적 GPS 좌표 처리
- 📊 **데이터 품질 향상**: 1,155개 타일 (6개 지역 통합)
- 🎛️ **최적화된 설정**: conf=0.16, iou=0.45로 민감한 탐지

---

## 🌐 FastAPI REST API

### 🎯 주요 엔드포인트

#### **🔍 피해목 탐지 API**
```http
POST /api/v1/inference/detect
Content-Type: multipart/form-data

# 업로드: 이미지 파일들 (최대 2GB)
# 결과: GPS 좌표가 포함된 CSV 파일
```

#### **⚙️ 멀티스케일 데이터 전처리 API**  
```http
POST /api/v1/preprocessing/create-dataset
Content-Type: multipart/form-data

# 업로드: GeoTIFF + TFW + CSV
# 결과: 멀티스케일 적응적 바운딩박스 학습용 데이터셋 (ZIP)
# 특징: GPS 밀도 기반으로 16~128px 크기 자동 조절
```

#### **🔗 다중 데이터셋 병합 API**
```http
POST /api/v1/preprocessing/merge-datasets  
Content-Type: multipart/form-data

# 업로드: 여러 지역의 데이터셋 ZIP 파일들
# 결과: 통합된 대용량 학습 데이터셋
# 예시: 6개 지역 → 1,155개 타일 통합
```

#### **📊 결과 시각화 API**
```http
POST /api/v1/visualization/auto-detect-and-visualize
Content-Type: multipart/form-data

# 업로드: 이미지 파일
# 결과: 자동 탐지 + 바운딩박스 시각화 이미지
# 설정: conf=0.16, iou=0.45 (촘촘한 피해목 개별 탐지)
```

### 📋 API 사용 예시

```python
import requests

# 🎯 피해목 탐지 API 호출 (최적화된 설정)
url = "http://localhost:8000/api/v1/inference/detect"
files = {"images_zip": open("forest_tiles.zip", "rb")}
data = {
    "confidence": 0.16,      # 민감한 탐지
    "iou_threshold": 0.45,   # 촘촘한 피해목 개별 탐지
    "save_visualization": True
}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"탐지된 피해목: {result['total_detections']}개")

# 🧠 멀티스케일 데이터셋 생성 API 호출
url = "http://localhost:8000/api/v1/preprocessing/create-dataset" 
files = {
    "geotiff_file": open("forest_map.tif", "rb"),
    "tfw_file": open("forest_map.tfw", "rb"),
    "csv_file": open("gps_coordinates.csv", "rb")
}
data = {
    "tile_size": 512,        # 타일 크기
    "file_prefix": "region1" # 파일 접두사
}

response = requests.post(url, files=files, data=data)
print("멀티스케일 적응적 데이터셋 생성 완료!")
print(f"결과 파일: {result['csv_file']}")
```

### 🏠 API 접속 주소
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc  
- **헬스체크**: http://localhost:8000/health

---

## 🎓 Google Colab 학습 가이드

### 📋 4단계 학습 프로세스

| 단계 | 내용 | 소요시간 |
|------|------|----------|
| **1단계** | 환경 설정 + ZIP 데이터 자동 탐지 | 5분 |
| **1.5단계** | 데이터 전처리 + 24px 최적화 | 10분 |
| **2단계** | YOLOv11s 모델 학습 | 1-2시간 |
| **3-4단계** | 성능 평가 + 결과 분석 | 10분 |

### 🔧 하이퍼파라미터 최적화

**성능별 권장 설정:**

```python
# 기본 성능 (mAP 0.6-0.8)
training_config = {
    'epochs': 200,
    'lr0': 0.0001,
    'batch': 8,
    'patience': 75
}

# 고급 성능 (mAP 0.8-0.9+)  
training_config = {
    'epochs': 300,
    'lr0': 0.00005,
    'batch': 12,
    'patience': 100,
    'weight_decay': 0.0005
}
```

### 📈 학습 모니터링

Colab에서 실시간으로 확인 가능한 지표:
- **Loss 곡선**: box_loss, cls_loss, dfl_loss
- **성능 지표**: mAP@0.5, Precision, Recall  
- **학습 진행**: GPU 메모리 사용량, 남은 시간

---

## 📊 결과 예시

### 🎯 탐지 결과 CSV
```csv
filename,x_center,y_center,confidence,longitude,latitude
forest_01.jpg,1245.2,678.9,0.92,127.123456,37.654321
forest_01.jpg,2145.8,1023.1,0.87,127.134567,37.645432
```

### 📊 추론 시각화 결과 및 conf 임계값 최적화 부분

<img width="512" alt="추론 시각화 예시" src="https://github.com/user-attachments/assets/3370d830-3b66-4d91-8965-6897b85fa86a" />
<img width="512" alt="딥러닝 신뢰도 임계값" src="https://github.com/user-attachments/assets/3aeb8c9f-5da6-4f6d-81a2-9a2a409fd3a2" />

---

## 💻 시스템 요구사항

### 🌐 Google Colab (권장)
- **무료 GPU**: T4/V100 사용 가능
- **메모리**: 12GB RAM 기본 제공
- **저장공간**: Google Drive 연동 (15GB 무료)
- **브라우저**: Chrome, Firefox, Safari 등

### 🏠 로컬 환경 (API 서버)
- **Python**: 3.10+ 
- **메모리**: 8GB+ (대용량 파일 처리용)
- **저장공간**: 10GB+ 여유 공간
- **GPU**: CUDA 호환 GPU (선택사항)

---

## 💡 FAQ

### ❓ **멀티스케일 적응적 바운딩박스란?**
- **GPS 밀도 분석**: 각 피해목 주변 50px 반경 내 GPS 포인트 개수 계산
- **적응적 크기**: 밀도가 높으면 16px (밀집), 낮으면 128px (외딴)
- **성능 향상**: 고정 크기(32px) 대비 mAP 0.5 → 0.7+ 개선

### ❓ **API 추론 설정 최적화는?**
```python
# 현재 최적화된 설정
confidence = 0.16     # 민감한 탐지 (약한 신호도 포착)
iou_threshold = 0.45  # 촘촘한 피해목도 개별 탐지
```

### ❓ **데이터셋은 어떻게 준비하나요?**
```
1. 멀티스케일 데이터셋 생성 (각 지역별)
   └── GPS 밀도 기반 적응적 바운딩박스 자동 생성
   
2. 다중 데이터셋 병합 
   └── 여러 지역 데이터를 하나로 통합
   
3. 최종 결과: 1,000개+ 타일 (권장)
```

### ❓ **성능이 낮으면 어떻게 하나요?**
1. **캐시 버그 확인**: 각 지도별 독립적 GPS 처리 여부
2. **더 많은 지역**: 다양한 지역 데이터셋 병합 (목표: 1,000개+ 타일)
3. **설정 조정**: conf=0.16, iou=0.45로 최적화
4. **멀티스케일 활용**: GPS 밀도 기반 적응적 바운딩박스

### ❓ **캐시 버그가 뭔가요?**
- **문제**: 첫 번째 지도의 GPS 좌표가 모든 후속 지도에 재사용
- **증상**: 모든 지도에서 동일한 라벨 개수 (170~171개)
- **해결**: 각 지도 처리 시 픽셀 좌표 캐시 초기화
- **결과**: 지도별 독립적 GPS 처리, 올바른 라벨 개수

### ❓ **결과 파일은 어디에 저장되나요?**
- **Colab**: Google Drive `/pinetree_scan/results/`
- **API**: `data/api_outputs/` 폴더
  - `inference/`: 탐지 결과 CSV, 시각화 이미지
  - `tiles/`: 멀티스케일 데이터셋 ZIP
  - `visualizations/`: 바운딩박스 표시된 이미지

---

## 📚 참고 자료

- [YOLOv11 공식 문서](https://docs.ultralytics.com/)
- [Google Colab 사용법](https://colab.research.google.com/)
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [rasterio 공식 문서](https://rasterio.readthedocs.io/)

---
