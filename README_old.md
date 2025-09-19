# 🌲 Pinetree Scan: 소나무 피해목 AI 탐지 시스템

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/YOLOv11s-ultralytics-yellowgreen?logo=github" />
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white" />
  <img src="https://img.shields.io/badge/rasterio-%23007396.svg?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/mAP-0.9+-success" />
</p>

> **드론/항공 이미지에서 YOLOv11s 딥러닝 모델로 소나무 피해목을 자동 탐지하고, GPS 좌표로 변환하여 저장하는 AI 기반 공간정보 시스템**

---

## 🚀 주요 특징

- 🎯 **YOLOv11s 기반 고정밀 탐지**: 24px 최적화로 개별 나무 탐지 성능 극대화
- 🌐 **Google Colab 완벽 지원**: 클라우드 GPU로 무료 학습 환경 제공
- 🔥 **FastAPI REST API**: 대용량 파일(2GB) 지원, Swagger UI 자동 문서화
- 📊 **실시간 학습 모니터링**: 학습 곡선, 성능 지표 실시간 추적
- 🗺️ **정밀 GPS 변환**: GeoTIFF 공간정보 기반 위경도 좌표 자동 변환
- 📈 **실용적 성능**: mAP 0.9+ 달성으로 상용 수준 정확도

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
│   └── routers/                         # API 엔드포인트
│       ├── inference.py                 # 피해목 탐지 API
│       ├── preprocessing.py             # 데이터 전처리 API
│       └── visualization.py             # 결과 시각화 API
├── 🤖 models/                           # 학습된 모델
│   └── colab_yolo/best.pt              # YOLOv11s 탐지 모델
├── 📁 data/
│   ├── api_outputs/                     # API 결과 저장소
│   └── training_data/                   # 학습 데이터
└── 📈 results/                          # 학습 결과
```

---

## 🎯 빠른 시작

### 🔥 Google Colab으로 학습하기 (권장)

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
| **YOLOv11s-24px** | **0.95+** | **0.85+** | **0.92+** | **0.89+** | 24px 최적화 |
| YOLOv8s-64px | 0.90 | 0.75 | 0.85 | 0.82 | 기존 버전 |

**📈 성능 개선 효과:**
- 🎯 **개별 나무 탐지**: 64px → 24px로 바운딩박스 최적화
- 🔍 **오탐지 감소**: 한 박스당 하나의 나무로 정밀도 향상  
- 📍 **정확한 GPS**: 더 정밀한 위치 좌표 제공

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

#### **⚙️ 데이터 전처리 API**  
```http
POST /api/v1/preprocessing/tile_and_label
Content-Type: multipart/form-data

# 업로드: GeoTIFF + TFW + CSV
# 결과: 학습용 타일 데이터셋 (ZIP)
```

#### **📊 결과 시각화 API**
```http
POST /api/v1/visualization/mark_boxes
Content-Type: multipart/form-data

# 업로드: 이미지 + 탐지 결과 CSV
# 결과: 바운딩박스가 표시된 이미지
```

### 📋 API 사용 예시

```python
import requests

# 피해목 탐지 API 호출
url = "http://localhost:8000/api/v1/inference/detect"
files = {"files": open("forest_image.jpg", "rb")}
data = {"confidence": 0.3, "save_csv": True}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"탐지된 피해목: {result['total_detections']}개")
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

### 📊 시각화 결과
탐지된 피해목에 바운딩박스가 표시된 이미지가 자동 생성됩니다.

<img width="512" alt="탐지 결과 예시" src="https://github.com/user-attachments/assets/1507422b-e1ad-4811-ab2f-41350473894d" />

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

### ❓ **Google Colab vs 로컬 환경?**
- **Colab**: 무료 GPU, 환경 설정 불필요, 학습에 최적화
- **로컬**: API 서버, 대용량 추론, 커스텀 파이프라인

### ❓ **학습 데이터는 어떻게 준비하나요?**
```
training_data/
├── images/          # 이미지 파일들
├── labels/          # YOLO 라벨 파일들
└── data.yaml        # 데이터셋 설정
```

### ❓ **성능이 낮으면 어떻게 하나요?**
1. **더 많은 데이터** 추가
2. **에포크 수 증가** (200 → 300)
3. **학습률 조정** (lr0: 0.0001 → 0.00005)
4. **24px 최적화** 적용

### ❓ **결과 파일은 어디에 저장되나요?**
- **Colab**: Google Drive `/pinetree_scan/results/`
- **API**: `data/api_outputs/` 폴더

---

## 📚 참고 자료

- [YOLOv11 공식 문서](https://docs.ultralytics.com/)
- [Google Colab 사용법](https://colab.research.google.com/)
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [rasterio 공식 문서](https://rasterio.readthedocs.io/)

---

## 📞 지원 및 기여

- **Issues**: [GitHub Issues](https://github.com/Jeedoli/pinetree_scan/issues)
- **License**: MIT License
- **기여**: Pull Request 환영

---

> **🌲 지속 가능한 산림 관리를 위한 AI 기술로 더 나은 미래를 만들어가세요!**
