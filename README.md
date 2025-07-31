# 🌲 Pinetree Scan: 소나무재선충병 피해목 자동 탐지 파이프라인

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/YOLOv8-ultralytics-yellowgreen?logo=github" />
  <img src="https://img.shields.io/badge/rasterio-%23007396.svg?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/OS-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey" />
</p>

> **드론/항공 GeoTIFF 이미지에서 소나무재선충병 피해목을 딥러닝(YOLO)으로 자동 탐지하고, 피해목 위치를 위경도(GPS)로 변환·CSV로 저장하는 공간정보 기반 자동화 파이프라인**

---

## 🧰 기술스택

| 구분 | 내용 |
|------|------|
| 언어 | Python 3.10+ |
| 딥러닝 | Ultralytics YOLOv8 |
| 공간정보 | rasterio, affine |
| 데이터 | pandas, numpy |
| 기타 | pyproj(좌표계 변환), argparse 등 |

---

---

## 🚀 프로젝트 한눈에 보기

- **목표**: 공간정보(GeoTIFF, tfw, prj)와 피해목 TM좌표(csv)만으로, 소나무재선충병 피해목을 YOLO로 자동 탐지 및 위치(GPS) 변환
- **주요 기능**:
  - 대용량 GeoTIFF → 타일 분할(자동)
  - 피해목 TM좌표 → 픽셀 변환 → YOLO 라벨 자동 생성
  - YOLO 학습/추론/CSV 결과 자동화
  - 탐지 결과를 위경도(GPS)로 변환해 CSV로 저장
- **기술스택**: Python, rasterio, pandas, numpy, Ultralytics YOLOv8

---

## 🗂️ 폴더 구조 (예시)
```
data/
  training_images/   # 원본 GeoTIFF, tfw, prj, sample01.csv(피해목 TM좌표)
  tiles/images/      # 타일 이미지 (자동 생성)
  tiles/labels/      # 타일별 YOLO 라벨(txt, 자동 생성)
  results/           # 추론 결과 CSV 등
scripts/
  tile_and_label.py  # 타일 분할+라벨 자동화
  yolo_infer_to_gps.py # YOLO 추론+좌표 변환+CSV 저장
  ...
```

---

## 🛠️ 주요 파이프라인

### 1. 타일 분할 및 라벨 자동 생성
- `scripts/tile_and_label.py` 실행
- 원본 tif를 타일(1024/4096px)로 분할, 각 타일별 YOLO 라벨(txt) 자동 생성
- **rasterio** 라이브러리 사용 (Pillow 아님)

### 2. YOLO 학습
- data.yaml에서 `train: tiles/images`, `val: tiles/images`로 지정
- `yolo detect train ...` 명령으로 학습
- 모델 파일(`best.pt`, `last.pt`)은 `results/프로젝트명/weights/`에 저장

### 3. 피해목 탐지 및 GPS 변환
- `scripts/yolo_infer_to_gps.py` 실행
- 학습된 모델(`best.pt`)로 타일/원본 이미지 추론
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
| results/프로젝트명/weights/best.pt | 학습된 YOLO 모델 |
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
yolo detect train data=data/data.yaml model=yolov8n.pt epochs=50 imgsz=640 project=results name=pinetree-damage-tiles4096
```

### 피해목 탐지 및 GPS 변환
```bash
python3 scripts/yolo_infer_to_gps.py --weights results/프로젝트명/weights/best.pt --source data/tiles/images
# 결과: data/results/sample01_gps_YYYYMMDD_HHMMSS.csv
```

---

## 📜 주요 스크립트 설명

### 1. `tile_and_label.py`
- **기능**: 대용량 GeoTIFF 이미지를 타일로 분할하고, TM 좌표를 YOLO 라벨로 변환
- **사용법**:
  ```bash
  poetry run python scripts/tile_and_label.py --input data/large_image.tif --output data/tiles --tfw data/large_image.tfw
  ```

### 2. `yolo_infer_to_gps.py`
- **기능**: YOLO 추론 결과를 CSV로 저장하며, YOLO 형식 좌표를 위경도(GPS)로 변환
- **사용법**:
  ```bash
  poetry run python scripts/yolo_infer_to_gps.py --weights models/best.pt --source data/tiles/images --output data/infer_results/results.csv
  ```

### 3. `mark_inference_boxes.py`
- **기능**: YOLO 추론 결과를 이미지에 시각화하여 저장
- **사용법**:
  ```bash
  poetry run python scripts/mark_inference_boxes.py --tiles_dir data/tiles/images --csv data/infer_results/results.csv
  ```
- **결과 예시**:
  아래는 추론 결과를 시각화한 이미지 예시입니다.
<img width="1024" height="1024" alt="sample01_8_7_marked" src="https://github.com/user-attachments/assets/1507422b-e1ad-4811-ab2f-41350473894d" />

  

---

## 💡 FAQ & 참고

- **Q. 결과 CSV는 어떻게 생성되나요?**
  - YOLO 추론+좌표 변환 스크립트 실행 시, 탐지 결과가 년월일시간이 포함된 새 CSV로 저장됩니다.
- **Q. 모델 파일(best.pt)은 어디에 저장되고, 어떻게 사용하나요?**
  - `results/프로젝트명/weights/best.pt`에 저장, 추론/좌표 변환/시각화에 사용
- **Q. 타일 분할은 어떤 라이브러리로 하나요?**
  - rasterio(공간정보 이미지 전용) 사용
- **Q. csv에 데이터가 중첩 저장되나요?**
  - 아니요. 항상 새로 생성(덮어쓰기)되며, 실행 시점의 결과만 포함

---

## 📚 참고자료
- [ultralytics YOLO 공식문서](https://docs.ultralytics.com/)
- [rasterio 공식문서](https://rasterio.readthedocs.io/)
