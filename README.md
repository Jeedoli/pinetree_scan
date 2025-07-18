
# Pinetree Scan - 소나무재선충병 피해목 자동 탐지 파이프라인

## 프로젝트 개요
- 드론/항공 촬영 GeoTIFF 이미지에서 소나무재선충병 피해목을 딥러닝(YOLO)으로 자동 탐지
- 피해목 위치를 위도·경도(GPS)로 변환하여 CSV로 저장
- 공간정보(tif+tfw+prj)와 피해목 중심좌표(csv)만으로 완전 자동화

---

## 폴더 구조
```
data/
  training_images/   # 원본 GeoTIFF, tfw, prj, damaged_tree.csv(피해목 TM좌표)
  tiles/images/      # 1024x1024 타일 이미지 (자동 생성)
  tiles/labels/      # 타일별 YOLO 라벨(txt, 자동 생성)
  results/           # 추론 결과 CSV 등
scripts/
  tile_and_label.py  # 타일 분할+라벨 자동화
  yolo_infer_to_gps.py # YOLO 추론+좌표 변환+CSV 저장
  ...
```

---

## 주요 파이프라인

### 1. 타일 분할 및 라벨 자동 생성
- `scripts/tile_and_label.py` 실행
- 원본 tif를 4096x4096 타일로 분할, 각 타일별 YOLO 라벨(txt) 자동 생성
- **rasterio** 라이브러리 사용 (Pillow 아님)

### 2. YOLO 학습
- data.yaml에서 `train: tiles/images`, `val: tiles/images`로 지정
- `yolo detect train ...` 명령으로 학습
- 모델 파일(`best.pt`, `last.pt`)은 `results/pinetree-damage-tiles4096/weights/`에 저장

### 3. 피해목 탐지 및 GPS 변환
- `scripts/yolo_infer_to_gps.py` 실행
- 학습된 모델(`best.pt`)로 타일/원본 이미지 추론
- 탐지된 피해목의 중심 픽셀좌표를 위경도(GPS)로 변환
- **결과 CSV는 실행 시점의 년월일시간이 포함된 파일명으로 자동 저장**
  - 예: `damaged_trees_gps_20250718_153012.csv`
- 기존 CSV를 덮어쓰지 않고, 실행할 때마다 새로운 결과 파일 생성

---

## 주요 파일 설명
| 파일/폴더 | 설명 |
|-----------|------|
| data/training_images/sample01.tif | 원본 GeoTIFF 이미지 |
| data/training_images/sample01.tfw | 월드파일(좌표계) |
| data/training_images/damaged_tree.csv | 피해목 중심 TM좌표(csv) |
| data/tiles/images/ | 타일 이미지(4096x4096, 자동 생성) |
| data/tiles/labels/ | 타일별 YOLO 라벨(txt, 자동 생성) |
| results/pinetree-damage-tiles4096/weights/best.pt | 학습된 YOLO 모델 |
| data/results/damaged_trees_gps_YYYYMMDD_HHMMSS.csv | 탐지 결과(GPS 좌표, 자동 생성) |

---

## 자주 묻는 질문(FAQ)

- **Q. damaged_trees_gps.csv는 어떻게 생성되나요?**
  - YOLO 추론+좌표 변환 스크립트 실행 시, 탐지 결과가 년월일시간이 포함된 새 CSV로 저장됩니다.
  - 기존 파일을 덮어쓰지 않고, 실행할 때마다 새로운 결과 파일이 생성됩니다.

- **Q. 모델 파일(best.pt)은 어디에 저장되고, 어떻게 사용하나요?**
  - `results/pinetree-damage-tiles4096/weights/best.pt`에 저장됩니다.
  - 추론, 좌표 변환, 시각화 등 모든 후처리에 바로 사용합니다.
  - 예시: `yolo detect predict model=.../best.pt source=...`

- **Q. 타일 분할은 어떤 라이브러리로 하나요?**
  - rasterio(공간정보 이미지 전용)를 사용합니다. Pillow는 사용하지 않습니다.

- **Q. csv에 데이터가 중첩 저장되나요?**
  - 아니요. 항상 새로 생성(덮어쓰기)되며, 실행 시점의 결과만 포함됩니다.

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
```
python3 scripts/tile_and_label.py
```

### YOLO 학습
```
yolo detect train data=data/data.yaml model=yolov8n.pt epochs=50 imgsz=640 project=results name=pinetree-damage-tiles4096
```

### 피해목 탐지 및 GPS 변환
```
python3 scripts/yolo_infer_to_gps.py --weights results/pinetree-damage-tiles4096/weights/best.pt --source data/tiles/images
# 결과: data/results/damaged_trees_gps_YYYYMMDD_HHMMSS.csv
```

---

## 문의/참고
- ultralytics YOLO 공식문서: https://docs.ultralytics.com/
- rasterio 공식문서: https://rasterio.readthedocs.io/
