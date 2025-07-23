from ultralytics import YOLO

# 데이터셋 구성 파일 경로
DATA_YAML = "../data/data.yaml"
MODEL_OUT = "../models/yolo_pinetree.pt"

# 사전학습 YOLOv8 모델로 피해목 탐지 학습
model = YOLO("yolov8n.pt")
model.train(data=DATA_YAML, epochs=50, imgsz=640, batch=8)

import glob
# 학습된 모델(best.pt)을 프로젝트 모델 폴더로 복사
import shutil

best_model = glob.glob("runs/detect/train/weights/best.pt")
if best_model:
    shutil.copy(best_model[0], MODEL_OUT)
    print(f"학습된 모델이 {MODEL_OUT}에 저장되었습니다.")
else:
    print("학습된 모델(best.pt)을 찾을 수 없습니다.")
