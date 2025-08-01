# -----------------------------
# 피해목/정상목 YOLO 추론 결과 스크립트
# -----------------------------
# 사용법 예시:
# poetry run python scripts/yolo_infer.py --weights models/colab_yolo/best.pt --source data/inference_images --output data/infer_results/damaged_trees_YYYYMMDD_HHMMSS.csv
#
# --weights: YOLO 모델 가중치 경로
# --source: 추론할 이미지 폴더
# --output: 결과 CSV 파일 경로(생략 시 자동 생성)
# -----------------------------

import argparse
import os

import pandas as pd
import rasterio
from ultralytics import YOLO


def get_args():
    """
    커맨드라인 인자 파싱 함수
    --weights: YOLO 모델 가중치 경로
    --source: 추론할 이미지 폴더
    --output: 결과 CSV 파일 경로(기본값: 오늘 날짜/시간 포함)
    """
    import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="models/colab_yolo/best.pt",
        help="YOLO 모델 가중치 경로 (예: models/colab_yolo/best.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/inference_images",
        help="인퍼런스(추론)용 이미지 폴더 (예: data/inference_images)",
    )
    # 오늘 날짜와 시간 포함한 기본 파일명
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 프로젝트 루트 기준으로 절대경로 생성
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_csv = os.path.join(
        project_root, "data/infer_results", f"sample01_{now}.csv"
    )
    parser.add_argument("--output", type=str, default=default_csv, help="결과 CSV 경로")
    return parser.parse_args()


def infer_and_save(weights, source, output):
    """
    YOLO 모델로 이미지 폴더 내 모든 이미지를 추론하고,
    탐지된 박스의 정보를 CSV로 저장
    - weights: YOLO 모델 가중치 경로
    - source: 추론할 이미지 폴더
    - output: 결과 CSV 파일 경로
    """
    model = YOLO(weights)  # YOLO 모델 로드
    results = []  # 결과 저장 리스트
    class_names = ["damaged", "healthy"]  # 클래스명 리스트
    damaged_count = 0  # 피해목 탐지 개수
    import cv2

    # source가 파일이면 리스트로, 디렉토리면 내부 이미지 리스트로
    if os.path.isfile(source):
        img_files = [source]
    else:
        img_files = [
            os.path.join(source, fname)
            for fname in os.listdir(source)
            if fname.lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png"))
        ]

    for img_path in img_files:
        fname = os.path.basename(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is not None and len(img.shape) == 3 and img.shape[2] == 4:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            cv2.imwrite(img_path, img_rgb)
        yolo_result = model(img_path, conf=0.05, iou=0.25)
        for r in yolo_result:
            for box in r.boxes:
                class_id = int(box.cls[0])
                if class_id == 0:
                    xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                    x_center = (xmin + xmax) / 2 / img.shape[1]
                    y_center = (ymin + ymax) / 2 / img.shape[0]
                    width = (xmax - xmin) / img.shape[1]
                    height = (ymax - ymin) / img.shape[0]
                    results.append(
                        {
                            "filename": fname,
                            "class_id": class_id,
                            "center_x": x_center,
                            "center_y": y_center,
                            "width": width,
                            "height": height,
                        }
                    )
                    damaged_count += 1
    # 결과가 있으면 CSV로 저장
    if results:
        out_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        out_df.to_csv(output, index=False)
        print(f"피해목 탐지: {damaged_count}건")
        print(f"결과가 {output}에 저장되었습니다.")
    else:
        print("피해목이 탐지되지 않았습니다.")


if __name__ == "__main__":
    # 커맨드라인 인자 받아서 추론 및 결과 저장 실행
    args = get_args()
    infer_and_save(args.weights, args.source, args.output)
