# -----------------------------
# 피해목/정상목 YOLO 추론 및 결과 위경도 변환 CSV 저장 스크립트
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
        project_root, "data/infer_results", f"damaged_trees_gps_{now}.csv"
    )
    parser.add_argument("--output", type=str, default=default_csv, help="결과 CSV 경로")
    return parser.parse_args()


def pixel_to_gps(image_path, x, y):
    """
    바운딩 박스 중심 픽셀 좌표(x, y)를 위도/경도(GPS)로 변환
    - image_path: 이미지 파일 경로
    - x, y: 이미지 내 픽셀 좌표
    - return: (위도, 경도) 튜플
    """
    with rasterio.open(image_path) as src:
        xp, yp = src.transform * (x, y)
        # 이미지가 이미 WGS84라면 그대로 반환
        if src.crs and src.crs.to_string().startswith("EPSG:4326"):
            return yp, xp
        else:
            try:
                import pyproj

                proj = pyproj.Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
                lon, lat = proj.transform(xp, yp)
                return lat, lon
            except Exception as e:
                print(f"좌표 변환 오류: {e}")
                return None, None


def infer_and_save(weights, source, output):
    """
    YOLO 모델로 이미지 폴더 내 모든 이미지를 추론하고,
    탐지된 박스의 중심 픽셀 좌표를 위도/경도로 변환해 CSV로 저장
    - weights: YOLO 모델 가중치 경로
    - source: 추론할 이미지 폴더
    - output: 결과 CSV 파일 경로
    """
    model = YOLO(weights)  # YOLO 모델 로드
    results = []  # 결과 저장 리스트
    class_names = ["damaged", "healthy"]  # 클래스명 리스트
    damaged_count = 0  # 피해목 탐지 개수
    healthy_count = 0  # 정상목 탐지 개수
    import cv2

    for fname in os.listdir(source):
        # 이미지 파일만 처리
        if not fname.lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(source, fname)
        # 4채널 이미지는 3채널(RGB)로 변환 후 임시 저장
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is not None and len(img.shape) == 3 and img.shape[2] == 4:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            cv2.imwrite(img_path, img_rgb)
        # confidence threshold를 높이고, max_det 명시
        yolo_result = model(img_path, conf=0.5, max_det=1000)
        for r in yolo_result:
            for box in r.boxes:
                class_id = int(box.cls[0])  # 클래스 ID (0: 피해목, 1: 정상목)
                if class_id == 0:  # 피해목만 결과에 포함
                    xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                    x_center = (xmin + xmax) / 2
                    y_center = (ymin + ymax) / 2
                    # 중심 픽셀 좌표를 위경도로 변환
                    lat, lon = pixel_to_gps(img_path, x_center, y_center)
                    if lat is not None and lon is not None:
                        results.append(
                            {
                                "filename": fname,
                                "latitude": lat,
                                "longitude": lon,
                                "class_id": class_id,
                                "class_name": class_names[class_id],
                            }
                        )
                        damaged_count += 1
    # 결과가 있으면 CSV로 저장
    if results:
        out_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        out_df.to_csv(output, index=False)
        print(f"피해목 탐지: {damaged_count}건, 정상목 탐지: {healthy_count}건")
        print(f"결과가 {output}에 저장되었습니다.")
        if damaged_count == 0:
            print("※ 피해목(0) 박스가 하나도 탐지되지 않았습니다.")
        if healthy_count == 0:
            print("※ 정상목(1) 박스가 하나도 탐지되지 않았습니다.")
    else:
        print("피해목/정상목 모두 탐지되지 않았습니다.")


if __name__ == "__main__":
    # 커맨드라인 인자 받아서 추론 및 결과 저장 실행
    args = get_args()
    infer_and_save(args.weights, args.source, args.output)
