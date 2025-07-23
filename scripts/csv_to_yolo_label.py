import os

import numpy as np
import pandas as pd


def tm_to_pixel(x, y, tfw):
    # tfw: [A, D, B, E, C, F] (GDAL)
    A, D, B, E, C, F = tfw
    px = (x - C) / A
    py = (y - F) / E
    return px, py


def make_yolo_label(
    csv_path, tfw_path, out_txt, img_width, img_height, bbox_size=16, class_id=0
):
    df = pd.read_csv(csv_path)
    with open(tfw_path, "r") as f:
        tfw = [float(line.strip()) for line in f.readlines()]
    lines = []
    for _, row in df.iterrows():
        x_tm, y_tm = row["x"], row["y"]
        px, py = tm_to_pixel(x_tm, y_tm, tfw)
        # YOLO: x_center, y_center, w, h (normalized)
        x_center = px / img_width
        y_center = py / img_height
        w = bbox_size / img_width
        h = bbox_size / img_height
        # 이미지 경계 체크
        if 0 <= px < img_width and 0 <= py < img_height:
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    with open(out_txt, "w") as f:
        f.write("\n".join(lines))
    print(f"YOLO 라벨 {out_txt} 생성 완료. 객체 수: {len(lines)}")


if __name__ == "__main__":
    # 경로 및 파라미터
    csv_path = "data/training_images/damaged_tree.csv"
    tfw_path = "data/training_images/sample01.tfw"
    out_txt = "data/train/labels/sample01.txt"

# ---------------------------------------------
# CSV 피해목 좌표 → YOLO 라벨(txt) 변환 스크립트
# ---------------------------------------------

import os

import numpy as np
import pandas as pd


def tm_to_pixel(x, y, tfw):
    """
    TM 좌표(x, y)를 이미지 픽셀 좌표(px, py)로 변환
    - tfw: [A, D, B, E, C, F] (GDAL)
    """
    A, D, B, E, C, F = tfw
    px = (x - C) / A
    py = (y - F) / E
    return px, py


def make_yolo_label(
    csv_path, tfw_path, out_txt, img_width, img_height, bbox_size=16, class_id=0
):
    """
    CSV 피해목 좌표와 TFW 파일을 이용해 YOLO 라벨(txt) 파일 생성
    - csv_path: 피해목 위치 CSV
    - tfw_path: 좌표 변환 파일
    - out_txt: 출력 YOLO 라벨 파일 경로
    - img_width, img_height: 전체 이미지 크기
    - bbox_size: 바운딩박스 크기(픽셀)
    - class_id: YOLO 클래스 ID
    """
    df = pd.read_csv(csv_path)
    with open(tfw_path, "r") as f:
        tfw = [float(line.strip()) for line in f.readlines()]
    lines = []
    for _, row in df.iterrows():
        x_tm, y_tm = row["x"], row["y"]
        px, py = tm_to_pixel(x_tm, y_tm, tfw)
        # YOLO: x_center, y_center, w, h (정규화)
        x_center = px / img_width
        y_center = py / img_height
        w = bbox_size / img_width
        h = bbox_size / img_height
        # 이미지 경계 내에 있는 경우만 라벨로 변환
        if 0 <= px < img_width and 0 <= py < img_height:
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    with open(out_txt, "w") as f:
        f.write("\n".join(lines))
    print(f"YOLO 라벨 {out_txt} 생성 완료. 객체 수: {len(lines)}")


if __name__ == "__main__":
    # 예시 경로 및 파라미터 (실제 경로/크기로 수정 필요)
    csv_path = "data/training_images/damaged_tree.csv"
    tfw_path = "data/training_images/sample01.tfw"
    out_txt = "data/train/labels/sample01.txt"
    img_width = 18737
    img_height = 30323
    bbox_size = 16  # 2m
    class_id = 0  # 피해목
    make_yolo_label(
        csv_path, tfw_path, out_txt, img_width, img_height, bbox_size, class_id
    )
