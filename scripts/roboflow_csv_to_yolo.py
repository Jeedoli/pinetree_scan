# ---------------------------------------------
# 라벨 CSV → YOLO 포맷(txt) 변환 스크립트
# ---------------------------------------------

# ---------------------------------------------
# Roboflow 라벨 CSV → YOLO 포맷(txt) 변환 스크립트
# ---------------------------------------------

import glob
import os

import cv2


def roboflow_csv_to_yolo(img_dir, label_csv, out_dir):
    """
    Roboflow에서 export한 라벨 CSV 파일을 YOLO 포맷(txt)으로 변환
    - img_dir: 이미지 폴더 경로
    - label_csv: Roboflow 라벨 CSV 파일 경로 (filename, label, xmin, ymin, xmax, ymax)
    - out_dir: 변환된 YOLO 라벨 저장 폴더
    """
    import pandas as pd

    df = pd.read_csv(label_csv)
    os.makedirs(out_dir, exist_ok=True)
    for fname in df["filename"].unique():
        img_path = os.path.join(img_dir, fname)
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        rows = df[df["filename"] == fname]
        lines = []
        for _, row in rows.iterrows():
            class_id = int(row["label"])  # 클래스 ID
            xmin, ymin, xmax, ymax = map(
                float, [row["xmin"], row["ymin"], row["xmax"], row["ymax"]]
            )
            # YOLO 포맷: 중심좌표/너비/높이 (정규화)
            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            box_w = (xmax - xmin) / w
            box_h = (ymax - ymin) / h
            lines.append(f"{class_id} {x_center} {y_center} {box_w} {box_h}")
        # txt 파일로 저장 (이미지명과 동일하게)
        with open(
            os.path.join(
                out_dir, fname.replace(".tif", ".txt").replace(".jpg", ".txt")
            ),
            "w",
        ) as f:
            f.write("\n".join(lines))
    print(f"YOLO 라벨 변환 완료: {out_dir}")


if __name__ == "__main__":

    # 예시 경로: 이미지 폴더, Roboflow 라벨 CSV, 출력 폴더
    roboflow_csv_to_yolo(
        "../data/training_images", "../data/train_labels.csv", "../data/labels"
    )
