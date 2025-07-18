import os
import pandas as pd
import numpy as np

def tm_to_pixel(x, y, tfw):
    # tfw: [A, D, B, E, C, F] (GDAL)
    A, D, B, E, C, F = tfw
    px = (x - C) / A
    py = (y - F) / E
    return px, py

def make_yolo_label(csv_path, tfw_path, out_txt, img_width, img_height, bbox_size=16, class_id=0):
    df = pd.read_csv(csv_path)
    with open(tfw_path, 'r') as f:
        tfw = [float(line.strip()) for line in f.readlines()]
    lines = []
    for _, row in df.iterrows():
        x_tm, y_tm = row['x'], row['y']
        px, py = tm_to_pixel(x_tm, y_tm, tfw)
        # YOLO: x_center, y_center, w, h (normalized)
        x_center = px / img_width
        y_center = py / img_height
        w = bbox_size / img_width
        h = bbox_size / img_height
        # 이미지 경계 체크
        if 0 <= px < img_width and 0 <= py < img_height:
            lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    with open(out_txt, 'w') as f:
        f.write('\n'.join(lines))
    print(f"YOLO 라벨 {out_txt} 생성 완료. 객체 수: {len(lines)}")

if __name__ == '__main__':
    # 경로 및 파라미터
    csv_path = 'data/training_images/damaged_tree.csv'
    tfw_path = 'data/training_images/sample01.tfw'
    out_txt = 'data/train/labels/sample01.txt'
    img_width = 18737
    img_height = 30323
    bbox_size = 16  # 2m
    class_id = 0    # 피해목
    make_yolo_label(csv_path, tfw_path, out_txt, img_width, img_height, bbox_size, class_id)
