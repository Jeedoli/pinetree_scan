import os
import cv2
import glob

def convert_to_yolo_format(img_dir, label_csv, out_dir):
    import pandas as pd
    df = pd.read_csv(label_csv)
    os.makedirs(out_dir, exist_ok=True)
    for fname in df['filename'].unique():
        img_path = os.path.join(img_dir, fname)
        if not os.path.exists(img_path):
            continue
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        rows = df[df['filename'] == fname]
        lines = []
        for _, row in rows.iterrows():
            class_id = int(row['label'])
            xmin, ymin, xmax, ymax = map(float, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            box_w = (xmax - xmin) / w
            box_h = (ymax - ymin) / h
            lines.append(f"{class_id} {x_center} {y_center} {box_w} {box_h}")
        with open(os.path.join(out_dir, fname.replace('.tif', '.txt').replace('.jpg', '.txt')), 'w') as f:
            f.write('\n'.join(lines))
    print(f'YOLO 라벨 변환 완료: {out_dir}')

if __name__ == '__main__':
    # 예시 경로: 이미지 폴더, 라벨 CSV, 출력 폴더
    convert_to_yolo_format('../data/training_images', '../data/train_labels.csv', '../data/labels')
