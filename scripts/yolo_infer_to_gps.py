import os
import pandas as pd
import rasterio
from ultralytics import YOLO

import argparse

def get_args():
    import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='../models/yolo_pinetree.pt', help='YOLO 모델 가중치 경로')
    parser.add_argument('--source', type=str, default='../data/inference_images', help='인퍼런스(추론)용 이미지 폴더')
    # 오늘 날짜와 시간 포함한 기본 파일명
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    default_csv = f'../data/results/damaged_trees_gps_{now}.csv'
    parser.add_argument('--output', type=str, default=default_csv, help='결과 CSV 경로')
    return parser.parse_args()

# 바운딩 박스 중심 픽셀 좌표를 위도/경도로 변환
def pixel_to_gps(image_path, x, y):
    with rasterio.open(image_path) as src:
        xp, yp = src.transform * (x, y)
        if src.crs and src.crs.to_string().startswith('EPSG:4326'):
            return yp, xp
        else:
            try:
                import pyproj
                proj = pyproj.Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
                lon, lat = proj.transform(xp, yp)
                return lat, lon
            except Exception as e:
                print(f'좌표 변환 오류: {e}')
                return None, None

def infer_and_save(weights, source, output):
    model = YOLO(weights)
    results = []
    class_names = ['damaged', 'healthy']
    damaged_count = 0
    healthy_count = 0
    for fname in os.listdir(source):
        if not fname.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(source, fname)
        # confidence threshold를 낮춰서 박스가 아예 없는지 확인
        yolo_result = model(img_path, conf=0.01)
        for r in yolo_result:
            for box in r.boxes:
                class_id = int(box.cls[0])
                if class_id in [0, 1]:
                    xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                    x_center = (xmin + xmax) / 2
                    y_center = (ymin + ymax) / 2
                    lat, lon = pixel_to_gps(img_path, x_center, y_center)
                    if lat is not None and lon is not None:
                        results.append({
                            'filename': fname,
                            'latitude': lat,
                            'longitude': lon,
                            'class_id': class_id,
                            'class_name': class_names[class_id]
                        })
                        if class_id == 0:
                            damaged_count += 1
                        elif class_id == 1:
                            healthy_count += 1
    if results:
        out_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        out_df.to_csv(output, index=False)
        print(f'피해목 탐지: {damaged_count}건, 정상목 탐지: {healthy_count}건')
        print(f'결과가 {output}에 저장되었습니다.')
        if damaged_count == 0:
            print('※ 피해목(0) 박스가 하나도 탐지되지 않았습니다.')
        if healthy_count == 0:
            print('※ 정상목(1) 박스가 하나도 탐지되지 않았습니다.')
    else:
        print('피해목/정상목 모두 탐지되지 않았습니다.')

if __name__ == '__main__':
    args = get_args()
    infer_and_save(args.weights, args.source, args.output)
