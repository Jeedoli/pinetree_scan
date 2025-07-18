
import os
import pandas as pd
import rasterio
from ultralytics import YOLO

IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'training_images')
RESULT_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'results', 'damaged_trees_gps.csv')
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'yolo_pinetree.pt')  # 사용자 맞춤 모델 경로

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

def detect_and_save():
    model = YOLO(MODEL_PATH)
    results = []
    for fname in os.listdir(IMG_DIR):
        if not fname.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(IMG_DIR, fname)
        yolo_result = model(img_path)
        for r in yolo_result:
            for box in r.boxes:
                # 피해목 클래스만 추출 (예: class_id==1)
                if int(box.cls[0]) == 1:
                    xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                    x_center = (xmin + xmax) / 2
                    y_center = (ymin + ymax) / 2
                    lat, lon = pixel_to_gps(img_path, x_center, y_center)
                    if lat is not None and lon is not None:
                        results.append({'filename': fname, 'latitude': lat, 'longitude': lon})
    if results:
        out_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)
        out_df.to_csv(RESULT_CSV, index=False)
        print(f'피해목 GPS 좌표가 {RESULT_CSV}에 저장되었습니다.')
    else:
        print('피해목이 없습니다.')

if __name__ == '__main__':
    detect_and_save()
