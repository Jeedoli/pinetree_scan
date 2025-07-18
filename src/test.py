import os

import pandas as pd
import rasterio

IMAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'images')
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'results')
RESULT_CSV = os.path.join(RESULT_DIR, 'pinetree_results.csv')


def get_geotiff_bounds(image_path):
    with rasterio.open(image_path) as src:
        bounds = src.bounds  # (left, bottom, right, top)
        crs = src.crs        # 좌표계 정보
        return bounds, crs

def main():
    results = []
    if not os.path.exists(IMAGE_DIR):
        print(f"이미지 폴더가 존재하지 않습니다: {IMAGE_DIR}")
        return
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    for fname in os.listdir(IMAGE_DIR):
        if fname.lower().endswith(('.tif', '.tiff')):
            img_path = os.path.join(IMAGE_DIR, fname)
            try:
                bounds, crs = get_geotiff_bounds(img_path)
                results.append({
                    'filename': fname,
                    'left': bounds.left,
                    'bottom': bounds.bottom,
                    'right': bounds.right,
                    'top': bounds.top,
                    'crs': crs.to_string() if crs else ''
                })
            except Exception as e:
                print(f"{fname} 처리 중 오류: {e}")
    if results:
        df = pd.DataFrame(results)
        df.to_csv(RESULT_CSV, index=False)
        print(f'Results saved to {RESULT_CSV}')
    else:
        print('좌표 정보가 추출된 이미지가 없습니다.')

if __name__ == '__main__':
    main()
