import os
import rasterio
import numpy as np
import pandas as pd
from rasterio.windows import Window
from rasterio.transform import Affine

# 설정
SRC_TIF = 'data/inference_images/sample01.tif'
SRC_TFW = 'data/inference_images/sample01.tfw'
SRC_CSV = 'data/inference_images/damaged_tree.csv'
OUT_IMG_DIR = 'data/tiles/images'
OUT_LBL_DIR = 'data/tiles/labels'
TILE_SIZE = 1024
CLASS_ID = 0
BBOX_SIZE = 16  # 픽셀 단위

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

def load_tfw(tfw_path):
    with open(tfw_path) as f:
        vals = [float(x.strip()) for x in f.readlines()]
    return vals  # [A, D, B, E, C, F]

def tm_to_pixel(x, y, tfw):
    A, D, B, E, C, F = tfw
    px = (x - C) / A
    py = (y - F) / E
    return px, py

def main():
    tfw = load_tfw(SRC_TFW)
    df = pd.read_csv(SRC_CSV)
    with rasterio.open(SRC_TIF) as src:
        width, height = src.width, src.height
        n_tiles_x = int(np.ceil(width / TILE_SIZE))
        n_tiles_y = int(np.ceil(height / TILE_SIZE))
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                x0 = tx * TILE_SIZE
                y0 = ty * TILE_SIZE
                w = min(TILE_SIZE, width - x0)
                h = min(TILE_SIZE, height - y0)
                window = Window(x0, y0, w, h)
                # RGB 3채널만 추출
                tile_img = src.read([1, 2, 3], window=window)
                tile_name = f'sample01_{tx}_{ty}.tif'
                out_img_path = os.path.join(OUT_IMG_DIR, tile_name)
                orig_affine = src.transform
                tile_affine = orig_affine * Affine.translation(x0, y0)
                with rasterio.open(
                    out_img_path, 'w',
                    driver='GTiff',
                    height=h, width=w,
                    count=3,
                    dtype=src.dtypes[0],
                    crs=src.crs,
                    transform=tile_affine
                ) as dst:
                    dst.write(tile_img)
                # 타일 내 bbox 라벨 생성
                lines = []
                for _, row in df.iterrows():
                    px, py = tm_to_pixel(row['x'], row['y'], tfw)
                    # 타일 내 상대좌표
                    rel_x = px - x0
                    rel_y = py - y0
                    if 0 <= rel_x < w and 0 <= rel_y < h:
                        x_center = rel_x / w
                        y_center = rel_y / h
                        bw = BBOX_SIZE / w
                        bh = BBOX_SIZE / h
                        lines.append(f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")
                out_lbl_path = os.path.join(OUT_LBL_DIR, tile_name.replace('.tif', '.txt'))
                with open(out_lbl_path, 'w') as f:
                    f.write('\n'.join(lines))
                print(f"타일 {tile_name} 저장, 라벨 {len(lines)}개")
    print(f"총 타일: {n_tiles_x * n_tiles_y}")

if __name__ == '__main__':
    main()
