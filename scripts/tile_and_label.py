import datetime
import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window

# ---------------------------------------------
# 대용량 이미지 타일 분할 및 YOLO 라벨 자동 생성 스크립트
# ---------------------------------------------
#
# 사용법:
# 1. 원본 이미지(.tif), 좌표 변환 파일(.tfw), 피해목 위치 CSV 파일을 설정합니다.
# 2. 타일 크기, 클래스 ID, 바운딩박스 크기 등 필요한 설정을 조정합니다.
# 3. 스크립트를 실행하면 타일 이미지와 YOLO 라벨 파일이 생성됩니다.
#
# 명령어:
# poetry run python tile_and_label.py
#
# 특이사항:
# - 라벨이 없는 타일은 .tif 파일도 저장되지 않습니다.
# - 출력 폴더는 현재 날짜를 기준으로 하위 폴더가 자동 생성됩니다.
# - 타일 이미지와 라벨 파일은 각각 지정된 폴더에 저장됩니다.
#

# ====== 설정 ======
SRC_TIF = "data/inference_images/sample02.tiff"  # 원본 이미지 경로
SRC_TFW = "data/inference_images/sample02.tfw"  # 좌표 변환 파일
SRC_CSV = "data/inference_images/sample02.csv"  # 피해목 위치 CSV
OUT_IMG_DIR = "data/tiles/images"  # 타일 이미지 저장 폴더
OUT_LBL_DIR = "data/tiles/labels"  # 타일 라벨 저장 폴더
TILE_SIZE = 1024  # 타일 한 변 크기 (픽셀)
CLASS_ID = 0  # YOLO 클래스 ID (피해목)
BBOX_SIZE = 16  # 바운딩박스 크기 (픽셀)

# 현재 날짜를 기반으로 하위 폴더 생성
current_date = datetime.datetime.now().strftime("%m%d")
OUT_IMG_DIR = os.path.join(OUT_IMG_DIR, current_date)  # 타일 이미지 저장 폴더
OUT_LBL_DIR = os.path.join(OUT_LBL_DIR, current_date)  # 타일 라벨 저장 폴더

# 출력 폴더 생성
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)


def load_tfw(tfw_path):
    """
    TFW 파일에서 변환 파라미터 읽기
    - return: [A, D, B, E, C, F] 리스트
    """
    with open(tfw_path) as f:
        vals = [float(x.strip()) for x in f.readlines()]
    return vals


def tm_to_pixel(x, y, tfw):
    """
    TM 좌표(x, y)를 이미지 픽셀 좌표(px, py)로 변환
    - tfw: [A, D, B, E, C, F]
    """
    A, D, B, E, C, F = tfw
    px = (x - C) / A
    py = (y - F) / E
    return px, py


def main():
    """
    이미지 타일 분할 및 각 타일별 YOLO 라벨(txt) 생성
    """
    tfw = load_tfw(SRC_TFW)
    df = pd.read_csv(SRC_CSV)
    with rasterio.open(SRC_TIF) as src:
        width, height = src.width, src.height
        n_tiles_x = int(np.ceil(width / TILE_SIZE))
        n_tiles_y = int(np.ceil(height / TILE_SIZE))
        total_labels = 0  # 총 라벨 개수 초기화
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                x0 = tx * TILE_SIZE
                y0 = ty * TILE_SIZE
                w = min(TILE_SIZE, width - x0)
                h = min(TILE_SIZE, height - y0)
                window = Window(x0, y0, w, h)
                # RGB 3채널만 추출하여 타일 이미지 생성
                tile_img = src.read([1, 2, 3], window=window)
                tile_name = f"sample01_{tx}_{ty}.tif"
                # 타일 내 bbox 라벨 생성
                lines = []
                for _, row in df.iterrows():
                    px, py = tm_to_pixel(row["x"], row["y"], tfw)
                    # 타일 내 상대좌표로 변환
                    rel_x = px - x0
                    rel_y = py - y0
                    if 0 <= rel_x < w and 0 <= rel_y < h:
                        x_center = rel_x / w
                        y_center = rel_y / h
                        bw = BBOX_SIZE / w
                        bh = BBOX_SIZE / h
                        lines.append(
                            f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"
                        )
                if len(lines) > 0:  # 라벨이 있는 경우에만 저장
                    out_lbl_path = os.path.join(
                        OUT_LBL_DIR, tile_name.replace(".tif", ".txt")
                    )
                    with open(out_lbl_path, "w") as f:
                        f.write("\n".join(lines))
                    total_labels += len(lines)  # 각 타일의 라벨 개수를 누적
                    print(f"타일 {tile_name} 저장, 라벨 {len(lines)}개")

                    # 타일 이미지 저장
                    out_img_path = os.path.join(OUT_IMG_DIR, tile_name)
                    orig_affine = src.transform
                    tile_affine = orig_affine * Affine.translation(x0, y0)
                    with rasterio.open(
                        out_img_path,
                        "w",
                        driver="GTiff",
                        height=h,
                        width=w,
                        count=3,
                        dtype=src.dtypes[0],
                        crs=src.crs,
                        transform=tile_affine,
                    ) as dst:
                        dst.write(tile_img)
                else:
                    print(f"타일 {tile_name} 저장되지 않음 (라벨 없음)")
    print(f"총 타일: {n_tiles_x * n_tiles_y}")
    print(f"총 라벨: {total_labels}")


if __name__ == "__main__":
    # 메인 실행: 타일 분할 및 라벨 생성
    main()
