# --------------------------------------
# tile_infer_vs_gt_viewer.py
# --------------------------------------
# sample01_16_10.txt의 YOLO 라벨(초록 박스, GT)과
# 딥러닝 추론 결과 CSV(빨간 X, 위경도→픽셀 변환)를
# 동일 타일 이미지 위에 동시에 시각화하여
# 실제 피해목(정답) 위치와 추론된 피해목 위치를 픽셀 단위로 직접 비교/검증하는 코드
#
# * 초록 박스: 사람이 직접 마킹한 피해목(GT, ground truth)
# * 빨간 X: 딥러닝 추론 결과(모델이 예측한 피해목)
#
# 이 시각화 결과를 통해 실제 탐지 성능, 오차, 오탐/미탐 여부를 직관적으로 확인할 수 있음
#
# 사용법 예시:
# poetry run python scripts/tile_infer_vs_gt_viewer.py
# (타일/라벨/추론결과 경로는 코드 내에서 직접 수정)
# --------------------------------------
import matplotlib.pyplot as plt
import numpy as np
import rasterio

tif_path = "data/tiles/images/sample01_16_10.tif"
txt_path = "data/tiles/labels/sample01_16_10.txt"

# 1. 타일 이미지 불러오기
with rasterio.open(tif_path) as src:
    img = src.read(1)
    height, width = img.shape

# 2. YOLO txt 라벨 읽기 (class x_center y_center w h)
yolo_labels = []
with open(txt_path, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) == 5:
            cls, x_c, y_c, w, h = map(float, parts)
            yolo_labels.append((cls, x_c, y_c, w, h))

# 3. 추론 결과 CSV(위경도) → 픽셀 변환 및 시각화
import pandas as pd
import pyproj

# 추론 결과 CSV 경로
csv_pred = "data/infer_results/damaged_trees_gps_20250728_190111.csv"

# 타일의 affine, CRS 정보
with rasterio.open(tif_path) as src:
    affine = src.transform
    crs = src.crs

# TM 좌표계 정의 (tile_and_label.py와 동일)
crs_tm = (
    "+proj=tmerc +lat_0=38 +lon_0=127 +k=1 "
    "+x_0=200000 +y_0=600000 +datum=WGS84 +units=m +no_defs"
)
proj_wgs_to_tm = pyproj.Transformer.from_crs("EPSG:4326", crs_tm, always_xy=True)

# 추론 결과 중 해당 타일에 해당하는 행만 필터
df_pred = pd.read_csv(csv_pred)
df_pred_tile = df_pred[df_pred["filename"] == "sample01_16_10.tif"]


def tm_to_pixel(x_tm, y_tm, affine):
    col, row = ~affine * (x_tm, y_tm)
    return col, row


fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img, cmap="gray")

# 3-1. GT YOLO 박스(초록)
for cls, x_c, y_c, w, h in yolo_labels:
    xc_pix = x_c * width
    yc_pix = y_c * height
    w_pix = w * width
    h_pix = h * height
    x1 = xc_pix - w_pix / 2
    y1 = yc_pix - h_pix / 2
    rect = plt.Rectangle(
        (x1, y1), w_pix, h_pix, edgecolor="lime", facecolor="none", linewidth=2
    )
    ax.add_patch(rect)

# 3-2. 추론 결과 박스(빨간색, 중심점만 마킹)
for _, row in df_pred_tile.iterrows():
    lat, lon = row["latitude"], row["longitude"]
    # 위경도 → TM
    x_tm, y_tm = proj_wgs_to_tm.transform(lon, lat)
    # TM → 픽셀
    col, row_pix = tm_to_pixel(x_tm, y_tm, affine)
    # 중심점만 빨간 X로 표시
    ax.scatter(
        [col],
        [row_pix],
        c="red",
        marker="x",
        s=80,
        label=(
            "추론 결과" if "추론 결과" not in ax.get_legend_handles_labels()[1] else ""
        ),
    )

ax.set_title("sample01_16_10 YOLO GT(초록) vs 추론 결과(빨강 X)")
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())
plt.tight_layout()
plt.show()
