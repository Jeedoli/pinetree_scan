import argparse
import os

import cv2
import pandas as pd
import pyproj
import rasterio


def main():
    parser = argparse.ArgumentParser(description="위경도→픽셀 좌표 검증")
    parser.add_argument("--img", type=str, required=True, help="검증할 이미지 경로")
    parser.add_argument("--csv", type=str, required=True, help="검증할 CSV 경로")
    args = parser.parse_args()

    img_path = args.img
    csv_path = args.csv
    img_name = os.path.basename(img_path)
    out_path = f"좌표검증_{img_name.replace('.tif', '')}.png"

    img = cv2.imread(img_path)
    if img is None:
        print(f"이미지 파일을 읽을 수 없습니다: {img_path}")
        return
    df = pd.read_csv(csv_path)
    df_img = df[(df["filename"] == img_name) & (df["class_id"] == 0)]

    with rasterio.open(img_path) as src:
        for _, row in df_img.iterrows():
            lat, lon = float(row["latitude"]), float(row["longitude"])
            # 이미지 CRS가 EPSG:4326이 아니면 변환
            if src.crs and not src.crs.to_string().startswith("EPSG:4326"):
                proj = pyproj.Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x_img, y_img = proj.transform(lon, lat)
                col, row_pix = src.index(x_img, y_img)
            else:
                col, row_pix = src.index(lon, lat)
            print(f"위경도({lat},{lon}) → 이미지 픽셀({col},{row_pix})")
            # 박스 그리기
            w, h = 20, 20
            xmin, ymin = col - w // 2, row_pix - h // 2
            xmax, ymax = col + w // 2, row_pix + h // 2
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

    cv2.imwrite(out_path, img)
    print(f"검증 결과 이미지 저장: {out_path}")


if __name__ == "__main__":
    main()
