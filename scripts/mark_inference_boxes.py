# --------------------------------------
# 여러 타일 이미지에 추론 결과 박스 자동 마킹 스크립트 (OpenCV)
# --------------------------------------
# 사용법 예시:
# poetry run python scripts/mark_inference_boxes.py --tiles_dir data/tiles/images/0729 --csv data/infer_results/0730_tiles_inference_results.csv
# --tiles_dir: 타일 이미지 폴더 경로
# --csv: 추론 결과 CSV 경로 (filename 컬럼 포함)
# --------------------------------------

import argparse
import datetime
import os

import cv2
import pandas as pd
import rasterio


def main():
    parser = argparse.ArgumentParser(
        description="여러 타일 이미지에 추론 결과 박스 자동 마킹"
    )
    parser.add_argument(
        "--tiles_dir", type=str, required=True, help="타일 이미지 폴더 경로"
    )
    parser.add_argument("--csv", type=str, required=True, help="추론 결과 CSV 경로")
    args = parser.parse_args()

    # 마킹 이미지 저장 폴더 (날짜 기반)
    today_date = datetime.datetime.now().strftime("%Y%m%d")
    out_dir = os.path.join(
        "data", "infer_results", "inference_tiles_marked", today_date
    )
    os.makedirs(out_dir, exist_ok=True)

    # 추론 결과 CSV 불러오기
    df = pd.read_csv(args.csv)

    # 폴더 내 모든 tif/tiff 이미지에 대해 반복
    for fname in os.listdir(args.tiles_dir):
        if not fname.lower().endswith((".tif", ".tiff")):
            continue
        img_path = os.path.join(args.tiles_dir, fname)
        print(f"[DEBUG] 처리 중: {fname}")
        # 해당 이미지에 대한 결과만 필터링
        df_img = df[df["filename"] == fname]
        print(f"[DEBUG] 매칭된 row 수: {len(df_img)} (filename={fname})")
        if df_img.empty:
            print(f"[DEBUG] {fname}에 해당하는 추론 결과 없음. 건너뜀.")
            continue
        try:
            # 이미지 불러오기
            with rasterio.open(img_path) as src:
                img = src.read()
                img = img.transpose(1, 2, 0)
                if img.shape[2] > 3:
                    img = img[:, :, :3]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                # 박스 마킹
                for _, row in df_img.iterrows():
                    if all(k in row for k in ["xmin", "ymin", "xmax", "ymax"]):
                        xmin, ymin, xmax, ymax = (
                            int(row["xmin"]),
                            int(row["ymin"]),
                            int(row["xmax"]),
                            int(row["ymax"]),
                        )
                    elif all(k in row for k in ["x", "y"]):
                        x, y = int(row["x"]), int(row["y"])
                        w, h = 20, 20
                        xmin, ymin, xmax, ymax = (
                            x - w // 2,
                            y - h // 2,
                            x + w // 2,
                            y + h // 2,
                        )
                    elif all(
                        k in row for k in ["center_x", "center_y", "width", "height"]
                    ):
                        # YOLO 형식 좌표를 픽셀 좌표로 변환
                        center_x, center_y = float(row["center_x"]), float(
                            row["center_y"]
                        )
                        width, height = float(row["width"]), float(row["height"])
                        image_width, image_height = img.shape[1], img.shape[0]

                        xmin = int((center_x - width / 2) * image_width)
                        ymin = int((center_y - height / 2) * image_height)
                        xmax = int((center_x + width / 2) * image_width)
                        ymax = int((center_y + height / 2) * image_height)

                        # 박스가 이미지 영역 내에 있는지 체크
                        if (
                            xmin < 0
                            or ymin < 0
                            or xmax > image_width
                            or ymax > image_height
                        ):
                            print(
                                f"[WARNING] 박스가 이미지 영역 밖: {xmin},{ymin},{xmax},{ymax}"
                            )
                            continue
                    else:
                        print(f"[DEBUG] 좌표 정보 없음: {row}")
                        continue

                    color = (
                        (0, 0, 255)
                        if "class_id" in row and row["class_id"] == 0
                        else (0, 255, 0)
                    )
                    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                # 저장 경로
                base = os.path.splitext(fname)[0]
                out_path = os.path.join(out_dir, f"{base}_marked.png")
                cv2.imwrite(out_path, img)
                print(f"{fname} → {out_path} 저장 완료")
        except Exception as e:
            print(f"[ERROR] {fname} 처리 중 에러 발생: {e}")


if __name__ == "__main__":
    main()
