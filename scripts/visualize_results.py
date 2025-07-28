# --------------------------------------
# 추론 결과 CSV를 folium 지도에 시각화하고, pandas로 통계 분석하는 스크립트
# --------------------------------------
# 사용법 예시:
# poetry run python scripts/visualize_results.py --csv data/infer_results/damaged_trees_gps_20250725_164727.csv --output data/infer_results/visualization.html
#
# --csv: 추론 결과 CSV 경로
# --output: 저장할 지도 HTML 파일 경로(기본값: results_map.html)
# --------------------------------------

import argparse

import folium
import pandas as pd


def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(
        description="추론 결과 CSV를 지도에 시각화 및 통계 분석"
    )
    parser.add_argument("--csv", type=str, required=True, help="추론 결과 CSV 경로")
    parser.add_argument(
        "--output", type=str, default="results_map.html", help="지도 HTML 저장 경로"
    )
    args = parser.parse_args()

    # 1. 결과 CSV 불러오기
    df = pd.read_csv(args.csv)

    # 2. 클래스별 탐지 개수 통계 출력
    print("클래스별 탐지 개수:")
    print(df["class_name"].value_counts())

    # 3. folium 지도 시각화
    if not df.empty:
        # 첫 번째 객체의 좌표를 지도 중심으로 설정
        center = [df["latitude"].iloc[0], df["longitude"].iloc[0]]
        m = folium.Map(location=center, zoom_start=15)
        for _, row in df.iterrows():
            # 피해목(0)은 빨간색, 정상목(1)은 초록색 마커
            color = "red" if row["class_id"] == 0 else "green"
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"{row['filename']} ({row['class_name']})",
            ).add_to(m)
        m.save(args.output)
        print(f"지도 시각화 결과: {args.output} 파일 생성")
    else:
        print("데이터가 없습니다.")


if __name__ == "__main__":
    main()
