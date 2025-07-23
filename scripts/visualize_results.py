# 지도 시각화 및 통계 분석 예제 스크립트
# --------------------------------------
# 추론 결과 CSV를 활용해 folium 지도 시각화 및 pandas 통계 분석
# --------------------------------------

import folium
import pandas as pd

# 결과 CSV 경로 (필요시 경로 수정)
RESULT_CSV = "../data/infer_results/damaged_trees_gps_latest.csv"

# 1. 결과 CSV 불러오기
df = pd.read_csv(RESULT_CSV)

# 2. 통계 분석 예시
print("클래스별 탐지 개수:")
print(df["class_name"].value_counts())

# 3. 지도 시각화 (folium)
if not df.empty:
    # 중심 좌표(첫 객체 기준)
    center = [df["latitude"].iloc[0], df["longitude"].iloc[0]]
    m = folium.Map(location=center, zoom_start=15)
    for _, row in df.iterrows():
        color = "red" if row["class_id"] == 0 else "green"
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"{row['filename']} ({row['class_name']})",
        ).add_to(m)
    m.save("results_map.html")
    print("지도 시각화 결과: results_map.html 파일 생성")
else:
    print("데이터가 없습니다.")
