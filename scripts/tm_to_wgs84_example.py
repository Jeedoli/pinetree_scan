x, y = 361920.16, 327375.90

# ---------------------------------------------
# TM중부 → WGS84(위도/경도) 좌표 변환 예제 스크립트
# ---------------------------------------------

from pyproj import Transformer

# TM중부(EPSG:5186) 좌표계를 WGS84(EPSG:4326)로 변환하는 Transformer 생성
transformer = Transformer.from_crs(5186, 4326, always_xy=True)

# 예시 TM 좌표 (x, y)
x, y = 361920.16, 327375.90

# 변환 실행: TM → 위도/경도
lon, lat = transformer.transform(x, y)

# 결과 출력
print(f"TM중부({x}, {y}) → 위도경도: {lat:.8f}, {lon:.8f}")
