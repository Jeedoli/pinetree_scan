from pyproj import Transformer

# TM중부(EPSG:5186) → WGS84(EPSG:4326)
transformer = Transformer.from_crs(5186, 4326, always_xy=True)

x, y = 361920.16, 327375.90
lon, lat = transformer.transform(x, y)
print(f"TM중부({x}, {y}) → 위도경도: {lat:.8f}, {lon:.8f}")
