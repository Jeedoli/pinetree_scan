# API 설정 및 환경 변수
import os
from pathlib import Path

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent
API_ROOT = Path(__file__).parent

# 기본 디렉토리 설정
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# API 출력 디렉토리
API_OUTPUT_BASE = DATA_DIR / "api_outputs"
API_RESULTS_DIR = API_OUTPUT_BASE / "inference"
API_TILES_DIR = API_OUTPUT_BASE / "tiles"
API_VISUALIZATION_DIR = API_OUTPUT_BASE / "visualizations"

# 기본 모델 설정  
DEFAULT_MODEL_PATH = MODELS_DIR / "colab_yolo" / "best.pt"
# FALLBACK_MODEL_PATH는 제거 (실제 존재하는 모델만 사용)

# 🎯 추론(Inference) API 설정
DEFAULT_CONFIDENCE = 0.28  # 추론 시 탐지 신뢰도 임계값 (28%)
DEFAULT_IOU_THRESHOLD = 0.6  # 추론 시 NMS IoU 임계값 (중복 탐지 제거용)

# 📦 데이터셋 생성(Preprocessing) 설정
DEFAULT_TILE_SIZE = 1024  # 데이터셋 생성 시 타일 분할 크기 (1024px)
# 🎯 멀티스케일 바운딩박스 설정 (적응적 크기)
USE_ADAPTIVE_BBOX = True     # 멀티스케일 바운딩박스 사용 여부 (True: 적응적, False: 고정)
DEFAULT_BBOX_SIZE = 32       # 고정 모드일 때 사용할 크기 (32px)
ADAPTIVE_BBOX_MIN_SIZE = 16  # 적응적 모드 최소 크기 (16px)
ADAPTIVE_BBOX_MAX_SIZE = 128 # 적응적 모드 최대 크기 (128px)

# 🎨 추론 결과 시각화 설정
VISUALIZATION_BBOX_COLOR = (0, 255, 0)          # 탐지된 바운딩박스 색상 (녹색)
VISUALIZATION_BBOX_THICKNESS = 2                # 바운딩박스 선 두께
VISUALIZATION_FONT_SCALE = 0.5                  # 신뢰도 텍스트 폰트 크기
VISUALIZATION_FONT_THICKNESS = 1                # 신뢰도 텍스트 두께
VISUALIZATION_TEXT_COLOR = (255, 255, 255)      # 신뢰도 텍스트 색상 (흰색)

# 📦 데이터셋 및 추론 공통 설정
DEFAULT_CLASS_ID = 0  # 피해목 클래스 ID

# 🎨 추론 시각화 기본 설정 (호환성)
DEFAULT_BOX_COLOR_BGR = (0, 0, 255)  # 기본 바운딩박스 색상 (빨간색)
DEFAULT_BOX_THICKNESS = 2  # 기본 바운딩박스 선 두께

# 📁 파일 업로드 제한 설정
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB (데이터셋 생성용 대용량 GeoTIFF 지원)
MAX_SINGLE_IMAGE_SIZE = 200 * 1024 * 1024  # 200MB (추론용 단일 이미지 제한)
ALLOWED_IMAGE_EXTENSIONS = {'.tif', '.tiff', '.jpg', '.jpeg', '.png'}  # 지원되는 이미지 형식
ALLOWED_CSV_EXTENSIONS = {'.csv'}  # 데이터셋 생성용 GPS 좌표 파일 형식
ALLOWED_TFW_EXTENSIONS = {'.tfw'}  # 지리참조 정보 파일 형식

# 🌐 API 서버 설정
API_HOST = "0.0.0.0"
API_PORT = 8000

def ensure_directories():
    """필요한 디렉토리들을 생성합니다."""
    directories = [
        API_OUTPUT_BASE,
        API_RESULTS_DIR,
        API_TILES_DIR, 
        API_VISUALIZATION_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_available_model():
    """사용 가능한 모델 경로를 반환합니다."""
    if DEFAULT_MODEL_PATH.exists():
        return str(DEFAULT_MODEL_PATH)
    else:
        return None
