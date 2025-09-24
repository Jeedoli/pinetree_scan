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

# YOLO 기본 파라미터 (실제 모델 성능에 맞춘 최적화)
DEFAULT_CONFIDENCE = 0.02  # 0.25 → 0.02 (실제 모델 탐지 신뢰도에 맞춤)
DEFAULT_IOU_THRESHOLD = 0.45  # 0.25 → 0.45 (기본 YOLO 설정)

# 전처리 기본 설정 (디테일 보존을 위한 1024 복원)
DEFAULT_TILE_SIZE = 1024  # 2048 → 1024 (세밀한 탐지)
DEFAULT_BBOX_SIZE = 32    # 24 → 32 (중간 크기 피해목 최적화)

# 🌲 소나무 전용 시각화 설정
VISUALIZATION_BBOX_COLOR = (0, 255, 0)          # 기본 녹색
VISUALIZATION_BBOX_THICKNESS = 2                # 얇은 선 두께
VISUALIZATION_FONT_SCALE = 0.5                  # 작은 폰트 크기
VISUALIZATION_FONT_THICKNESS = 1                # 얇은 폰트
VISUALIZATION_TEXT_COLOR = (255, 255, 255)      # 흰색 텍스트

# 🎯 소나무 전용 탐지 임계값 (실제 성능 기반 조정)
DEFAULT_CONFIDENCE_THRESHOLD = 0.02             # 실제 탐지 성능에 맞춘 신뢰도
DEFAULT_IOU_THRESHOLD = 0.45                    # 적절한 IoU (중복 제거)
PINE_DETECTION_MIN_CONFIDENCE = 0.01            # 최소 신뢰도 (매우 낮게)
PINE_DETECTION_MAX_DETECTIONS = 100             # 최대 탐지 수

# 🌲 바운딩 박스 크기 조정
BBOX_SCALE_FACTOR = 0.85                        # 15% 축소
BBOX_MIN_SIZE = 20                              # 최소 바운딩 박스 크기
DEFAULT_CLASS_ID = 0

# 시각화 기본 설정
DEFAULT_BOX_COLOR_BGR = (0, 0, 255)  # 빨간색
DEFAULT_BOX_THICKNESS = 2

# 파일 업로드 제한
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB (대용량 GeoTIFF 파일 지원)
MAX_SINGLE_IMAGE_SIZE = 200 * 1024 * 1024  # 200MB (단일 이미지 추론용)
ALLOWED_IMAGE_EXTENSIONS = {'.tif', '.tiff', '.jpg', '.jpeg', '.png'}
ALLOWED_CSV_EXTENSIONS = {'.csv'}
ALLOWED_TFW_EXTENSIONS = {'.tfw'}

# API 서버 설정
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
