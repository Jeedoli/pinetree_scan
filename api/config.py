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

# YOLO 기본 파라미터 (mAP50: 75.8% 모델에 최적화)
DEFAULT_CONFIDENCE = 0.15  # 학습 성능 기반 최적 신뢰도 (Precision 74.9% 고려)
DEFAULT_IOU_THRESHOLD = 0.45  # 표준 YOLO IoU 임계값

# 전처리 기본 설정 (디테일 보존을 위한 1024 복원)
DEFAULT_TILE_SIZE = 1024  # 2048 → 1024 (세밀한 탐지)
# DEFAULT_BBOX_SIZE = 32    # ❌ 고정 크기 제거 - Multi-scale detection 지원

# 🎯 Multi-Scale Detection 설정 (다양한 크기의 피해목 탐지)
MIN_BBOX_SIZE = 10        # 최소 바운딩박스 크기 (작은 피해목)
MAX_BBOX_SIZE = 200       # 최대 바운딩박스 크기 (큰 피해목)
DEFAULT_BBOX_SIZES = [16, 32, 64, 128]  # 다중 스케일 앵커 크기

# 🌲 소나무 전용 시각화 설정
VISUALIZATION_BBOX_COLOR = (0, 255, 0)          # 기본 녹색
VISUALIZATION_BBOX_THICKNESS = 2                # 얇은 선 두께
VISUALIZATION_FONT_SCALE = 0.5                  # 작은 폰트 크기
VISUALIZATION_FONT_THICKNESS = 1                # 얇은 폰트
VISUALIZATION_TEXT_COLOR = (255, 255, 255)      # 흰색 텍스트

# 🎯 소나무 전용 탐지 임계값 (Multi-scale 최적화)
DEFAULT_CONFIDENCE_THRESHOLD = 0.15             # 향상된 신뢰도 (오탐지 감소)
DEFAULT_IOU_THRESHOLD = 0.45                    # 적절한 IoU (중복 제거)
PINE_DETECTION_MIN_CONFIDENCE = 0.05            # 최소 신뢰도 (균형)
PINE_DETECTION_MAX_DETECTIONS = 200             # 증가된 최대 탐지 수

# � Multi-Scale Detection 파라미터
MULTISCALE_INFERENCE = True                     # 다중 스케일 추론 활성화
SCALE_FACTORS = [0.8, 1.0, 1.2, 1.5]          # 다양한 스케일로 추론
NMS_IOU_THRESHOLD = 0.3                         # Non-Maximum Suppression IoU
CONFIDENCE_SCALE_ADJUSTMENT = True              # 크기별 신뢰도 조정

# �🌲 바운딩 박스 크기 조정 (제거된 고정값 대신 동적 조정)
BBOX_SCALE_FACTOR = 0.9                         # 10% 축소 (더 정확한 크기)
DYNAMIC_BBOX_SIZING = True                      # 동적 바운딩박스 크기 조정
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
