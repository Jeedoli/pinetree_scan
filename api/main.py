# FastAPI 메인 애플리케이션
# 소나무재선충병 피해목 탐지 API 서버

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import uvicorn
import os
import tempfile
import shutil
from typing import List, Optional
import json
from datetime import datetime
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from api import config

# 라우터 import
from api.routers import inference, preprocessing, visualization, model_performance, ai_analysis

# FastAPI 앱 생성
app = FastAPI(
    title="🌲 Pinetree Damage Detection & AI Analysis API",
    description="소나무재선충병 피해목 자동 탐지 + RAG 기반 AI 분석 REST API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# 대용량 파일 업로드를 위한 설정
app.state.max_upload_size = config.MAX_FILE_SIZE

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(inference.router, prefix="/api/v1/inference", tags=["추론/탐지"])
app.include_router(preprocessing.router, prefix="/api/v1/preprocessing", tags=["전처리"])
app.include_router(visualization.router, prefix="/api/v1/visualization", tags=["시각화"])
app.include_router(model_performance.router, prefix="/api/v1/model-performance", tags=["모델 성능 분석"])
app.include_router(ai_analysis.router, prefix="/api/v1/ai-analysis", tags=["🤖 AI 분석 챗봇"])

# 서버 시작 시 필요한 디렉토리 생성
@app.on_event("startup")
async def startup_event():
    config.ensure_directories()
    print("🌲 Pinetree Damage Detection API 서버 시작됨")
    print(f"📁 출력 디렉토리 준비 완료: {config.API_OUTPUT_BASE}")
    
    # 사용 가능한 모델 확인
    available_model = config.get_available_model()
    if available_model:
        print(f"✅ 기본 모델 발견: {available_model}")
    else:
        print("⚠️  경고: 기본 모델 파일을 찾을 수 없습니다. 추론 API 사용 전 모델 파일을 확인하세요.")

# 루트 엔드포인트
@app.get("/")
async def root():
    return {
        "message": "🌲 Pinetree Damage Detection & AI Analysis API",
        "version": "2.0.0",
        "description": "소나무재선충병 피해목 자동 탐지 + RAG 기반 AI 분석 REST API",
        "docs": "/docs",
        "max_file_size": f"{config.MAX_FILE_SIZE / (1024**3):.1f}GB",
        "features": [
            "🎯 3단계 멀티스케일 YOLO 탐지",
            "🤖 RAG 기반 AI 챗봇 분석",
            "📊 종합 리포트 자동 생성",
            "🗺️ GPS 좌표 정밀 변환"
        ],
        "endpoints": {
            "inference": "/api/v1/inference",
            "preprocessing": "/api/v1/preprocessing", 
            "visualization": "/api/v1/visualization",
            "ai_analysis": "/api/v1/ai-analysis",
            "model_performance": "/api/v1/model-performance"
        },
        "recommended_apis": {
            "🚀 통합 딥러닝 데이터셋 생성": "/api/v1/preprocessing/create_complete_training_dataset",
            "🔍 이미지 추론": "/api/v1/inference/predict",
            "📊 배치 추론": "/api/v1/inference/batch_predict", 
            "🎨 결과 시각화": "/api/v1/visualization/create_visualization",
            "🎯 모델 성능 분석": "/api/v1/model-performance/analyze"
        },
        "legacy_apis": {
            "⚠️ 타일링만 (레거시)": "/api/v1/preprocessing/tile_and_label"
        },
        "features": [
            "🎯 24px 바운딩박스 최적화 (개별 나무 탐지)",
            "🚀 Google Colab 최적화 딥러닝 데이터셋",
            "📦 자동 train/validation 분할",
            "🗺️ TM좌표계 ↔ 픽셀좌표 변환",
            "📊 실시간 처리 진행률 표시",
            "🎨 결과 시각화 및 ZIP 다운로드"
        ]
    }

# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "pinetree-damage-api"
    }

# 로그 테스트 엔드포인트 (실시간 로그 확인용)
@app.get("/test_logs")
async def test_logs():
    """실시간 로그 테스트를 위한 엔드포인트"""
    import time
    
    print("🔄 로그 테스트 시작", flush=True)
    
    for i in range(10):
        time.sleep(0.5)  # 0.5초 대기
        message = f"📊 테스트 진행률: {(i+1)*10}% ({i+1}/10)"
        print(message, flush=True)
    
    print("✅ 로그 테스트 완료", flush=True)
    
    return {"message": "로그 테스트 완료", "status": "success"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
