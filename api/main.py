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
from . import config

# 라우터 import
from .routers import inference, preprocessing, visualization, utilities

# FastAPI 앱 생성
app = FastAPI(
    title="🌲 Pinetree Damage Detection API",
    description="소나무재선충병 피해목 자동 탐지 REST API",
    version="1.0.0",
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
app.include_router(utilities.router, prefix="/api/v1/utilities", tags=["유틸리티"])

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
        "message": "🌲 Pinetree Damage Detection API",
        "version": "1.0.0",
        "description": "소나무재선충병 피해목 자동 탐지 REST API",
        "docs": "/docs",
        "max_file_size": f"{config.MAX_FILE_SIZE / (1024**3):.1f}GB",
        "endpoints": {
            "inference": "/api/v1/inference",
            "preprocessing": "/api/v1/preprocessing", 
            "visualization": "/api/v1/visualization",
            "utilities": "/api/v1/utilities"
        }
    }

# 헬스체크 엔드포인트
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "pinetree-damage-api"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
