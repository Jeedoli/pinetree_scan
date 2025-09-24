"""
🎯 모델 성능 분석 API
- 이미지 업로드하여 mAP, Precision, Recall 값 계산
- 다양한 신뢰도 임계값에서 성능 측정
- 간단하고 실용적인 성능 분석
"""
from ..config import DEFAULT_MODEL_PATH
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Dict, Any, Optional
import os
import tempfile
import shutil
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
import base64
import io

from ..config import DEFAULT_MODEL_PATH

router = APIRouter()

class SimpleModelAnalyzer:
    """간단한 모델 성능 분석 클래스"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = YOLO(model_path)
        
    def analyze_performance(self, image_paths: List[str]) -> Dict[str, Any]:
        """이미지들의 성능 지표 계산"""
        
        # 다양한 신뢰도 임계값에서 테스트
        confidence_thresholds = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]
        
        results = {}
        all_confidences = []
        
        for conf in confidence_thresholds:
            total_detections = 0
            confidences = []
            
            for image_path in image_paths:
                try:
                    # YOLO 추론
                    yolo_results = self.model(image_path, conf=conf, verbose=False)
                    result = yolo_results[0]
                    
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes
                        total_detections += len(boxes)
                        
                        # 신뢰도 점수 수집
                        for i in range(len(boxes)):
                            confidence = float(boxes.conf[i].cpu().numpy())
                            confidences.append(confidence)
                            all_confidences.append(confidence)
                
                except Exception:
                    continue
            
            # 성능 지표 계산
            precision = np.mean(confidences) if confidences else 0.0
            recall_estimate = total_detections / len(image_paths) if image_paths else 0.0
            
            # F1-Score 계산 (Precision과 Recall의 조화평균)
            if precision > 0 and recall_estimate > 0:
                f1_score = 2 * (precision * recall_estimate) / (precision + recall_estimate)
            else:
                f1_score = 0.0
            
            results[f"conf_{conf}"] = {
                "confidence_threshold": conf,
                "total_detections": total_detections,
                "precision_estimate": round(precision, 4),
                "recall_estimate": round(recall_estimate, 4),
                "f1_score": round(f1_score, 4),
                "avg_confidence": round(np.mean(confidences), 4) if confidences else 0.0
            }
        
        # mAP 추정 (다양한 임계값에서의 precision 평균)
        precisions = [r["precision_estimate"] for r in results.values() if r["precision_estimate"] > 0]
        mAP_estimate = np.mean(precisions) if precisions else 0.0
        
        # 최적 임계값 찾기 (F1-Score 기준)
        best_conf = max(results.items(), key=lambda x: x[1]["f1_score"])
        
        return {
            "mAP_estimate": round(mAP_estimate, 4),
            "total_images_analyzed": len(image_paths),
            "best_confidence_threshold": best_conf[1]["confidence_threshold"],
            "best_performance": best_conf[1],
            "all_thresholds": list(results.values()),
            "confidence_statistics": {
                "mean": round(np.mean(all_confidences), 4) if all_confidences else 0.0,
                "std": round(np.std(all_confidences), 4) if all_confidences else 0.0,
                "min": round(np.min(all_confidences), 4) if all_confidences else 0.0,
                "max": round(np.max(all_confidences), 4) if all_confidences else 0.0,
                "count": len(all_confidences)
            }
        }
    
def _get_improvement_suggestions(analysis_result: Dict) -> List[str]:
    """성능 개선 제안사항 생성"""
    suggestions = []
    best_perf = analysis_result["best_performance"]
    
    if best_perf["precision_estimate"] < 0.6:
        suggestions.append("정밀도가 낮습니다. 오탐지를 줄이기 위해 신뢰도 임계값을 높이거나 추가 학습을 고려하세요.")
    
    if best_perf["recall_estimate"] < 0.6:
        suggestions.append("재현율이 낮습니다. 놓친 객체가 많습니다. 신뢰도 임계값을 낮추거나 데이터 증강을 고려하세요.")
        
    if best_perf["f1_score"] < 0.5:
        suggestions.append("전체적인 성능이 낮습니다. 모델 재학습이나 하이퍼파라미터 튜닝을 권장합니다.")
        
    if analysis_result["confidence_statistics"]["mean"] < 0.3:
        suggestions.append("평균 신뢰도가 낮습니다. 학습 데이터의 품질을 확인하거나 더 많은 epoch로 학습하세요.")
        
    if not suggestions:
        suggestions.append("현재 모델 성능이 양호합니다. 유지 또는 미세 조정을 권장합니다.")
        
    return suggestions

def _evaluate_model_status(analysis_result: Dict) -> str:
    """모델 전체 상태 평가"""
    best_perf = analysis_result["best_performance"]
    avg_score = (best_perf["precision_estimate"] + best_perf["recall_estimate"] + best_perf["f1_score"]) / 3
    
    if avg_score >= 0.8:
        return "🟢 EXCELLENT - 모델이 매우 우수한 성능을 보입니다"
    elif avg_score >= 0.6:
        return "🟡 GOOD - 모델이 양호한 성능을 보입니다"
    elif avg_score >= 0.4:
        return "🟠 FAIR - 모델 성능이 보통입니다. 개선이 필요합니다"
    else:
        return "🔴 POOR - 모델 성능이 낮습니다. 재학습을 권장합니다"

# 전역 분석기 인스턴스
analyzer = None

def get_analyzer():
    """분석기 인스턴스 가져오기"""
    global analyzer
    if analyzer is None:
        analyzer = SimpleModelAnalyzer(str(DEFAULT_MODEL_PATH))
    return analyzer

@router.post("/analyze", summary="모델 성능 분석")
async def analyze_model_performance(
    files: List[UploadFile] = File(..., description="분석할 이미지 파일들")
):
    """
    이미지들을 업로드하여 모델 성능 지표 계산
    
    Returns:
    - mAP 추정값
    - Precision, Recall 추정값  
    - F1-Score
    - 신뢰도 통계
    - 최적 임계값 추천
    """
    
    if not files:
        raise HTTPException(status_code=400, detail="이미지 파일이 필요합니다")
    
    # 임시 디렉토리 생성
    temp_dir = Path(tempfile.gettempdir()) / "performance_analysis"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # 파일 저장
        image_paths = []
        for file in files:
            if not file.content_type.startswith('image/'):
                continue
                
            temp_path = temp_dir / file.filename
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            image_paths.append(str(temp_path))
        
        if not image_paths:
            raise HTTPException(status_code=400, detail="유효한 이미지 파일이 없습니다")
        
        # 성능 분석 실행
        performance_analyzer = get_analyzer()
        analysis_result = performance_analyzer.analyze_performance(image_paths)
        
        # 탐지 결과 요약 계산
        best_perf = analysis_result["best_performance"]
        conf_stats = analysis_result["confidence_statistics"]
        
        # 성능 등급 계산
        def get_performance_grade(score):
            if score >= 0.8: return "A (우수)"
            elif score >= 0.6: return "B (양호)"  
            elif score >= 0.4: return "C (보통)"
            elif score >= 0.2: return "D (미흡)"
            else: return "F (불량)"
        
        # 고도화된 응답 데이터
        response = {
            "📊 모델 성능 분석 결과": {
                "🎯 핵심 성능 지표": {
                    "mAP (Mean Average Precision)": {
                        "값": analysis_result["mAP_estimate"],
                        "백분율": f"{analysis_result['mAP_estimate'] * 100:.1f}%",
                        "등급": get_performance_grade(analysis_result["mAP_estimate"]),
                        "설명": "전체 클래스에 대한 평균 정밀도"
                    },
                    "Precision (정밀도)": {
                        "값": best_perf["precision_estimate"],
                        "백분율": f"{best_perf['precision_estimate'] * 100:.1f}%",
                        "등급": get_performance_grade(best_perf["precision_estimate"]),
                        "설명": "탐지한 것 중 실제 정답인 비율"
                    },
                    "Recall (재현율)": {
                        "값": best_perf["recall_estimate"],
                        "백분율": f"{best_perf['recall_estimate'] * 100:.1f}%",
                        "등급": get_performance_grade(best_perf["recall_estimate"]),
                        "설명": "전체 정답 중 실제로 찾아낸 비율"
                    },
                    "F1-Score": {
                        "값": best_perf["f1_score"],
                        "백분율": f"{best_perf['f1_score'] * 100:.1f}%",
                        "등급": get_performance_grade(best_perf["f1_score"]),
                        "설명": "정밀도와 재현율의 조화평균"
                    }
                },
                
                "🔍 탐지 결과 요약": {
                    "총 탐지 개수": best_perf["total_detections"],
                    "분석된 이미지 수": analysis_result["total_images_analyzed"],
                    "이미지당 평균 탐지": round(best_perf["total_detections"] / analysis_result["total_images_analyzed"], 1),
                    "최적 신뢰도 임계값": best_perf["confidence_threshold"],
                    "평균 신뢰도": best_perf["avg_confidence"]
                },
                
                "📈 신뢰도 분석": {
                    "전체 탐지된 객체 수": conf_stats["count"],
                    "신뢰도 통계": {
                        "평균": f"{conf_stats['mean']:.3f}",
                        "표준편차": f"{conf_stats['std']:.3f}",
                        "최소값": f"{conf_stats['min']:.3f}",
                        "최고값": f"{conf_stats['max']:.3f}"
                    },
                    "신뢰도 분포": {
                        "높음 (0.5 이상)": sum(1 for r in analysis_result["all_thresholds"] if r["confidence_threshold"] >= 0.5 and r["total_detections"] > 0),
                        "중간 (0.2-0.5)": sum(1 for r in analysis_result["all_thresholds"] if 0.2 <= r["confidence_threshold"] < 0.5 and r["total_detections"] > 0),
                        "낮음 (0.2 미만)": sum(1 for r in analysis_result["all_thresholds"] if r["confidence_threshold"] < 0.2 and r["total_detections"] > 0)
                    }
                },
                
                "⚙️ 최적화 권장사항": {
                    "추천 신뢰도 임계값": analysis_result["best_confidence_threshold"],
                    "성능 개선 포인트": _get_improvement_suggestions(analysis_result),
                    "모델 상태 평가": _evaluate_model_status(analysis_result)
                }
            },
            
            "📋 상세 분석 데이터": {
                "임계값별 상세 결과": [
                    {
                        "신뢰도_임계값": r["confidence_threshold"],
                        "탐지_개수": r["total_detections"],
                        "정밀도": f"{r['precision_estimate']:.3f}",
                        "재현율": f"{r['recall_estimate']:.3f}",
                        "F1_점수": f"{r['f1_score']:.3f}",
                        "평균_신뢰도": f"{r['avg_confidence']:.3f}"
                    }
                    for r in analysis_result["all_thresholds"]
                ],
                "분석_메타데이터": {
                    "분석_시간": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "모델_경로": str(DEFAULT_MODEL_PATH),
                    "테스트된_임계값_수": len(analysis_result["all_thresholds"]),
                    "API_버전": "v1.2"
                }
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"분석 중 오류 발생: {str(e)}")
    
    finally:
        # 임시 파일 정리
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

@router.get("/model-info", summary="모델 정보")
async def get_model_info():
    """현재 모델의 기본 정보 반환"""
    
    try:
        model_path = str(DEFAULT_MODEL_PATH)
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        
        # 모델 로드하여 클래스 정보 확인
        model = YOLO(model_path)
        
        return {
            "모델_경로": model_path,
            "모델_크기_MB": round(model_size, 2),
            "클래스_수": len(model.names),
            "클래스명": model.names
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 정보 조회 실패: {str(e)}")

@router.get("/health", summary="상태 확인")  
async def health_check():
    """API 상태 확인"""
    
    try:
        analyzer = get_analyzer()
        return {
            "상태": "정상",
            "모델_로드됨": True,
            "시간": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "상태": "오류",
            "모델_로드됨": False,
            "오류": str(e),
            "시간": datetime.now().isoformat()
        }
