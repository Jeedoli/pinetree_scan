# AI 분석 라우터 - RAG 기반 챗봇
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import pandas as pd
import json
import datetime
from transformers import pipeline
import torch
import sys
from pathlib import Path

# 프로젝트 경로 설정
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from ..rag_system import get_rag_system

router = APIRouter()

# 전역 변수들
text_generator = None
rag_system = None

@router.on_event("startup")
async def initialize_ai_system():
    """AI 시스템 초기화"""
    global text_generator, rag_system
    
    try:
        print("🤖 AI 분석 시스템 초기화 시작...")
        
        # 1. RAG 시스템 로드
        print("📚 RAG 시스템 로딩...")
        rag_system = get_rag_system()
        
        # 2. 간단한 텍스트 생성 파이프라인 (CPU 기반)
        print("🧠 텍스트 생성 모델 로딩...")
        try:
            text_generator = pipeline(
                "text-generation",
                model="gpt2",  # 가벼운 기본 모델
                device=-1,     # CPU 사용
                pad_token_id=50256
            )
            print("✅ GPT-2 모델 로드 완료")
        except Exception as e:
            print(f"⚠️ GPT-2 로드 실패, 기본 분석기 사용: {e}")
            text_generator = None
        
        print("🎯 AI 분석 시스템 초기화 완료!")
        
    except Exception as e:
        print(f"❌ AI 시스템 초기화 실패: {e}")

@router.post("/simple-chat")
async def simple_chat_bot(
    question: str = Form(..., description="질문 (CSV 파일 없이 순수 질문만)")
):
    """
    🤖 Knowledge Base 기반 RAG 챗봇
    
    ✅ 8개 전문 문서 지식 베이스 활용
    - PDF 논문 (무인항공기 딥러닝 탐지)
    - 처리 가이드 (신뢰도별 대응방안)
    - GPS 방제 계획
    - 생태학적 특성 등
    
    🔍 FAISS 의미 검색으로 관련 정보 자동 추출
    🤖 GPT-4o-mini가 지식 기반 답변 생성
    
    질문 예시:
    - "YOLO 모델이 뭐야?"
    - "mAP 70% 이상이면 현업에서 사용 가능해?"
    - "소나무재선충병 확산 패턴은?"
    - "드론 탐지 정확도를 높이는 방법은?"
    """
    
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG 시스템이 초기화되지 않았습니다")
    
    try:
        # 1. 질문에서 관련 지식 검색
        print(f"🔍 질문 분석: {question}")
        relevant_context = ""
        knowledge_sources = 0
        
        try:
            relevant_context = rag_system.get_context_for_question(question)
            relevant_docs = rag_system.simple_search(question, top_k=3)
            knowledge_sources = len(relevant_docs)
            print(f"📖 {knowledge_sources}개 관련 문서 발견")
        except Exception as e:
            print(f"⚠️ 지식 검색 실패: {e}")
            relevant_context = "기본 전문 지식을 활용합니다."
        
        # 2. 순수 질문 기반 응답 생성
        analysis = rag_system.generate_korean_response(
            question=question,
            context=relevant_context,
            data_analysis=""  # CSV 데이터 없음
        )
        
        # 디버깅: 응답 확인
        print(f"🎯 최종 분석 결과 길이: {len(analysis)} 글자")
        print(f"🎯 최종 분석 결과: {analysis[:100]}...")
        
        # 응답이 비어있거나 너무 짧으면 기본 메시지
        if not analysis or len(analysis.strip()) < 20:
            analysis = "질문을 이해했지만 적절한 답변을 생성하지 못했습니다. 다른 방식으로 질문해주세요."
        
        return {
            "success": True,
            "question": question,
            "ai_answer": analysis,
            "knowledge_sources_used": knowledge_sources,
            "context_used": bool(relevant_context),
            "model_info": {
                "type": "Knowledge Base RAG 챗봇",
                "llm_model": "gpt-4o-mini",
                "knowledge_base_docs": len(rag_system.documents) if rag_system else 0,
                "search_method": "FAISS Semantic Search",
                "mode": "지식 기반 답변 (RAG)"
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"챗봇 응답 생성 실패: {str(e)}")

@router.post("/chat-analysis")
async def rag_based_chat_analysis(
    csv_file: UploadFile = File(..., description="탐지 결과 CSV 파일"),
    question: str = Form(..., description="분석 질문"),
    use_rag: bool = Form(default=True, description="RAG 지식 베이스 사용 여부")
):
    """
    🧠 RAG 기반 스마트 분석 챗봇
    
    업로드된 탐지 결과 CSV를 분석하고, 
    전문 지식 베이스를 활용하여 질문에 답변합니다.
    """
    
    try:
        # 1. CSV 데이터 로드 및 분석
        print(f"📊 CSV 데이터 분석 시작...")
        df = pd.read_csv(csv_file.file)
        data_summary = analyze_detection_data(df)
        
        # 2. RAG 기반 관련 지식 검색
        relevant_context = ""
        knowledge_sources = 0
        
        if use_rag and rag_system:
            try:
                print(f"🔍 질문 관련 지식 검색: {question}")
                relevant_context = rag_system.get_context_for_question(question)
                knowledge_sources = len(rag_system.simple_search(question, top_k=3))
                print(f"📖 {knowledge_sources}개 관련 문서 발견")
            except Exception as e:
                print(f"⚠️ RAG 검색 실패: {e}")
                relevant_context = "전문 지식 베이스에서 관련 정보를 찾을 수 없습니다."
        
        # 3. 전문가 수준의 분석 생성
        analysis = generate_expert_analysis(
            question=question,
            data_summary=data_summary,
            relevant_context=relevant_context,
            use_llm=(text_generator is not None)
        )
        
        return {
            "success": True,
            "question": question,
            "ai_analysis": analysis,
            "data_summary": data_summary,
            "knowledge_context": relevant_context if use_rag else "RAG 사용 안함",
            "knowledge_sources_found": knowledge_sources,
            "rag_enabled": use_rag,
            "model_info": {
                "rag_system": "Simple Keyword-based RAG",
                "text_generator": "DialoGPT-medium + GPT-2 백업",
                "korean_support": "한국어 맥락 특화 처리",
                "knowledge_base_docs": len(rag_system.documents) if rag_system else 0,
                "type": "무료 오픈소스 기반"
            },
            "timestamp": datetime.datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI 분석 실패: {str(e)}")

@router.post("/generate-report")
async def generate_comprehensive_report(
    csv_file: UploadFile = File(..., description="탐지 결과 CSV 파일")
):
    """
    📋 종합 분석 리포트 자동 생성
    
    탐지 결과를 기반으로 전문적인 종합 분석 리포트를 생성합니다.
    """
    
    try:
        # CSV 데이터 분석
        df = pd.read_csv(csv_file.file)
        data_summary = analyze_detection_data(df)
        
        # 미리 정의된 분석 질문들
        analysis_questions = [
            "전체적인 피해 규모를 평가해주세요",
            "우선 방제가 필요한 지역은 어디인가요?",
            "신뢰도 분포를 해석하고 권고사항을 제시해주세요",
            "GPS 좌표를 기반으로 확산 패턴을 분석해주세요"
        ]
        
        # 각 질문에 대한 분석 실행
        report_sections = {}
        for question in analysis_questions:
            try:
                relevant_context = ""
                if rag_system:
                    relevant_context = rag_system.get_context_for_question(question)
                
                analysis = generate_expert_analysis(
                    question=question,
                    data_summary=data_summary,
                    relevant_context=relevant_context,
                    use_llm=(text_generator is not None)
                )
                
                # 질문을 섹션 제목으로 변환
                section_title = question.replace("해주세요", "").replace("인가요?", "")
                report_sections[section_title] = analysis
                
            except Exception as e:
                report_sections[question] = f"분석 실패: {str(e)}"
        
        return {
            "success": True,
            "report_sections": report_sections,
            "data_summary": data_summary,
            "generated_at": datetime.datetime.now().isoformat(),
            "report_info": {
                "total_sections": len(report_sections),
                "knowledge_base_used": rag_system is not None,
                "ai_model_used": text_generator is not None
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"리포트 생성 실패: {str(e)}")

def analyze_detection_data(df: pd.DataFrame) -> dict:
    """탐지 데이터 상세 분석"""
    try:
        # 기본 통계
        total_detections = len(df)
        
        # 신뢰도별 분포
        high_conf = len(df[df['confidence'] >= 0.7]) if 'confidence' in df.columns else 0
        medium_conf = len(df[(df['confidence'] >= 0.4) & (df['confidence'] < 0.7)]) if 'confidence' in df.columns else 0
        low_conf = len(df[df['confidence'] < 0.4]) if 'confidence' in df.columns else 0
        avg_confidence = round(df['confidence'].mean(), 3) if 'confidence' in df.columns else 0
        
        # 공간 정보
        files_processed = df['filename'].nunique() if 'filename' in df.columns else 1
        has_gps = all(col in df.columns for col in ['tm_x', 'tm_y'])
        
        # GPS 범위 (있는 경우)
        gps_info = {}
        if has_gps:
            gps_info = {
                "x_range": f"{df['tm_x'].min():.0f} ~ {df['tm_x'].max():.0f}",
                "y_range": f"{df['tm_y'].min():.0f} ~ {df['tm_y'].max():.0f}",
                "coverage_area": calculate_coverage_area(df['tm_x'], df['tm_y'])
            }
        
        return {
            "total_detections": total_detections,
            "confidence_distribution": {
                "high_70_plus": high_conf,
                "medium_40_70": medium_conf, 
                "low_below_40": low_conf,
                "average": avg_confidence
            },
            "spatial_analysis": {
                "files_processed": files_processed,
                "detections_per_file": round(total_detections / files_processed, 1) if files_processed > 0 else 0,
                "has_gps_coordinates": has_gps,
                "gps_info": gps_info
            },
            "risk_assessment": assess_risk_level(high_conf, medium_conf, low_conf, total_detections)
        }
        
    except Exception as e:
        return {"error": f"데이터 분석 실패: {str(e)}"}

def calculate_coverage_area(x_coords, y_coords) -> str:
    """GPS 좌표 기반 커버리지 면적 계산"""
    try:
        if len(x_coords) < 2 or len(y_coords) < 2:
            return "면적 계산 불가"
        
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        area_sqm = x_range * y_range
        area_sqkm = area_sqm / 1_000_000
        
        if area_sqkm >= 1:
            return f"약 {area_sqkm:.1f}km²"
        else:
            return f"약 {area_sqm:.0f}m²"
            
    except Exception:
        return "면적 계산 불가"

def assess_risk_level(high_conf: int, medium_conf: int, low_conf: int, total: int) -> dict:
    """위험도 평가"""
    if total == 0:
        return {"level": "데이터 없음", "description": "탐지 결과 없음"}
    
    high_ratio = high_conf / total
    medium_ratio = medium_conf / total
    
    if high_ratio >= 0.7:
        level = "매우 높음"
        description = "즉시 대규모 방제 작업 필요"
    elif high_ratio >= 0.4:
        level = "높음" 
        description = "신속한 방제 계획 수립 필요"
    elif medium_ratio >= 0.5:
        level = "보통"
        description = "재조사 후 방제 계획 결정"
    else:
        level = "낮음"
        description = "지속적 모니터링 필요"
    
    return {
        "level": level,
        "description": description,
        "high_confidence_ratio": round(high_ratio * 100, 1)
    }

def generate_expert_analysis(question: str, data_summary: dict, relevant_context: str, use_llm: bool = False) -> str:
    """전문가 수준의 분석 생성 - DialoGPT-medium 기반"""
    
    # 데이터 기반 기본 분석
    base_analysis = generate_rule_based_analysis(question, data_summary, relevant_context)
    
    # RAG 시스템의 한국어 응답 생성 사용
    if rag_system and hasattr(rag_system, 'generate_korean_response'):
        try:
            print("🤖 DialoGPT-medium으로 한국어 응답 생성 중...")
            enhanced_analysis = rag_system.generate_korean_response(
                question=question, 
                context=relevant_context, 
                data_analysis=base_analysis
            )
            
            # 기본 분석과 AI 생성 응답 결합
            if enhanced_analysis and len(enhanced_analysis.strip()) > 50:
                return f"{base_analysis}\n\n🤖 **AI 전문가 추가 분석**\n{enhanced_analysis}"
            
        except Exception as e:
            print(f"⚠️ DialoGPT 분석 실패, 기본 분석 사용: {e}")
    
    # LLM 백업 (기존 GPT-2)
    elif use_llm and text_generator:
        try:
            enhanced_analysis = enhance_with_llm(question, base_analysis, text_generator)
            return enhanced_analysis
        except Exception as e:
            print(f"⚠️ 백업 LLM 분석 실패: {e}")
    
    return base_analysis

def generate_rule_based_analysis(question: str, data_summary: dict, relevant_context: str) -> str:
    """규칙 기반 전문가 분석 생성"""
    
    # 질문 유형 분석
    question_lower = question.lower()
    
    analysis_parts = []
    
    # 기본 데이터 해석
    total = data_summary.get("total_detections", 0)
    high_conf = data_summary.get("confidence_distribution", {}).get("high_70_plus", 0)
    avg_conf = data_summary.get("confidence_distribution", {}).get("average", 0)
    risk_level = data_summary.get("risk_assessment", {}).get("level", "알 수 없음")
    
    # 피해 규모 관련 질문
    if any(keyword in question_lower for keyword in ["규모", "평가", "전체", "피해"]):
        if total > 3000:
            scale_assessment = "대규모 피해 지역으로 판단됩니다."
        elif total > 1000:
            scale_assessment = "중규모 피해가 발생한 지역입니다."
        elif total > 100:
            scale_assessment = "소규모 피해가 확인된 지역입니다."
        else:
            scale_assessment = "경미한 피해 수준입니다."
        
        analysis_parts.append(f"""
🎯 **피해 규모 평가**
- 총 {total}개 피해목 탐지: {scale_assessment}
- 평균 신뢰도 {avg_conf}: {'높은 정확도' if avg_conf >= 0.6 else '보통 정확도' if avg_conf >= 0.4 else '정밀 검토 필요'}
- 위험 등급: {risk_level}
""")
    
    # 우선순위/방제 관련 질문  
    if any(keyword in question_lower for keyword in ["우선", "방제", "처리", "대응"]):
        analysis_parts.append(f"""
🚨 **우선 방제 권고사항**
- 고신뢰도 지역 ({high_conf}개): 즉시 방제 필요
- 중신뢰도 지역: 2주 내 재조사 후 방제 결정
- 저신뢰도 지역: 모니터링 강화
- 권장 작업 순서: 고신뢰도 → 클러스터 경계 → 내부 지역
""")
    
    # 신뢰도 관련 질문
    if any(keyword in question_lower for keyword in ["신뢰도", "정확도", "분포"]):
        conf_dist = data_summary.get("confidence_distribution", {})
        analysis_parts.append(f"""
📊 **신뢰도 분포 분석**  
- 고신뢰도 (70%+): {conf_dist.get('high_70_plus', 0)}개
- 중신뢰도 (40-70%): {conf_dist.get('medium_40_70', 0)}개  
- 저신뢰도 (40% 미만): {conf_dist.get('low_below_40', 0)}개
- 평균 신뢰도: {conf_dist.get('average', 0)}

💡 **권고사항**: {'신뢰도가 높아 즉시 방제 가능' if avg_conf >= 0.6 else '재조사를 통한 정확도 향상 필요'}
""")
    
    # GPS/좌표 관련 질문
    if any(keyword in question_lower for keyword in ["gps", "좌표", "위치", "확산", "패턴"]):
        spatial_info = data_summary.get("spatial_analysis", {})
        has_gps = spatial_info.get("has_gps_coordinates", False)
        
        if has_gps:
            gps_info = spatial_info.get("gps_info", {})
            analysis_parts.append(f"""
📍 **GPS 좌표 분석**
- 탐지 범위: X({gps_info.get('x_range', 'N/A')}), Y({gps_info.get('y_range', 'N/A')})
- 커버리지: {gps_info.get('coverage_area', 'N/A')}
- 파일당 평균 탐지: {spatial_info.get('detections_per_file', 0)}개

🗺️ **확산 패턴 권고**: 클러스터 분석을 통한 차단선 설치 및 우선순위 방제 계획 수립
""")
        else:
            analysis_parts.append(f"""
⚠️ **GPS 좌표 정보 부족**
- 현재 픽셀 좌표만 제공됨
- 정확한 현장 방제를 위해 GPS 좌표 변환 필요
- 지리참조 정보(.tfw) 파일 확인 권장
""")
    
    # RAG 컨텍스트 추가
    if relevant_context and "관련 정보를 찾을 수 없습니다" not in relevant_context:
        analysis_parts.append(f"""
📚 **전문 지식 기반 참고사항**
{relevant_context}
""")
    
    # 기본 분석이 없는 경우
    if not analysis_parts:
        analysis_parts.append(f"""
📊 **기본 분석 결과**
- 총 탐지 수: {total}개
- 평균 신뢰도: {avg_conf}
- 위험 등급: {risk_level}

💡 더 구체적인 분석을 위해 질문을 명확히 해주시면 상세한 답변을 제공하겠습니다.
""")
    
    return "\n".join(analysis_parts)

def enhance_with_llm(question: str, base_analysis: str, llm_pipeline) -> str:
    """LLM을 활용한 분석 향상 (선택적)"""
    try:
        prompt = f"""다음 분석을 바탕으로 전문가 수준의 조언을 추가해주세요:

질문: {question}

기본 분석:
{base_analysis}

추가 전문가 조언:"""
        
        response = llm_pipeline(
            prompt,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=llm_pipeline.tokenizer.eos_token_id
        )
        
        full_response = response[0]['generated_text']
        enhanced_part = full_response.replace(prompt, "").strip()
        
        if enhanced_part and len(enhanced_part) > 10:
            return base_analysis + f"\n\n🧠 **AI 추가 분석**\n{enhanced_part}"
        else:
            return base_analysis
            
    except Exception as e:
        print(f"⚠️ LLM 향상 실패: {e}")
        return base_analysis