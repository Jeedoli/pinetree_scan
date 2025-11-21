# 소나무 피해목 탐지 시스템

드론으로 찍은 항공사진 ***tiff, tfw*** 확장자 파일을 기반으로 소나무재선충병 피해목을 자동으로 찾아 TM좌표로 변환하여 csv로 저장시켜주는 프로젝트입니다.

## 프로젝트 소개

산림청이나 지자체에서 소나무재선충병 피해목을 찾으려면 산을 직접 돌아다니며 일일이 확인해야 합니다.
드론으로 찍은 사진이 있어도 사람이 하나하나 눈으로 확인하려면 시간이 오래 걸립니다.

이 문제를 해결하기 위해 YOLO 객체 탐지 모델을 사용했고, 탐지 결과를 GPS 좌표로 변환해서
현장에서 바로 대응 할 수 있도록 개선해보았습니다.

추가로 Langchain + RAG을 배워보고 활용해보고 싶었기에 현재 프로젝트에 응용하여
소나무재선충병이나 탐지 방법에 대해 궁금한 점을 물어볼 수 있는 AI 챗봇으로 개발해 보았습니다.
LangChain을 활용해 OpenAI API를 쉽게 호출할 수 있게 하였고, RAG(Retrieval-Augmented Generation)를 써서 관련 자료나 논문 등등 데이터를 특정한 디렉토리에 관리해준다면 해당 데이터를 검색해서 답변해줄 수 있게 진행했습니다.

## 주요 기능

### 1. 피해목 자동 탐지
- **YOLOv11s** 모델을 사용한 실시간 객체 탐지
- Confidence Threshold 0.16 적용 (낮게 설정하여 피해목 놓치지 않게)
- IoU 0.45로 중복 탐지 제거
- 배치 처리로 여러 이미지 동시 분석 가능

### 2. GeoTIFF 좌표 변환
- **rasterio**로 GeoTIFF 메타데이터 읽기 (affine transform 정보)
- 픽셀 좌표 → TM 좌표 → GPS 좌표(위도/경도) 3단계 변환
- **pyproj**로 다양한 좌표계 지원 (EPSG:5186, EPSG:4326 등)
- 변환 결과를 CSV로 자동 저장

### 3. 대용량 이미지 전처리
- 큰 항공사진을 512x512 또는 1024x1024 타일로 자동 분할
- 타일 간 오버랩 설정으로 경계 피해목 놓치지 않음
- ZIP 파일로 압축하여 저장 및 다운로드

### 4. 탐지 결과 시각화
- 탐지된 피해목 위치에 바운딩박스 + 신뢰도 표시
- 모든 탐지 결과(타일화 이미지)를 하나의 이미지로 병합 (merged detection)
- 타일로 쪼개진 개별 이미지도 별도 저장

### 5. AI 챗봇 (RAG 시스템)
- OpenAI GPT-4o-mini + LangChain으로 자연스러운 대화
- FAISS 의미 기반 검색으로 관련 문서 자동 검색
- 8개 지식 베이스 문서 활용 (논문, 가이드라인, 통계 데이터 등)
- 질문에 맞는 참고 자료 자동 선택하여 답변 생성

## 시스템 동작 흐름

### 피해목 탐지 플로우

| 단계 | 작업 | 사용 기술 | 입력 | 출력 |
|------|------|-----------|------|------|
| 1 | 이미지 업로드 | FastAPI | `.tif` / `.zip` | 이미지 파일 |
| 2 | 타일 분할 (선택) | OpenCV, NumPy | 대용량 이미지 | 512x512 타일들 |
| 3 | YOLO 추론 | YOLOv11s, PyTorch | 타일 이미지 | 바운딩박스 좌표 |
| 4 | 픽셀 → GPS 변환 | rasterio, pyproj | GeoTIFF + 픽셀좌표 | 위도/경도 |
| 5 | 결과 저장 | pandas | 탐지 결과 | CSV 파일 |
| 6 | 시각화 생성 | OpenCV | 이미지 + 탐지정보 | 바운딩박스 이미지 |

### AI 챗봇 플로우

| 단계 | 작업 | 사용 기술 | 입력 | 출력 |
|------|------|-----------|------|------|
| 1 | 질문 입력 | FastAPI | 사용자 질문 | 텍스트 |
| 2 | 문서 임베딩 | Sentence Transformers | 질문 텍스트 | 384차원 벡터 |
| 3 | 의미 검색 | FAISS | 질문 벡터 | 관련 문서 3개 |
| 4 | 프롬프트 생성 | LangChain ChatPromptTemplate | 질문 + 문서 | 프롬프트 |
| 5 | LLM 호출 | OpenAI GPT-4o-mini | 프롬프트 | AI 답변 |
| 6 | 응답 파싱 | LangChain StrOutputParser | AI 응답 | 정제된 텍스트 |
| 7 | API 응답 | FastAPI | 답변 + 메타데이터 | JSON |

## 사용한 기술

**객체 탐지**
- YOLOv11s (Ultralytics 라이브러리)
- PyTorch

**API 서버**
- FastAPI
- Uvicorn

**좌표 처리**
- rasterio (GeoTIFF 파일 읽기)
- pyproj (픽셀 → GPS 좌표 변환)

**AI 챗봇**
- OpenAI GPT-4o-mini API
- LangChain (프롬프트 관리)
- FAISS (문서 검색)
- Sentence Transformers (임베딩)

**기타**
- pandas, numpy, OpenCV (데이터 처리)
- Poetry (패키지 관리)

## 프로젝트 구조

```
pinetree_scan/
├── api/
│   ├── main.py                    # FastAPI 앱
│   ├── config.py                  # 설정 (conf=0.16, iou=0.45)
│   ├── rag_system.py              # RAG 챗봇 엔진
│   ├── run_server.py              # 서버 실행 스크립트
│   └── routers/
│       ├── inference.py           # 탐지 API
│       ├── preprocessing.py       # 데이터 전처리 API
│       ├── visualization.py       # 시각화 API
│       ├── model_performance.py   # 성능 평가 API
│       └── ai_analysis.py         # AI 챗봇 API
│
├── models/
│   └── colab_yolo/
│       └── best.pt                # 학습된 YOLO 모델
│
├── knowledge_base/                # RAG용 지식 베이스 (추후 다른 논문 및 데이터 추가 예정)
│   ├── detection_guidelines/
│   │   └── 신뢰도별_처리방안.md
│   ├── domain_expertise/
│   │   └── 소나무재선충병_생태학적특성.md
│   ├── forest_management/
│   │   └── GPS좌표_기반_방제계획.md
│   ├── 무인항공기를이용한딥러닝기반의소나무재선충병감염목탐지.pdf
│   ├── 산림관리_매뉴얼.json
│   ├── 산림병해_추가정보.txt
│   ├── 지역별_탐지현황.csv
│   └── 프로젝트_설정.json
│
├── notebooks/
│   └── YOLOv11s_Smart_Training.ipynb  # Colab 학습 노트북
│
├── data/
│   └── api_outputs/               # API 결과 저장
│
├── pyproject.toml                 # Poetry 의존성
├── .env                           # OpenAI API 키 (gitignore됨)
└── README.md
```

## 시작하기

### 1. 저장소 클론

```bash
git clone https://github.com/Jeedoli/pinetree_scan.git
cd pinetree_scan
```

### 2. 의존성 설치

Poetry 사용:
```bash
poetry install
```

또는 pip:
```bash
pip install -r api/requirements.txt
```

### 3. 환경 변수 설정 (AI 챗봇 사용 시)

`.env` 파일 생성:
```bash
OPENAI_API_KEY=sk-proj-your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

### 4. 서버 실행

```bash
poetry run python api/run_server.py
```

서버가 켜지면 http://localhost:8000/docs 에서 Swagger UI로 API를 테스트할 수 있습니다.

처음 실행할 때 knowledge_base의 파일들을 읽어서 임베딩을 만드는데 10초 정도 걸립니다.
한번 만들어지면 `embeddings_cache.pkl` 파일로 캐싱처리되어 다음부턴 좀 더 빨라집니다.

## API 사용법

### 피해목 탐지 API

```bash
curl -X POST "http://localhost:8000/api/v1/inference/detect" \
  -F "images_zip=@forest_images.zip" \
  -F "confidence=0.16" \
  -F "iou_threshold=0.45" \
  -F "save_visualization=true"
```

결과:
- CSV 파일 (탐지 결과 + GPS 좌표)
- 시각화 이미지 (바운딩박스 표시)

### 추론용 타일 분할 API

```bash
curl -X POST "http://localhost:8000/api/v1/preprocessing/tile-for-inference" \
  -F "image_file=@big_forest_map.tif" \
  -F "tile_size=1024"
```

결과: ZIP 파일 (타일 이미지들)

### AI 챗봇 API

```bash
curl -X POST "http://localhost:8000/api/v1/ai-analysis/simple-chat" \
  -d "question=소나무재선충병이란?"
```

결과:
```json
{
  "success": true,
  "ai_answer": "소나무재선충병은 소나무재선충이라는 벌레가 원인이 되어...",
  "knowledge_sources_used": 3,
  "model_info": {
    "type": "Knowledge Base RAG 챗봇",
    "llm_model": "gpt-4o-mini",
    "search_method": "FAISS Semantic Search"
  }
}
```

## AI 챗봇

탐지 시스템만 있으면 실제 사용하는 분들이 YOLO가 뭔지, 신뢰도가 뭔지 모를 수 있어서
질문할 수 있는 챗봇을 추가했습니다.

**작동 방식**
1. 사용자가 질문 입력
2. `knowledge_base/` 폴더의 문서들을 FAISS로 검색 (의미 기반)
3. 관련도 높은 문서 3개 추출
4. 문서 내용과 질문을 OpenAI API에 전송
5. GPT-4o-mini가 답변 생성

**질문 예시**
```
Q: YOLO가 뭐야?
A: You Only Look Once의 약자로, 이미지에서 객체를 실시간으로 탐지하는 AI 기술입니다...

Q: 신뢰도 70% 이상이면?
A: 고신뢰도로 분류되어 즉시 방제 대상이 됩니다. GPS 좌표 기반으로 현장 방제팀이 파견됩니다...
```

**지식 베이스 구성**
- 무인항공기 딥러닝 탐지 논문 (PDF)
- 신뢰도별 처리 가이드 (MD)
- GPS 방제 계획 (MD)
- 소나무재선충병 생태 특성 (MD)
- 지역별 탐지 현황 (CSV)
- 산림 관리 매뉴얼 (JSON)
- 기타 설정 파일들

총 8개 파일이고 PDF에서 텍스트 추출해서 사용합니다.

## 모델 학습

`notebooks/YOLOv11s_Smart_Training.ipynb` 파일을 Google Colab에서 열어서 실행하면 됩니다.
로컬 환경에 GPU가 없어서 Colab의 무료 GPU(T4)를 사용했습니다.

학습된 모델은 `models/colab_yolo/best.pt` 경로에 저장되어 있어야 합니다.

## 설정값

API에서 사용하는 추론 설정은 `api/config.py`에 있습니다:
- confidence: 0.16 (낮게 설정해서 피해목 놓치지 않게)
- iou_threshold: 0.45

처음엔 confidence를 0.5로 했는데 놓치는 게 많아서 0.16으로 낮췄습니다.
대신 오탐이 좀 늘어나긴 했지만 실제 현장에서는 확인 절차가 있으니까 괜찮다고 판단했습니다.

## 개발하면서 겪은 문제들

**1. 메모리 부족**
- 큰 이미지를 통째로 처리하니까 메모리가 부족했습니다
- 타일로 나눠서 처리하는 방식으로 변경했습니다
- `preprocessing.py`에 타일 분할 API를 따로 만들었습니다

**2. OpenAI API 비용**
- GPT-5를 쓰기에는 좀 부담됐었음..
- GPT-4o-mini로 변경했더니 훨씬 저렴해졌습니다 (1회당 $0.001)
- 그래도 충분히 좋은 답변을 만들어줍니다.



## 알아두면 좋은 점

- GeoTIFF 파일이 아닌 일반 이미지는 GPS 좌표 변환이 안 됩니다
- 서버 실행하면 knowledge_base 폴더의 파일들을 한번에 로드하는데 시간이 좀 걸림

## 개선하고 싶은 부분

- [ ] 모델 성능 향상 (더 많은 데이터로 재학습)
- [ ] 배치 크기 조정 옵션 추가
- [ ] 결과를 DB에 저장하는 기능
- [ ] 프론트엔드 (지금은 Swagger UI만 있음)
- [ ] 모델 로드하는데 걸리는 로딩 개선


## 참고한 것들

- Ultralytics YOLOv11 공식 문서
- FastAPI 공식 튜토리얼
- LangChain RAG 가이드
- rasterio 예제 코드들
- 무인항공기를 이용한 딥러닝 기반의 소나무재선충병 감염목 탐지 논문
- LLM을 괴롭히며, 많이 물어본게 제일 도움 되기는 했습니다!
---

만든 사람: Jeedoli
