# 빅데이터 검색 프로젝트

## 환경 준비

### 1. .env 추가
전달 드린 .env.txt를 .env로 이름 바꿔서 최상위 디렉토리에 저장

### 2. Python requirements 설치 (3.12 기준)
```bash
pip install -r requirements.txt
```

### 3. Elasticsearch 인덱싱
```bash
cd indexing && docker-compose up -d
python index_papers.py papers
```

## 실행 방법

### 프론트엔드
```bash
cd frontend && streamlit run streamlit_app.py
```

### 백엔드
```bash
cd backend && uvicorn main:app --reload
```

### 검색 테스트
```bash
python test/test_search.py              # 대화형 모드
python test/test_search.py "BERT"       # 단일 검색
python test/test_search.py "LoRA" -c    # 모드 비교
```

---

## 📚 하이브리드 검색 사용법 (`backend/elastic_search.py`)

### 검색 모드
| 모드 | 설명 | 사용 케이스 |
|------|------|-------------|
| `hybrid` | 벡터 + BM25 결합 (기본값) | 일반적인 검색, 품질 중요 |
| `vector` | 벡터 유사도 검색 | 의미/개념 기반 검색 |
| `bm25` | 키워드 매칭 검색 | 정확한 용어 검색 |

### 기본 사용법
```python
from backend.elastic_search import ElasticSearchClient

client = ElasticSearchClient()

# 하이브리드 검색 (기본)
result = client.paper_search("transformer attention")

# 모드 지정
result = client.paper_search("BERT", mode="vector")
result = client.paper_search("BERT", mode="bm25")

# 결과 개수 조절
result = client.paper_search("LoRA", top_k=10)

# SearchResult 객체로 받기
results = client.paper_search_with_results("Gemini")
for r in results:
    print(f"{r.source} p.{r.page}: {r.content[:50]}...")
```

### Agent/Tool에서 사용
```python
from backend.elastic_search import ElasticSearchClient

es_client = ElasticSearchClient()

def search_papers(query: str) -> str:
    """논문 DB에서 검색"""
    return es_client.paper_search(query, mode="hybrid", top_k=4)
```

---

## prompt 설명
- `system_message`: 에이전트의 페르소나, 핵심 행동 강령, 출력 형식을 정의하는 최상위 프롬프트
- `db_search_tool_description`: 로컬 Vector DB(ElasticSearch)에 저장된 핵심 논문(BERT, Transformer, LoRA, Gemini) 전용 검색 도구 정의서
- `web_search_tool_description`: Google Scholar를 이용한 외부 학술 자료 검색 도구 정의서
- `multiturn_memory`: 멀티턴 대화에서 문맥을 유지하기 위한 메모리 관리 프롬프트