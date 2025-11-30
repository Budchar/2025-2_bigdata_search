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
- `system_message`: LLM에 기본 적용할 프롬프트 (역할, 어투, 번역 등)
- `db_search_tool_description`: 언제/어떻게 DB(ES)를 이용해 검색할지 설명