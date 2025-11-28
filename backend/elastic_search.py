"""
하이브리드 검색 (벡터 + BM25) 지원 Elasticsearch 클라이언트
=============================================================

논문 검색을 위한 세 가지 검색 모드를 지원합니다:
- vector: 벡터 유사도 검색 (시맨틱 검색) - 의미 기반 검색
- bm25: BM25 키워드 검색 (전문 검색) - 키워드 매칭 검색
- hybrid: 벡터 + BM25 결합 (RRF 알고리즘) - 가장 정확한 검색 (기본값)

사전 요구사항
------------
1. Elasticsearch 실행 중 (기본: http://localhost:9200)
2. 인덱싱 완료 (indexing/index_papers.py 실행)
3. 필요한 패키지: elasticsearch, langchain-elasticsearch, langchain-huggingface

환경변수
-------
- ELASTIC_ENDPOINT: Elasticsearch URL (기본값: http://localhost:9200)

사용 예시
--------
```python
from backend.elastic_search import ElasticSearchClient

# 클라이언트 초기화 (임베딩 모델 로딩에 몇 초 소요)
client = ElasticSearchClient()

# 1. 기본 사용법 (하이브리드 검색, 문자열 반환)
result = client.paper_search("transformer attention mechanism")
print(result)

# 2. 검색 모드 지정
result = client.paper_search("BERT", mode="vector")   # 벡터 검색만
result = client.paper_search("BERT", mode="bm25")     # BM25 검색만
result = client.paper_search("BERT", mode="hybrid")   # 하이브리드 (기본값)

# 3. 검색 결과 개수 조절
result = client.paper_search("LoRA", top_k=10)

# 4. SearchResult 객체로 결과 받기 (프로그래밍 용도)
results = client.paper_search_with_results("Gemini multimodal")
for r in results:
    print(f"Source: {r.source}, Page: {r.page}")
    print(f"Score: {r.score}, Type: {r.search_type}")
    print(f"Content: {r.content[:100]}...")

# 5. 하이브리드 검색 가중치 조절
results = client.hybrid_search(
    query="attention mechanism",
    top_k=5,
    vector_weight=0.7,  # 벡터 검색 비중 높임
    bm25_weight=0.3
)
```

검색 모드 선택 가이드
------------------
| 상황                          | 추천 모드 |
|------------------------------|----------|
| 일반적인 검색                   | hybrid   |
| 정확한 용어/키워드 검색          | bm25     |
| 의미/개념 기반 검색             | vector   |
| 검색 품질이 중요한 경우          | hybrid   |

반환값 형식
---------
paper_search() 반환값 예시:
```
[1] Source: papers/transformer.pdf, Page: 9
    Score: 0.8559 (vector)
    Content: In this work, we presented the Transformer...

---
[2] Source: papers/bert.pdf, Page: 2
    Score: 0.8337 (vector)
    Content: BERT uses a bidirectional Transformer...
```

SearchResult 데이터 클래스:
- content: 검색된 텍스트 내용
- source: 원본 파일 경로 (예: papers/transformer.pdf)
- page: 페이지 번호
- score: 검색 점수
- search_type: 검색 타입 ('vector', 'bm25', 'hybrid')

Agent/Tool에서 사용하기
--------------------
```python
# agent.py에서 tool로 등록하는 예시
from backend.elastic_search import ElasticSearchClient

es_client = ElasticSearchClient()

def search_papers(query: str) -> str:
    \"\"\"논문 데이터베이스에서 관련 내용을 검색합니다.\"\"\"
    return es_client.paper_search(query, mode="hybrid", top_k=4)
```

주의사항
-------
- 첫 초기화 시 임베딩 모델(jhgan/ko-sbert-multitask) 다운로드에 시간이 걸릴 수 있음
- Elasticsearch가 실행 중이어야 함 (docker-compose up -d)
- 인덱싱이 완료되어 있어야 함 (papers-rag-local 인덱스)
"""

import os
from dataclasses import dataclass
from typing import List, Literal

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchStore
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


@dataclass
class SearchResult:
    """검색 결과를 담는 데이터 클래스"""

    content: str
    source: str
    page: int
    score: float
    search_type: str  # 'vector', 'bm25', 'hybrid'


class ElasticSearchClient:
    # 임베딩 모델의 출력 차원 (jhgan/ko-sbert-multitask는 768차원)
    EMBEDDING_DIM = 768

    def __init__(self):
        # 1. 인덱싱 코드와 동일한 로컬 임베딩 모델 설정
        model_name = "jhgan/ko-sbert-multitask"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}

        self.embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

        # 2. Elasticsearch 연결 정보
        self.es_url = os.environ.get("ELASTIC_ENDPOINT", "http://localhost:9200")
        self.index_name = "papers-rag-local"

        # 3. Elasticsearch 네이티브 클라이언트 (하이브리드 검색용)
        try:
            self.es_client = Elasticsearch(self.es_url)

            # LangChain ElasticsearchStore도 유지 (호환성)
            self.vector_store = ElasticsearchStore(
                es_url=self.es_url,
                index_name=self.index_name,
                embedding=self.embedding,
            )
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

        except Exception as e:
            print(f"Warning: Failed to connect to Elasticsearch: {e}")
            self.es_client = None
            self.vector_store = None

    def _get_query_embedding(self, query: str) -> List[float]:
        """쿼리 텍스트를 임베딩 벡터로 변환"""
        return self.embedding.embed_query(query)

    def vector_search(self, query: str, top_k: int = 4) -> List[SearchResult]:
        """
        벡터 유사도 검색 (시맨틱 검색)
        - kNN 알고리즘을 사용하여 의미적으로 유사한 문서 검색
        """
        if not self.es_client:
            return []

        query_vector = self._get_query_embedding(query)

        search_query = {
            "knn": {
                "field": "vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": top_k * 10,  # 후보군을 넉넉하게 설정
            },
            "_source": ["text", "metadata"],
        }

        try:
            response = self.es_client.search(index=self.index_name, body=search_query)
            return self._parse_results(response, "vector")
        except Exception as e:
            print(f"Vector search error: {e}")
            return []

    def bm25_search(self, query: str, top_k: int = 4) -> List[SearchResult]:
        """
        BM25 키워드 검색 (전문 검색)
        - 키워드 매칭 기반의 전통적인 검색
        """
        if not self.es_client:
            return []

        search_query = {
            "query": {"match": {"text": {"query": query, "operator": "or"}}},
            "size": top_k,
            "_source": ["text", "metadata"],
        }

        try:
            response = self.es_client.search(index=self.index_name, body=search_query)
            return self._parse_results(response, "bm25")
        except Exception as e:
            print(f"BM25 search error: {e}")
            return []

    def hybrid_search(
        self,
        query: str,
        top_k: int = 4,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> List[SearchResult]:
        """
        하이브리드 검색 (벡터 + BM25)
        - RRF (Reciprocal Rank Fusion) 알고리즘으로 두 검색 결과를 통합

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            vector_weight: 벡터 검색 가중치 (0~1)
            bm25_weight: BM25 검색 가중치 (0~1)
            rrf_k: RRF 알고리즘의 k 파라미터 (기본값 60)
        """
        if not self.es_client:
            return []

        query_vector = self._get_query_embedding(query)

        # Elasticsearch 8.x의 RRF를 사용한 하이브리드 검색
        search_query = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        # BM25 키워드 검색
                        {"match": {"text": {"query": query, "boost": bm25_weight}}}
                    ]
                }
            },
            "knn": {
                "field": "vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": top_k * 10,
                "boost": vector_weight,
            },
            "_source": ["text", "metadata"],
        }

        try:
            response = self.es_client.search(index=self.index_name, body=search_query)
            return self._parse_results(response, "hybrid")
        except Exception as e:
            print(f"Hybrid search error: {e}")
            # 폴백: 수동으로 RRF 계산
            return self._manual_rrf_hybrid(
                query, top_k, vector_weight, bm25_weight, rrf_k
            )

    def _manual_rrf_hybrid(
        self,
        query: str,
        top_k: int,
        vector_weight: float,
        bm25_weight: float,
        rrf_k: int,
    ) -> List[SearchResult]:
        """
        수동 RRF (Reciprocal Rank Fusion) 하이브리드 검색
        - ES 버전이 낮거나 네이티브 하이브리드가 실패할 경우 사용

        RRF 공식: score = Σ (weight / (k + rank))
        """
        # 각각의 검색 실행
        vector_results = self.vector_search(query, top_k * 2)
        bm25_results = self.bm25_search(query, top_k * 2)

        # 문서 ID (content hash) 기반으로 RRF 점수 계산
        rrf_scores = {}
        doc_map = {}

        # 벡터 검색 결과 RRF 점수 계산
        for rank, result in enumerate(vector_results, start=1):
            doc_id = hash(result.content)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (
                vector_weight / (rrf_k + rank)
            )
            doc_map[doc_id] = result

        # BM25 검색 결과 RRF 점수 계산
        for rank, result in enumerate(bm25_results, start=1):
            doc_id = hash(result.content)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (
                bm25_weight / (rrf_k + rank)
            )
            if doc_id not in doc_map:
                doc_map[doc_id] = result

        # RRF 점수로 정렬
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 결과 생성
        results = []
        for doc_id, score in sorted_docs[:top_k]:
            result = doc_map[doc_id]
            results.append(
                SearchResult(
                    content=result.content,
                    source=result.source,
                    page=result.page,
                    score=score,
                    search_type="hybrid",
                )
            )

        return results

    def _parse_results(self, response: dict, search_type: str) -> List[SearchResult]:
        """Elasticsearch 응답을 SearchResult 리스트로 변환"""
        results = []

        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            metadata = source.get("metadata", {})

            results.append(
                SearchResult(
                    content=source.get("text", ""),
                    source=metadata.get("source", "Unknown"),
                    page=metadata.get("page", 0),
                    score=hit.get("_score", 0.0),
                    search_type=search_type,
                )
            )

        return results

    def paper_search(
        self,
        query: str,
        mode: Literal["vector", "bm25", "hybrid"] = "hybrid",
        top_k: int = 4,
    ) -> str:
        """
        논문 검색 통합 인터페이스

        Args:
            query: 검색 쿼리
            mode: 검색 모드 ('vector', 'bm25', 'hybrid')
            top_k: 반환할 결과 수

        Returns:
            포맷팅된 검색 결과 문자열
        """
        if not self.es_client:
            return "Elasticsearch client is not initialized."

        # 검색 모드에 따라 적절한 메서드 호출
        if mode == "vector":
            results = self.vector_search(query, top_k)
        elif mode == "bm25":
            results = self.bm25_search(query, top_k)
        else:  # hybrid
            results = self.hybrid_search(query, top_k)

        if not results:
            return "No relevant papers found in local DB."

        # 결과 포맷팅
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"[{i}] Source: {result.source}, Page: {result.page}\n"
                f"    Score: {result.score:.4f} ({result.search_type})\n"
                f"    Content: {result.content}\n"
            )

        return "\n---\n".join(formatted_results)

    def paper_search_with_results(
        self,
        query: str,
        mode: Literal["vector", "bm25", "hybrid"] = "hybrid",
        top_k: int = 4,
    ) -> List[SearchResult]:
        """
        논문 검색 (SearchResult 객체 리스트 반환)
        - 프로그래밍 방식으로 결과를 처리할 때 사용
        """
        if not self.es_client:
            return []

        if mode == "vector":
            return self.vector_search(query, top_k)
        elif mode == "bm25":
            return self.bm25_search(query, top_k)
        else:
            return self.hybrid_search(query, top_k)
