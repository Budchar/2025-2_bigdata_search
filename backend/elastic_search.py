from datetime import datetime

from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_elasticsearch import ElasticsearchStore

from elasticsearch import Elasticsearch

load_dotenv()


class ElasticSearchClient:
    def __init__(self):
        # 1. 인덱싱 코드와 동일한 로컬 임베딩 모델 설정
        # (Indexing 폴더의 index_papers.py 설정과 일치시킴)
        model_name = "jhgan/ko-sbert-multitask"
        model_kwargs = {'device': 'cpu'}  # GPU가 있다면 'cuda'로 변경
        encode_kwargs = {'normalize_embeddings': True}

        self.embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        # 2. Elasticsearch 연결 정보 (환경변수 또는 기본값)
        es_url = os.environ.get('ELASTIC_ENDPOINT', 'http://localhost:9200')

        # 3. LangChain ElasticsearchStore 연결
        # (index_papers.py에서 생성한 'papers-rag-local' 인덱스 사용)
        try:
            self.vector_store = ElasticsearchStore(
                es_url=es_url,
                index_name="papers-rag-local",
                embedding=self.embedding,
                # 클라우드 사용 시 아래 인증 정보 추가 필요
                # es_cloud_id=os.environ.get('ELASTIC_CLOUD_ID'),
                # es_api_key=(os.environ.get('ELASTIC_API_ID'), os.environ.get("ELASTIC_API_KEY"))
            )
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        except Exception as e:
            print(f"Warning: Failed to connect to Elasticsearch: {e}")
            self.vector_store = None

    def site_search(self, query):
        response = self.es_client.search(
                index="site",
                knn={
                    "field": "information_embedding.predicted_value",
                    "query_vector_builder": {
                        "text_embedding": {
                            "model_id": self.es_model_id,
                            "model_text": f"query: {query}"
                        }
                    },
                    "k": 5,
                    "num_candidates": 20,
                }
        )

        formatted_results = []
        for hit in response["hits"]["hits"]:
            result = {
                "score": hit["_score"],
                "관광지명": hit["_source"]["name"],
                "주소": hit["_source"]["address"],
                "관광지설명": hit["_source"]["information"]
            }
            formatted_results.append(result)

        return formatted_results

    def paper_search(self, query: str):
        """
        로컬 ES에 저장된 논문 내용을 검색합니다.
        """
        if not self.vector_store:
            return "Elasticsearch client is not initialized."

        try:
            # LangChain 리트리버를 사용하여 유사도 검색
            docs = self.retriever.invoke(query)

            # 검색 결과를 문자열로 포맷팅
            results = []
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'Unknown')
                content = doc.page_content
                results.append(f"[Source: {source}, Page: {page}]\nContent: {content}\n")

            return "\n---\n".join(results) if results else "No relevant papers found in local DB."

        except Exception as e:
            return f"Error searching papers: {e}"
