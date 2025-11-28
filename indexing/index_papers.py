#!/usr/bin/env python3
"""
[무료 버전] 논문 PDF를 청킹 및 로컬 임베딩(HuggingFace)하여 Elasticsearch에 저장하는 스크립트
- 하이브리드 검색(벡터 + BM25)을 지원하는 스키마로 인덱싱
"""

import argparse
import sys
from pathlib import Path
from typing import List

from elasticsearch import Elasticsearch

# LangChain 관련 임포트
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_elasticsearch import ElasticsearchStore

# 💡 변경점: OpenAIEmbeddings 대신 HuggingFaceEmbeddings 사용
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class LocalRagIndexer:
    # 임베딩 모델의 출력 차원 (jhgan/ko-sbert-multitask는 768차원)
    EMBEDDING_DIM = 768

    def __init__(self, es_url: str, index_name: str, device: str = "cpu"):
        """
        초기화 및 설정

        Args:
            device: 'cpu' 또는 'cuda' (GPU가 있으면 'cuda' 권장)
        """
        self.es_url = es_url
        self.index_name = index_name

        print(
            "📥 임베딩 모델 로드 중... (처음 실행 시 다운로드에 시간이 걸릴 수 있습니다)"
        )

        # 💡 모델 선택 가이드:
        # 1. 영어 논문 위주라면: "sentence-transformers/all-MiniLM-L6-v2" (가볍고 빠름)
        # 2. 한글 논문 위주라면: "jhgan/ko-sbert-multitask" (한국어 성능 좋음)
        # 3. 다국어(한/영 혼용): "intfloat/multilingual-e5-small"

        model_name = "jhgan/ko-sbert-multitask"  # 한국어/영어 논문용 추천 모델

        model_kwargs = {"device": device}
        encode_kwargs = {
            "normalize_embeddings": True
        }  # 코사인 유사도 계산을 위해 정규화

        try:
            self.embedding = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
            print(f"✅ 로컬 모델({model_name}) 로드 완료!")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            sys.exit(1)

        # Elasticsearch 클라이언트 (인덱스 설정용)
        self.es_client = Elasticsearch(self.es_url)

        # 하이브리드 검색을 위한 인덱스 매핑 설정
        self._setup_hybrid_index()

        # Elasticsearch 연결
        self.vector_store = ElasticsearchStore(
            es_url=self.es_url,
            index_name=self.index_name,
            embedding=self.embedding,
        )

    def _setup_hybrid_index(self):
        """
        하이브리드 검색(벡터 + BM25)을 위한 인덱스 매핑 설정
        - text 필드: BM25 키워드 검색용 (한국어/영어 분석기 적용)
        - vector 필드: kNN 벡터 검색용
        """
        # 인덱스가 이미 존재하면 삭제하고 새로 생성 (개발 환경용)
        if self.es_client.indices.exists(index=self.index_name):
            print(f"⚠️  기존 인덱스 '{self.index_name}' 발견. 삭제 후 재생성합니다.")
            self.es_client.indices.delete(index=self.index_name)

        # 하이브리드 검색을 위한 인덱스 매핑
        index_settings = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "korean_english": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "stop"],
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    # 벡터 검색용 필드 (LangChain ElasticsearchStore 기본 필드명)
                    "vector": {
                        "type": "dense_vector",
                        "dims": self.EMBEDDING_DIM,
                        "index": True,
                        "similarity": "cosine",  # 코사인 유사도 사용
                    },
                    # BM25 키워드 검색용 텍스트 필드
                    "text": {
                        "type": "text",
                        "analyzer": "korean_english",
                        "fields": {"keyword": {"type": "keyword", "ignore_above": 256}},
                    },
                    # 메타데이터 필드들
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "keyword"},
                            "page": {"type": "integer"},
                            "chunk_id": {"type": "keyword"},
                        },
                    },
                }
            },
        }

        # 인덱스 생성
        self.es_client.indices.create(index=self.index_name, body=index_settings)
        print(f"✅ 하이브리드 검색용 인덱스 '{self.index_name}' 생성 완료!")

    def load_documents(self, path: Path, recursive: bool = False) -> List[Document]:
        """PDF 파일 로드"""
        print(f"📂 문서 로딩 중... 경로: {path}")

        if path.is_file():
            loader = PyPDFLoader(str(path))
            docs = loader.load()
        else:
            loader = DirectoryLoader(
                str(path),
                glob="**/*.pdf" if recursive else "*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True,
            )
            docs = loader.load()

        print(f"✅ 로딩 완료: 총 {len(docs)} 페이지")
        return docs

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """문서 청킹"""
        print("✂️  문서 청킹(Splitting) 진행 중...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,  # 한글/영어 혼용 시 500~700 정도가 적당
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        splits = text_splitter.split_documents(docs)
        print(f"✅ 청킹 완료: {len(splits)} 개의 청크 생성됨")
        return splits

    def index_documents(self, splits: List[Document]):
        """벡터 인덱싱 수행"""
        print(f"🚀 Elasticsearch({self.es_url})에 벡터 인덱싱 시작...")
        print("⏳ 로컬 CPU/GPU로 변환하므로 문서 양에 따라 시간이 걸릴 수 있습니다.")

        try:
            # 배치 사이즈를 조절하여 메모리 부족 방지 (한 번에 32개씩 처리)
            batch_size = 32
            total_splits = len(splits)

            for i in range(0, total_splits, batch_size):
                batch = splits[i : i + batch_size]
                self.vector_store.add_documents(batch)
                print(
                    f"   ... 진행률: {min(i + batch_size, total_splits)} / {total_splits} 완료"
                )

            print("✨ 모든 문서 인덱싱 완료!")

        except Exception as e:
            print(f"❌ 인덱싱 중 오류 발생: {e}")


def main():
    parser = argparse.ArgumentParser(description="[무료] RAG용 PDF 벡터 인덱싱 도구")
    parser.add_argument("path", type=str, help="PDF 파일 또는 디렉토리 경로")
    parser.add_argument("--host", type=str, default="localhost", help="ES 호스트")
    parser.add_argument("--port", type=int, default=9200, help="ES 포트")
    parser.add_argument(
        "--index", type=str, default="papers-rag-local", help="인덱스 이름"
    )
    parser.add_argument("--recursive", action="store_true", help="하위 디렉토리 포함")
    # GPU 사용 여부 옵션 추가
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="실행 디바이스 (cpu/cuda)",
    )

    args = parser.parse_args()

    es_url = f"http://{args.host}:{args.port}"

    # 1. 초기화
    indexer = LocalRagIndexer(es_url, args.index, device=args.device)

    # 2. 로드
    target_path = Path(args.path)
    if not target_path.exists():
        print("❌ 경로를 찾을 수 없습니다.")
        sys.exit(1)

    raw_docs = indexer.load_documents(target_path, args.recursive)
    if not raw_docs:
        print("⚠️ 처리할 문서가 없습니다.")
        sys.exit(0)

    # 3. 청킹
    chunks = indexer.split_documents(raw_docs)

    # 4. 인덱싱
    indexer.index_documents(chunks)


if __name__ == "__main__":
    main()
