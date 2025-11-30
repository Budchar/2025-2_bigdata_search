#!/usr/bin/env python3
"""
하이브리드 검색 테스트 스크립트

사용법:
    python test/test_search.py                    # 대화형 모드
    python test/test_search.py "검색어"           # 단일 검색
    python test/test_search.py "검색어" --mode vector  # 모드 지정
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.elastic_search import ElasticSearchClient


def print_divider(title: str = ""):
    print()
    if title:
        print(f"{'=' * 20} {title} {'=' * 20}")
    else:
        print("=" * 60)


def run_single_search(client: ElasticSearchClient, query: str, mode: str, top_k: int):
    """단일 검색 실행"""
    print_divider(f"{mode.upper()} 검색")
    print(f"🔍 쿼리: {query}")
    print(f"📊 모드: {mode}, Top-K: {top_k}")
    print("-" * 40)

    result = client.paper_search(query, mode=mode, top_k=top_k)
    print(result)


def run_comparison_search(client: ElasticSearchClient, query: str, top_k: int):
    """세 가지 모드 비교 검색"""
    print_divider("검색 모드 비교")
    print(f"🔍 쿼리: {query}")
    print()

    for mode in ["vector", "bm25", "hybrid"]:
        print(f"\n📌 [{mode.upper()}] 모드")
        print("-" * 40)
        result = client.paper_search(query, mode=mode, top_k=top_k)
        print(result)


def interactive_mode(client: ElasticSearchClient):
    """대화형 검색 모드"""
    print_divider("대화형 검색 모드")
    print("검색어를 입력하세요. 종료하려면 'q' 또는 'quit'을 입력하세요.")
    print()
    print("명령어:")
    print("  [검색어]              - 하이브리드 검색 (기본)")
    print("  v:[검색어]            - 벡터 검색")
    print("  b:[검색어]            - BM25 검색")
    print("  c:[검색어]            - 세 가지 모드 비교")
    print("  help                  - 도움말")
    print("  q, quit               - 종료")
    print()

    while True:
        try:
            user_input = input("🔍 검색> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n종료합니다.")
            break

        if not user_input:
            continue

        if user_input.lower() in ["q", "quit", "exit"]:
            print("종료합니다.")
            break

        if user_input.lower() == "help":
            print("\n명령어:")
            print("  [검색어]              - 하이브리드 검색 (기본)")
            print("  v:[검색어]            - 벡터 검색")
            print("  b:[검색어]            - BM25 검색")
            print("  c:[검색어]            - 세 가지 모드 비교")
            continue

        # 모드 파싱
        if user_input.startswith("v:"):
            query = user_input[2:].strip()
            mode = "vector"
        elif user_input.startswith("b:"):
            query = user_input[2:].strip()
            mode = "bm25"
        elif user_input.startswith("c:"):
            query = user_input[2:].strip()
            run_comparison_search(client, query, top_k=3)
            continue
        else:
            query = user_input
            mode = "hybrid"

        if query:
            run_single_search(client, query, mode, top_k=4)


def main():
    parser = argparse.ArgumentParser(
        description="하이브리드 검색 테스트 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python test_search.py                           # 대화형 모드
  python test_search.py "transformer attention"   # 단일 검색
  python test_search.py "BERT" --mode vector      # 벡터 검색
  python test_search.py "LoRA" --compare          # 모드 비교
        """,
    )
    parser.add_argument("query", nargs="?", help="검색 쿼리 (없으면 대화형 모드)")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["vector", "bm25", "hybrid"],
        default="hybrid",
        help="검색 모드 (기본: hybrid)",
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=4, help="검색 결과 개수 (기본: 4)"
    )
    parser.add_argument(
        "--compare", "-c", action="store_true", help="세 가지 모드 비교 검색"
    )

    args = parser.parse_args()

    # 클라이언트 초기화
    print("🚀 ElasticSearchClient 초기화 중...")
    print("   (임베딩 모델 로딩에 몇 초 걸릴 수 있습니다)")

    try:
        client = ElasticSearchClient()
        print("✅ 초기화 완료!")
    except Exception as e:
        print(f"❌ 초기화 실패: {e}")
        print("   Elasticsearch가 실행 중인지 확인하세요.")
        return

    # 실행 모드 결정
    if args.query:
        if args.compare:
            run_comparison_search(client, args.query, args.top_k)
        else:
            run_single_search(client, args.query, args.mode, args.top_k)
    else:
        interactive_mode(client)


if __name__ == "__main__":
    main()
