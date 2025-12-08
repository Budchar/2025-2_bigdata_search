import os
import json
import re
from dotenv import load_dotenv
from langchain_community.utilities import SerpAPIWrapper

load_dotenv()


class WebSearchClient:
    def __init__(self):
        # hl="en", gl="us" 설정을 통해 데이터 포맷을 통일
        self.search = SerpAPIWrapper(
            serpapi_api_key=os.getenv("SERPAPI_API_KEY"),
            params={
                "engine": "google_scholar",
                "hl": "en",
                "gl": "us",
                "num": 5,
            },
        )

    def google_scholar_search(self, query: str) -> str:
        """
        Google Scholar에서 query 관련 논문을 검색해 papers 리스트(JSON 문자열)로 반환.
        실패 시에는 rag_answer/related_papers 형태의 JSON 문자열을 반환한다.
        """
        try:
            raw = self.search.results(query)
        except Exception as e:
            msg = f"Google Scholar 검색 중 오류가 발생했습니다: {e}"
            return json.dumps(
                {
                    "rag_answer": msg,
                    "related_papers": [],
                },
                ensure_ascii=False,
            )

        organic = raw.get("organic_results", [])
        papers = []

        for item in organic:
            title = item.get("title")
            if not title:
                continue

            link = item.get("link") or ""
            snippet = item.get("snippet") or ""

            # 메타데이터 추출
            pub_info = item.get("publication_info") or {}
            authors = "Unknown"
            source = ""  # 기본값을 빈 문자열로 둠 (나중에 링크로 채우기 위함)
            year = None

            # 내부 함수: 텍스트에서 연도(1900~2099) 추출
            def find_year(text):
                if not text:
                    return None
                m = re.search(r"\b(19|20)\d{2}\b", str(text))
                return int(m.group(0)) if m else None

            # 1. 구조화된 데이터(dict) 파싱
            if isinstance(pub_info, dict):
                # 저자
                auth_list = pub_info.get("authors") or []
                if auth_list:
                    temp_authors = []
                    for a in auth_list:
                        if isinstance(a, dict):
                            temp_authors.append(a.get("name", ""))
                        elif isinstance(a, str):
                            temp_authors.append(a)
                    authors = ", ".join(temp_authors)

                # 연도 1차: 명시적 필드 확인
                year = pub_info.get("year")

                # 연도 2차: summary 텍스트 안에서 찾기 ("IEEE access, 2020" 등)
                if not year:
                    summary_text = pub_info.get("summary", "")
                    year = find_year(summary_text)

                # 출처(저널명)
                source = pub_info.get("journal")

            # 2. 문자열 데이터 파싱 (Fallback)
            else:
                text = str(pub_info)
                # 연도 추출
                year = find_year(text)

                # 저자 추출 (연도 앞부분, "-" 기준 앞부분 등)
                if year:
                    before_year = text.split(str(year))[0]
                    if "-" in before_year:
                        authors = before_year.split("-")[0].strip()
                    else:
                        authors = before_year.strip()
                else:
                    authors = text.split("-")[0].strip()

            # 연도 3차: 그래도 없으면 snippet에서 찾기
            if not year:
                year = find_year(snippet)

            # 출처가 없으면 '원문 링크'로 대체
            if not source:
                source = link

            # 인용수 파싱
            citations = 0
            inline = item.get("inline_links") or {}
            cited_by = inline.get("cited_by") or {}
            if isinstance(cited_by, dict):
                citations = cited_by.get("total", 0)

            papers.append(
                {
                    "id": item.get("result_id") or link,
                    "title": title,
                    "authors": authors,
                    "source": source,  # 이제 저널명이 없으면 URL이 들어감
                    "published_year": year if year else "Unknown",
                    "citation_count": int(citations),
                    "url": link,
                    "snippet": snippet,
                }
            )

        # main.py 의 normalize_agent_result 가 {"papers":[...]} 를 받아서
        # {rag_answer, related_papers} 로 변환해 준다.
        return json.dumps({"papers": papers}, ensure_ascii=False)
