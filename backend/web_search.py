# backend/web_search.py
import os
from dotenv import load_dotenv
from langchain_community.utilities import SerpAPIWrapper

load_dotenv()

class WebSearchClient:
    def __init__(self):
        # Google Scholar 엔진 사용 설정
        self.search = SerpAPIWrapper(
            serpapi_api_key=os.getenv("SERPAPI_API_KEY"),
            params={
                "engine": "google_scholar",
                "hl": "ko",  # 한국어 결과 우선
                "gl": "kr"   # 한국 지역 설정
            }
        )

    def google_scholar_search(self, query: str):
        """
        Performs a Google Scholar search.
        """
        try:
            # 검색 실행
            result = self.search.run(query)
            return result
        except Exception as e:
            return f"Error fetching results from Google Scholar: {e}"