from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import re
import ast

from .api import Body
from .agent import ESAgent

# 🚀 [수정] 훨씬 강력한 JSON 파싱 함수
def safe_parse_json(text):
    """LLM의 출력에서 JSON(dict)을 강제로 추출하는 강력한 함수"""
    # 1. 이미 dict라면 바로 반환
    if isinstance(text, dict):
        return text

    if not isinstance(text, str):
        return None

    s = text.strip()
    
    # 2. Markdown 코드 블록 제거 (```json ... ```)
    s = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE).strip()
    s = s.replace("```", "").strip()

    # 3. 텍스트 전체에서 가장 바깥쪽 { ... } 찾기 (Greedy Search)
    #    LLM이 사족을 붙여도 JSON 부분만 발라내기 위함
    match = re.search(r"\{.*\}", s, re.DOTALL)
    if match:
        candidate = match.group(0)
    else:
        candidate = s

    # 4. 파싱 시도 (순서 중요: json -> ast -> loose json)
    
    # 시도 1: Standard JSON
    try:
        return json.loads(candidate)
    except Exception:
        pass

    # 시도 2: Python Literal (홑따옴표 ' 처리 가능)
    try:
        return ast.literal_eval(candidate)
    except Exception:
        pass

    # 시도 3: 제어 문자 제거 후 재시도
    try:
        clean_s = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', candidate)
        return json.loads(clean_s)
    except Exception:
        pass

    return None

def _papers_to_related(papers):
    """Google Scholar 결과를 UI 포맷으로 변환"""
    related = []
    for p in papers:
        related.append(
            {
                "title": p.get("title", "제목 없음"),
                "authors": p.get("authors", "Unknown"),
                "published_year": p.get("published_year", "Unknown"),
                "citation_count": int(p.get("citation_count") or 0),
                "url": p.get("url") or p.get("source", ""),
                "snippet": p.get("snippet", "요약 없음"), # snippet이 비었을 때 대비
                "source": p.get("source", ""),
            }
        )
    return related

def normalize_agent_result(raw):
    """
    에이전트의 원시 응답(raw)을
    항상 { "rag_answer": str, "related_papers": [ ... ] } 형태로 변환.
    """
    
    # Output 껍데기 벗기기
    if isinstance(raw, dict) and "output" in raw:
        raw = raw["output"]

    # 1. 파싱 시도
    parsed = safe_parse_json(raw)

    # 2. 파싱 성공 시 (Dict)
    if isinstance(parsed, dict):
        # Case A: 완벽한 구조
        if "rag_answer" in parsed and "related_papers" in parsed:
            return parsed

        # Case B: Google Scholar 원본 포맷인 경우 변환
        if "papers" in parsed:
            return {
                "rag_answer": "웹 검색을 통해 최신 논문들을 찾았습니다.",
                "related_papers": _papers_to_related(parsed["papers"]),
            }

    # 3. 파싱 실패 시 (String) -> 에러 방지를 위해 빈 리스트 반환
    #    사용자 화면에 JSON 문자열이 그대로 노출되는 것을 막기 위함
    return {
        "rag_answer": str(raw), # 어쩔 수 없이 텍스트로 보여줌
        "related_papers": [],
    }

# ==============================
# FastAPI 앱 초기화
# ==============================

es_agnet = ESAgent()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/agent/query")
async def read_item(body: Body):
    # LangChain invoke
    raw = es_agnet.agent_chain.invoke({"input": body.message})
    
    # 결과 정규화 (JSON 파싱 포함)
    normalized = normalize_agent_result(raw)
    
    return {"result": normalized}
