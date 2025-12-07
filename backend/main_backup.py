from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import re
import ast

from .api import Body
from .agent import ESAgent


def safe_parse_json(text):
    """LLM이 만든 문자열에서 최대한 JSON(dict)을 뽑아오는 강화된 헬퍼."""
    if isinstance(text, dict):
        return text

    if not isinstance(text, str):
        return None

    s = text.strip()
    
    # 1. Markdown 코드 블록 제거 (```json ... ```)
    s = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE).strip()
    s = s.replace("```", "").strip()

    # 2. 가장 바깥쪽 중괄호 찾기
    m = re.search(r"\{.*\}", s, re.DOTALL)
    if m:
        s = m.group(0)

    # 3. 파싱 시도 (json.loads -> ast.literal_eval -> strict=False)
    try:
        return json.loads(s)
    except Exception:
        pass

    try:
        return ast.literal_eval(s)
    except Exception:
        pass

    # 4. [추가] LLM이 흔히 저지르는 실수 보정 (줄바꿈 문자 등)
    try:
        # 제어 문자 제거 후 재시도
        clean_s = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', s)
        return json.loads(clean_s)
    except Exception:
        pass

    return None


def _papers_to_related(papers):
    """{"papers":[...]} 형식을 front가 쓰는 related_papers 형식으로 변환."""
    related = []
    for p in papers:
        related.append(
            {
                "title": p.get("title", "제목 없음"),
                "authors": p.get("authors", "Unknown"),
                "published_year": p.get("published_year", "Unknown"),
                "citation_count": int(p.get("citation_count") or 0),
                "url": p.get("url") or p.get("source", ""),
                "snippet": p.get("snippet", ""),
                "source": p.get("source", ""),
            }
        )
    return related


def normalize_agent_result(raw):
    """
    에이전트의 원시 응답(raw)을
    항상 { "rag_answer": str, "related_papers": [ ... ] } 형태로 변환.
    """

    # 0. AgentExecutor가 {"input":..., "output": ...} 형식으로 줄 때 output만 뽑기
    if isinstance(raw, dict) and "output" in raw:
        val = raw["output"]
        # LangChain 버전에 따라 output 안에 output이 한 번 더 중첩될 수도 있어서 while로 풀어줌
        while isinstance(val, dict) and "output" in val and len(val) == 1:
            val = val["output"]
        raw = val

    # 1. 이미 dict 형태인 경우
    if isinstance(raw, dict):
        # (1) 우리가 원하는 최종 스키마면 그대로
        if "rag_answer" in raw and "related_papers" in raw:
            return raw

        # (2) google_scholar_search 원본: {"papers":[...]}
        if "papers" in raw:
            related = _papers_to_related(raw.get("papers", []))
            return {
                "rag_answer": "웹 검색을 통해 관련 논문들을 찾았습니다.",
                "related_papers": related,
            }

    # 2. 여기서부터는 문자열로 취급
    raw_str = raw if isinstance(raw, str) else str(raw)

    parsed = safe_parse_json(raw_str)

    # 문자열 안에서 JSON을 잘 꺼냈다면 다시 처리
    if isinstance(parsed, dict):
        # (1) 최종 스키마
        if "rag_answer" in parsed and "related_papers" in parsed:
            return parsed

        # (2) {"papers":[...]} 형식
        if "papers" in parsed:
            related = _papers_to_related(parsed.get("papers", []))
            return {
                "rag_answer": "웹 검색을 통해 관련 논문들을 찾았습니다.",
                "related_papers": related,
            }

    # 3. 그래도 안 되면 그냥 문자열 전체를 rag_answer로 보여주기
    return {
        "rag_answer": raw_str,
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
    """에이전트 응답을 항상 dict(JSON) 형태로 정규화해서 반환하는 엔드포인트."""
    raw = es_agnet.agent_chain.invoke({"input": body.message})
    normalized = normalize_agent_result(raw)
    return {"result": normalized}
