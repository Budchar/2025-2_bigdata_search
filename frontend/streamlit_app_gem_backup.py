import streamlit as st
import requests
import json
import re
import ast
import os

# ==============================
# 1. 강력한 파싱 및 유틸 함수
# ==============================

def format_rag_answer(text: str) -> str:
    """rag_answer 텍스트 내의 URL을 클릭 가능한 마크다운 링크로 변환"""
    if not isinstance(text, str):
        return str(text)
    
    # URL 패턴 감지 (http/https)
    url_pattern = re.compile(r"(https?://[^\s)]+)")
    def repl(m):
        url = m.group(1)
        return f"[{url}]({url})"
    return url_pattern.sub(repl, text)

def safe_parse_json(text):
    """
    [핵심] 텍스트 뭉치에서 JSON(Dict)을 강제로 추출하는 함수
    """
    # 1. 이미 dict라면 바로 반환
    if isinstance(text, dict):
        return text

    if not isinstance(text, str):
        return None

    # 2. 마크다운 코드 블록 제거 (```json ... ```)
    cleaned = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    # 3. 텍스트 전체에서 가장 바깥쪽 { ... } 찾기 (Greedy Search)
    #    LLM이 사족을 붙여도 JSON 부분만 발라내기 위함
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidate = match.group(0)
    else:
        candidate = cleaned

    # 4. 파싱 시도 (순서 중요: ast -> json -> loose json)
    
    # 시도 1: Python Literal (홑따옴표 ' 처리 가능)
    try:
        return ast.literal_eval(candidate)
    except Exception:
        pass

    # 시도 2: Standard JSON (쌍따옴표 " 필수)
    try:
        return json.loads(candidate)
    except Exception:
        pass

    # 시도 3: 따옴표 보정 후 JSON 시도
    try:
        # 홑따옴표를 쌍따옴표로 치환하되, 텍스트 내부는 건드리지 않도록 주의해야 함.
        # 이 방법은 최후의 수단입니다.
        return json.loads(candidate.replace("'", '"'))
    except Exception:
        pass

    return None

# ==============================
# 2. 논문 결과 카드 UI (Web/Local 통합)
# ==============================

def display_papers(papers):
    """
    논문 리스트를 받아서 깔끔한 카드 형태로 출력
    """
    if not papers:
        return

    st.markdown("### 📚 검색된 논문 목록")
    st.divider()

    for idx, p in enumerate(papers, start=1):
        # 데이터 안전하게 가져오기 (None 방지)
        title = p.get("title") or "제목 없음"
        authors = p.get("authors") or "Unknown"
        year = p.get("published_year") or "Unknown"
        citation = p.get("citation_count", 0)
        
        # URL 처리: url 필드가 없으면 source 필드 사용
        url = p.get("url") or p.get("source") or ""
        
        # 요약 처리: 여러 필드 확인 (snippet, summary_kr, content)
        snippet = p.get("snippet") or p.get("summary_kr") or p.get("content") or ""
        
        # 웹 링크인지 로컬 파일인지 확인
        is_web = str(url).startswith("http")

        # === [UI 렌더링] ===
        
        # 1. 헤더 (제목)
        st.markdown(f"#### {idx}. {title}")

        # 2. 메타 정보 (저자 | 연도 | 인용 | 링크)
        meta_info = f"""
- **👤 저자**: {authors}
- **📅 발간년도**: {year}
- **⭐ 인용수**: {citation}
"""
        st.markdown(meta_info)
        
        # 링크/경로 표시
        if is_web:
            st.markdown(f"- **🔗 링크**: [{url}]({url})")
        elif url:
            st.markdown(f"- **📂 경로**: `{url}`")

        # 3. 내용 요약 (Snippet) - 파란색 박스
        if snippet and snippet.strip():
            st.info(f"**📝 내용 요약/번역**\n\n{snippet}")
        else:
            st.caption("⚠️ 요약 내용이 없습니다.")

        # 4. 로컬 PDF 다운로드 버튼
        if not is_web and url and os.path.isfile(str(url)):
            try:
                with open(url, "rb") as f:
                    st.download_button(
                        label="📥 PDF 다운로드",
                        data=f.read(),
                        file_name=os.path.basename(url),
                        mime="application/pdf",
                        key=f"dn_{idx}_{hash(title)}"
                    )
            except Exception:
                pass
        
        st.divider()

# ==============================
# 3. 메인 앱 실행 로직
# ==============================

BACKEND_QUERY_URL = "http://127.0.0.1:8000/agent/query"

st.set_page_config(page_title="논문 검색 AI", page_icon="📄", layout="wide")
st.title("📄 Agentic RAG 논문 검색 서비스")

st.markdown("""
**사용 가이드**
- **로컬 검색**: 내부 DB에 있는 논문을 우선적으로 찾습니다.
- **웹 검색**: 로컬에 없으면 Google Scholar를 검색하여 결과를 통합합니다.
""")
st.divider()

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": {
                "rag_answer": "안녕하세요! 어떤 논문을 찾아드릴까요?", 
                "related_papers": []
            }
        }
    ]

# --- 채팅 기록 출력 (History) ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        content = msg["content"]
        
        # [중요] 저장된 기록이 문자열이라면 다시 파싱 시도
        if isinstance(content, str):
            parsed = safe_parse_json(content)
            if parsed:
                content = parsed
        
        # Dict로 변환 성공 시 예쁘게 출력
        if isinstance(content, dict):
            rag_answer = content.get("rag_answer", "")
            related_papers = content.get("related_papers", [])
            
            if rag_answer:
                st.markdown(format_rag_answer(rag_answer))
            if related_papers:
                display_papers(related_papers)
        
        # 여전히 텍스트라면 그대로 출력 (Fallback)
        else:
            st.markdown(str(content))

# --- 사용자 입력 처리 ---
user_input = st.chat_input("주제나 키워드를 입력하세요 (예: 자율주행, BERT)...")

if user_input:
    # 1. 사용자 메시지 UI 표시
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. 어시스턴트 로직
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("⏳ 논문을 검색하고 분석 중입니다...")

        try:
            # Backend 호출
            resp = requests.post(BACKEND_QUERY_URL, json={"message": user_input}, timeout=120)
            data = resp.json()
            
            # 응답 추출 (result 키 유무 대응)
            raw_result = data.get("result", data)
            
            # [핵심] 텍스트 뭉치를 JSON 객체로 강력 변환
            final_resp = safe_parse_json(raw_result)
            
            # 파싱 실패 시, 에러 메시지 대신 원본 텍스트라도 보여주기 위한 구조 생성
            if not final_resp:
                final_resp = {
                    "rag_answer": str(raw_result), # 원본 텍스트
                    "related_papers": []
                }

            # 세션에 저장 (나중에 다시 볼 때를 위해)
            st.session_state.messages.append({"role": "assistant", "content": final_resp})

            # 3. 화면 갱신
            placeholder.empty()
            
            # 답변(rag_answer) 출력
            ans = final_resp.get("rag_answer", "")
            st.markdown(format_rag_answer(ans))
            
            # 논문 리스트(related_papers) 출력
            papers = final_resp.get("related_papers", [])
            if papers:
                display_papers(papers)
            elif not ans:
                st.warning("검색 결과가 없습니다.")

        except Exception as e:
            placeholder.error(f"❌ 시스템 오류가 발생했습니다: {e}")