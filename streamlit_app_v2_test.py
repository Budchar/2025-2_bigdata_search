import streamlit as st
import requests
import time 
import json 
import uuid 

# -----------------------------------------------------------------
# 1. 설정
# -----------------------------------------------------------------

# FastAPI 백엔드 주소 (main.py가 실행되는 곳) - Mock 테스트 중에는 사용하지 않습니다.
BACKEND_BASE_URL = "http://127.0.0.1:8000/agent"
BACKEND_QUERY_URL = f"{BACKEND_BASE_URL}/query" # 검색 및 RAG
BACKEND_TRANSLATE_URL = f"{BACKEND_BASE_URL}/translate_summary" # 요약/번역

# Streamlit 페이지 설정 - 페이지 제목과 아이콘 설정
st.set_page_config(page_title="논문 검색 AI 에이전트", page_icon="📄", layout="wide")
st.title("📄 Agentic RAG 기반 논문 검색 및 분석 도구 (Mock Test Mode)")
st.warning("⚠️ 현재 Mock Test Mode입니다. 실제 백엔드 API를 호출하지 않습니다.") 

# -----------------------------------------------------------------
# 2. 멀티턴(Multi-turn)을 위한 세션 상태 관리
# -----------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "search_mode" not in st.session_state:
    st.session_state.search_mode = "질의응답(RAG)" 
if "translated_summaries" not in st.session_state:
    st.session_state.translated_summaries = {}
    
# --- [Expander 상태 관리를 위한 세션 상태] ---
# 키를 제거했으므로, Expander의 '제목 문자열'을 기준으로 열림 상태를 관리합니다.
if "expander_states" not in st.session_state:
    st.session_state.expander_states = {} # {expander_title: True/False} 형태로 저장


# -----------------------------------------------------------------
# 3. 사이드바 구성 (하이브리드 검색 옵션)
# -----------------------------------------------------------------

def clear_chat_history_on_mode_change():
    """검색 모드가 변경되면 대화 내역을 초기화하는 콜백 함수"""
    if "mode_selector" in st.session_state and st.session_state.mode_selector != st.session_state.search_mode:
        st.session_state.messages = [] 
        st.session_state.translated_summaries = {} 
        st.session_state.expander_states = {} # <-- Expander 상태 초기화
        st.session_state.search_mode = st.session_state.mode_selector

st.sidebar.header("🔍 검색 모드 설정")

selected_mode = st.sidebar.radio(
    "원하는 검색 방식을 선택하세요:",
    ("질의응답(RAG)", "키워드 검색"),
    key="mode_selector",
    on_change=clear_chat_history_on_mode_change
)
st.session_state.search_mode = selected_mode 

st.sidebar.markdown("""
**[하이브리드 검색 방식 안내]**
* **질의응답(RAG):** 질문의 의도를 분석하여 관련 논문을 찾고, 그 내용을 바탕으로 답변을 생성합니다.
* **키워드 검색:** 입력한 키워드를 포함하는 논문 리스트만 반환합니다. 
""")

# -----------------------------------------------------------------
# 4. 논문 요약/번역 요청 핸들러 (Mock 적용)
# -----------------------------------------------------------------

# --- [수정된 부분: paper_id와 expander_title을 모두 받습니다] ---
def request_translation(paper_id, expander_title):
    """
    개별 논문의 ID를 사용하여 백엔드에 요약/번역을 요청하고 결과를 세션에 저장합니다.
    """
    if not paper_id or paper_id in st.session_state.translated_summaries:
        return

    # --- [Mock Data: API 호출 대신 정해진 응답을 사용] ---
    st.toast("⏳ Mock 데이터로 요약/번역 요청을 처리 중입니다...")
    time.sleep(1.0) # 실제 통신 지연 효과 (1초 대기)
    
    mock_translation_result = {
        "summary_kr": f"[{paper_id}] 논문의 핵심 한국어 요약입니다: 이 Agentic RAG 시스템은 LLM과 ElasticSearch를 결합하여 문서 검색의 정확도를 극대화합니다.",
        "summary_en": f"[{paper_id}] The core English summary of the paper: This Agentic RAG system maximizes document retrieval accuracy by combining LLMs with ElasticSearch."
    }
    # -------------------------------------------------------
    
    try:
        # Mock 데이터를 사용
        translation_result = mock_translation_result
        
        st.session_state.translated_summaries[paper_id] = {
            "summary_kr": translation_result.get("summary_kr", "한국어 요약 없음"),
            "summary_en": translation_result.get("summary_en", "영어 요약 없음")
        }
        st.toast(f"✅ [{paper_id}] 논문 요약/번역 완료!")
        
        # --- [추가된 부분: 해당 Expander를 열린 상태로 설정 (Expander 제목 문자열 기준)] ---
        st.session_state.expander_states[expander_title] = True


    except Exception as e:
        st.error(f"❌ 요약/번역 요청 중 오류가 발생했습니다: {e}")
    
    # 버튼 클릭 후 전체 앱을 다시 실행하여 UI를 업데이트합니다.
    st.rerun()


# -----------------------------------------------------------------
# 5. 논문 검색 결과 표시 함수 
# -----------------------------------------------------------------
def display_papers(papers, message_index): 
    """검색된 논문 리스트를 보기 좋게 출력하는 함수"""
    st.subheader(f"✨ 검색 결과: {len(papers)}건의 관련 논문")
    
    if not papers:
        st.info("검색 결과가 없습니다.")
        return

    for i, paper in enumerate(papers):
        paper_id = paper.get('id', f'temp_{i}')
        
        # --- [Expander 제목 생성] ---
        expander_title = f"{i+1}. **{paper.get('title', '제목 없음')}** ({paper.get('authors', '저자 미상')})"
        # Expander 상태가 session_state에 없으면 기본값 False(닫힘)로 설정
        is_expanded = st.session_state.expander_states.get(expander_title, False) 
        
        with st.expander(
            expander_title, # <-- 제목 문자열 사용
            expanded=is_expanded, # <-- 세션 상태에 저장된 값으로 열림/닫힘 상태 설정
            # key=expander_key # key 인수는 제거되었음.
        ):
            # --- 5-1. 기본 초록 표시 ---
            st.markdown(f"**📚 초록 (원문):** {paper.get('summary', '요약 정보 없음')}")
            
            # --- 5-2. 요약/번역 결과 표시 ---
            if paper_id in st.session_state.translated_summaries:
                translated = st.session_state.translated_summaries[paper_id]
                st.markdown("---")
                st.info("✅ 요약/번역 결과")
                st.markdown(f"**🇰🇷 한국어 요약:** {translated['summary_kr']}")
                st.markdown(f"**🇺🇸 영어 원문 요약:** {translated['summary_en']}")
                
            # --- 5-3. 버튼 및 메타 정보 ---
            cols = st.columns([0.2, 0.8])
            
            with cols[0]:
                is_translated = paper_id in st.session_state.translated_summaries
                
                # --- [Critical Fix]: 버튼 키의 고유성을 UUID 전체로 보장합니다. ---
                # paper_id, message_index, i (논문 인덱스)를 모두 포함하고, 
                # 충돌 가능성이 있는 uuid.uuid4()[:8] 대신 전체 UUID를 사용합니다.
                button_key = f"translate_btn_{message_index}_{paper_id}_{i}_{uuid.uuid4()}" 
                
                st.button(
                    "🤖 요약/번역 요청", 
                    key=button_key, # <-- 고유성 확보
                    on_click=request_translation, 
                    args=(paper_id, expander_title), # <-- expander_title을 인수로 추가 전달
                    disabled=is_translated, 
                    help="AI 에이전트에게 이 논문의 요약과 번역을 요청합니다."
                )

            with cols[1]:
                url = paper.get('url')
                if url:
                    st.link_button("🔗 원문 바로가기", url=url, help="논문의 원문 페이지로 이동합니다.")

            st.caption(f"논문 ID: {paper_id}")


# -----------------------------------------------------------------
# 6. 채팅 UI 구성 (메인 화면) (Mock 적용)
# -----------------------------------------------------------------

# (A) 저장된 메시지(st.session_state.messages)를 순회하며 화면에 출력
for msg_idx, message in enumerate(st.session_state.messages): 
    with st.chat_message(message["role"]):
        # --- [추가 수정: 딕셔너리 메시지 처리] ---
        if isinstance(message["content"], dict):
            rag_answer = message["content"].get("rag_answer")
            related_papers = message["content"].get("related_papers")
            
            if rag_answer:
                st.markdown(rag_answer) 
            
            if related_papers:
                # 논문 목록을 출력하며 Expander 상태가 복원됩니다.
                display_papers(related_papers, msg_idx)
            
            if not rag_answer and not related_papers:
                 st.markdown("에이전트로부터 유효한 응답을 받지 못했습니다.")

        else:
            # 일반 텍스트 메시지 (사용자 입력)
            st.markdown(message["content"])

# (B) 사용자 입력 받기
if prompt := st.chat_input("논문에 대해 무엇이든 물어보세요."):
    
    # 1. 사용자의 메시지를 세션과 화면에 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. AI 응답 처리
    # AI 응답이 생성되면 st.rerun()이 발생하므로, 사용자 메시지 출력은 for 루프에서 처리됩니다. 
    # 하지만, 입력 직후에 시각적 피드백을 위해 먼저 출력하겠습니다.
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        
        # --- [Mock Data: API 호출 대신 정해진 응답을 사용] ---
        st.toast(f"⏳ Mock 데이터로 {st.session_state.search_mode} 쿼리를 처리 중입니다...")
        time.sleep(2.0) # 실제 통신 지연 효과 (2초 대기)
        
        # 검색 모드에 따른 Mock 응답 데이터 정의
        if st.session_state.search_mode == "질의응답(RAG)":
            mock_rag_answer = f"""
            안녕하세요! `{prompt}`에 대한 질의응답 결과입니다.
            **Agentic RAG**는 **GPT-4**와 **Elasticsearch**를 결합하여 사용자 질문에 가장 관련성 높은 논문(아래 목록)을 찾아 그 내용을 바탕으로 답변을 생성하는 진보된 기술입니다.

            특히, Custom Prompt Layer와 CoT(Chain-of-Thought)를 사용하여 응답의 품질과 추론 과정을 향상시킵니다.
            아래 관련 논문 목록을 참고하시고, 필요하면 개별 논문의 '요약/번역 요청' 버튼을 사용해보세요.
            """
        else: # 키워드 검색
             mock_rag_answer = f"키워드 검색 모드로 `{prompt}`를 처리했습니다. 아래는 관련성이 높은 논문 목록입니다. RAG 답변은 키워드 검색 모드에서는 제공되지 않습니다."


        ai_full_response = {
            "rag_answer": mock_rag_answer,
            "related_papers": [
                {
                    "id": "A001",
                    "title": "Agentic RAG: A New Paradigm for Grounded Generation using LLMs",
                    "authors": "Kim, Lee, Park (2024)",
                    "summary": "This paper introduces the Agentic Retrieval-Augmented Generation (RAG) framework, leveraging a multi-step planning agent to improve information retrieval accuracy and contextual understanding in large language models.",
                    "url": "https://arxiv.org/abs/2405.001"
                },
                {
                    "id": "B002",
                    "title": "Hybrid Search Strategies in Vector Databases for Scientific Literature",
                    "authors": "Choi, Jang (2023)",
                    "summary": "We explore the combination of keyword-based search (Elasticsearch) and vector-based search (Embedding) to achieve superior recall and precision in academic knowledge retrieval.",
                    "url": "https://doi.org/10.1109/IJCAI.2023.002"
                }
            ]
        }
        # -------------------------------------------------------
        
        # 3. AI의 전체 응답(딕셔너리)을 세션에 저장 (먼저 저장하여 인덱스 확정)
        # 이 시점에 저장이 되어야 다음 Rerun 시 for 루프에서 출력됩니다.
        st.session_state.messages.append({"role": "assistant", "content": ai_full_response})
        
        # 4. LLM 답변을 타이핑 효과로 출력 (이 부분이 채팅 응답이 됩니다)
        ai_answer_text = ai_full_response.get("rag_answer", "논문 검색이 완료되었습니다.")
        response_placeholder = st.empty()
        full_response = ""
        for chunk in ai_answer_text.split():
            full_response += chunk + " "
            time.sleep(0.02) 
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)
        
        # 5. 논문 목록을 현재 응답에 바로 출력합니다.
        if ai_full_response.get("related_papers"):
            current_message_index = len(st.session_state.messages) - 1
            # display_papers(ai_full_response["related_papers"], current_message_index) 
            # display_papers를 여기서 호출하지 않고, st.rerun() 후 for 루프에서 출력되도록 합니다.
            # 이중 출력 방지를 위해 타이핑 출력 후에는 st.rerun()을 호출합니다.
            pass

        st.toast("✅ Mock 데이터 처리 완료!")
    
    # AI 응답이 완전히 세션에 저장된 후, 전체 UI를 다시 그리기 위해 rerun을 호출합니다.
    st.rerun()
        

# -----------------------------------------------------------------
# 7. Mock 테스트 모드 복구 안내
# -----------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Mock 테스트 완료 시**
    
    백엔드 팀과의 연동을 위해 **Mock 코드를 모두 제거**하고
    **`BACKEND_QUERY_URL`**로 요청하는 코드를 다시 활성화해야 합니다.
    """
)