import streamlit as st
import requests
import time 
import json 
import uuid 

# -----------------------------------------------------------------
# 1. 설정
# -----------------------------------------------------------------

# FastAPI 백엔드 주소 (main.py가 실행되는 곳) - 실제 연동 주소
# 백엔드가 실행되는 환경에 맞게 이 주소를 변경해야 합니다.
BACKEND_BASE_URL = "http://127.0.0.1:8000/agent"
BACKEND_QUERY_URL = f"{BACKEND_BASE_URL}/query" # 검색 및 RAG
BACKEND_TRANSLATE_URL = f"{BACKEND_BASE_URL}/translate_summary" # 요약/번역

# Streamlit 페이지 설정 - 페이지 제목과 아이콘 설정
st.set_page_config(page_title="논문 검색 AI 에이전트", page_icon="📄", layout="wide")
st.title("📄 Agentic RAG 기반 논문 검색 및 분석 도구") # Mock 모드 경고 제거

# -----------------------------------------------------------------
# 2. 멀티턴(Multi-turn)을 위한 세션 상태 관리
# -----------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "search_mode" not in st.session_state:
    st.session_state.search_mode = "질의응답(RAG)" 
if "translated_summaries" not in st.session_state:
    st.session_state.translated_summaries = {}
    
# Expander 상태 관리를 위한 세션 상태
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
        st.session_state.expander_states = {} 
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
# 4. 논문 요약/번역 요청 핸들러 (API 연동)
# -----------------------------------------------------------------

def request_translation(paper_id, expander_title):
    """
    개별 논문의 ID를 사용하여 백엔드에 요약/번역을 요청하고 결과를 세션에 저장합니다.
    """
    if not paper_id or paper_id in st.session_state.translated_summaries:
        return

    st.toast("⏳ 백엔드 API로 요약/번역 요청을 처리 중입니다...")
    
    try:
        # --- [Mock 제거, 실제 API 호출] ---
        response = requests.post(
            BACKEND_TRANSLATE_URL,
            json={"paper_id": paper_id},
            timeout=30 # 30초 타임아웃 설정
        )
        response.raise_for_status() # HTTP 오류 발생 시 예외 발생
        
        translation_result = response.json()
        # ---------------------------------
        
        st.session_state.translated_summaries[paper_id] = {
            "summary_kr": translation_result.get("summary_kr", "한국어 요약 없음"),
            "summary_en": translation_result.get("summary_en", "영어 요약 없음")
        }
        st.toast(f"✅ [{paper_id}] 논문 요약/번역 완료!")
        
        st.session_state.expander_states[expander_title] = True

    except requests.exceptions.Timeout:
        st.error("❌ 요약/번역 요청 시간이 초과되었습니다. 백엔드 상태를 확인하세요.")
    except requests.exceptions.RequestException as e:
        st.error(f"❌ 요약/번역 요청 중 통신 오류가 발생했습니다: {e}")
    except json.JSONDecodeError:
        st.error("❌ 백엔드로부터 유효한 JSON 응답을 받지 못했습니다.")
    except Exception as e:
        st.error(f"❌ 알 수 없는 오류가 발생했습니다: {e}")
    
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
        
        # Expander 제목 생성
        expander_title = f"{i+1}. **{paper.get('title', '제목 없음')}** ({paper.get('authors', '저자 미상')})"
        # Expander 상태가 session_state에 없으면 기본값 False(닫힘)로 설정
        is_expanded = st.session_state.expander_states.get(expander_title, False) 
        
        with st.expander(
            expander_title, 
            expanded=is_expanded, 
        ):
            # 5-1. 기본 초록 표시
            st.markdown(f"**📚 초록 (원문):** {paper.get('summary', '요약 정보 없음')}")
            
            # 5-2. 요약/번역 결과 표시
            if paper_id in st.session_state.translated_summaries:
                translated = st.session_state.translated_summaries[paper_id]
                st.markdown("---")
                st.info("✅ 요약/번역 결과")
                st.markdown(f"**🇰🇷 한국어 요약:** {translated['summary_kr']}")
                st.markdown(f"**🇺🇸 영어 원문 요약:** {translated['summary_en']}")
                
            # 5-3. 버튼 및 메타 정보
            cols = st.columns([0.2, 0.8])
            
            with cols[0]:
                is_translated = paper_id in st.session_state.translated_summaries
                
                # 버튼 키의 고유성을 UUID 전체로 보장합니다.
                button_key = f"translate_btn_{message_index}_{paper_id}_{i}_{uuid.uuid4()}" 
                
                st.button(
                    "🤖 요약/번역 요청", 
                    key=button_key, 
                    on_click=request_translation, 
                    args=(paper_id, expander_title), 
                    disabled=is_translated, 
                    help="AI 에이전트에게 이 논문의 요약과 번역을 요청합니다."
                )

            with cols[1]:
                url = paper.get('url')
                if url:
                    st.link_button("🔗 원문 바로가기", url=url, help="논문의 원문 페이지로 이동합니다.")

            st.caption(f"논문 ID: {paper_id}")


# -----------------------------------------------------------------
# 6. 채팅 UI 구성 (메인 화면) (API 연동)
# -----------------------------------------------------------------

# (A) 저장된 메시지(st.session_state.messages)를 순회하며 화면에 출력
for msg_idx, message in enumerate(st.session_state.messages): 
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict):
            rag_answer = message["content"].get("rag_answer")
            related_papers = message["content"].get("related_papers")
            
            if rag_answer:
                st.markdown(rag_answer) 
            
            if related_papers:
                display_papers(related_papers, msg_idx)
            
            if not rag_answer and not related_papers:
                 st.markdown("에이전트로부터 유효한 응답을 받지 못했습니다.")

        else:
            st.markdown(message["content"])

# (B) 사용자 입력 받기
if prompt := st.chat_input("논문에 대해 무엇이든 물어보세요."):
    
    # 1. 사용자의 메시지를 세션에 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. 사용자 메시지를 화면에 출력 (즉각적인 피드백)
    with st.chat_message("user"):
        st.markdown(prompt)

    # 3. AI 응답 처리
    with st.chat_message("assistant"):
        st.toast(f"⏳ 백엔드 API로 {st.session_state.search_mode} 쿼리를 처리 중입니다...")
        
        response_placeholder = st.empty()
        response_placeholder.markdown("처리 중...") # API 호출 전 로딩 메시지
        
        try:
            # --- [Mock 제거, 실제 API 호출] ---
            payload = {
                "query": prompt, 
                "mode": st.session_state.search_mode
            }
            response = requests.post(
                BACKEND_QUERY_URL,
                json=payload,
                timeout=60 # RAG는 시간이 걸릴 수 있으므로 60초 타임아웃 설정
            )
            response.raise_for_status() 
            ai_full_response = response.json()
            # ---------------------------------
            
            # 4. AI의 전체 응답(딕셔너리)을 세션에 저장
            st.session_state.messages.append({"role": "assistant", "content": ai_full_response})

            # 5. LLM 답변을 타이핑 효과로 출력
            ai_answer_text = ai_full_response.get("rag_answer", "논문 검색이 완료되었습니다.")
            full_response = ""
            # 실제 API 응답은 딜레이가 이미 포함되어 있으므로, 딜레이를 줄이거나 제거합니다.
            for chunk in ai_answer_text.split():
                full_response += chunk + " "
                time.sleep(0.01) # 아주 짧은 딜레이만 적용
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
            
            st.toast("✅ 응답 수신 완료!")

        except requests.exceptions.Timeout:
            error_message = "❌ 검색 요청 시간이 초과되었습니다. 백엔드(LLM/RAG) 상태를 확인하세요."
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": {"rag_answer": error_message}})
            response_placeholder.markdown(error_message)
        except requests.exceptions.RequestException as e:
            error_message = f"❌ 검색 요청 중 통신 오류가 발생했습니다: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": {"rag_answer": error_message}})
            response_placeholder.markdown(error_message)
        except json.JSONDecodeError:
            error_message = "❌ 백엔드로부터 유효한 JSON 응답을 받지 못했습니다."
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": {"rag_answer": error_message}})
            response_placeholder.markdown(error_message)
        except Exception as e:
            error_message = f"❌ 알 수 없는 오류가 발생했습니다: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": {"rag_answer": error_message}})
            response_placeholder.markdown(error_message)
    
    # 응답 처리가 완료된 후 전체 UI를 갱신합니다.
    st.rerun()