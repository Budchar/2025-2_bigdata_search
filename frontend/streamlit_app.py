import streamlit as st
import requests
import time

# -----------------------------------------------------------------
# 1. 설정
# -----------------------------------------------------------------

# FastAPI 백엔드 주소 (main.py가 실행되는 곳)
# uvicorn 기본 포트는 8000입니다.
BACKEND_URL = "http://127.0.0.1:8000/agent/query"

# Streamlit 페이지 설정
st.set_page_config(page_title="논문 검색 AI 에이전트", page_icon="📄")
st.title("📄 논문 검색 AI 에이전트")

# -----------------------------------------------------------------
# 2. 멀티턴(Multi-turn)을 위한 세션 상태 관리
# -----------------------------------------------------------------
# Streamlit은 스크립트를 매번 다시 실행하므로, 
# st.session_state를 사용해 대화 내역을 저장해야 합니다.

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------------------------------------
# 3. 채팅 UI 구성
# -----------------------------------------------------------------

# 저장된 메시지(st.session_state.messages)를 순회하며 화면에 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. 사용자 입력 받기
# st.chat_input은 사용자 입력을 받을 때까지 대기합니다.
if prompt := st.chat_input("논문에 대해 무엇이든 물어보세요."):
    
    # (A) 사용자의 메시지를 세션과 화면에 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # (B) AI 응답 처리
    with st.chat_message("assistant"):
        # FastAPI 백엔드 API 호출
        try:
            # 로딩 스피너 표시
            with st.spinner("에이전트가 논문을 검색하고 생각 중입니다..."):
                
                # main.py의 /agent/query 엔드포인트 호출
                # api.py의 Body(message=...) 형식에 맞게 json 페이로드 구성
                payload = {"message": prompt}
                
                response = requests.post(BACKEND_URL, json=payload)
                
                # 요청 실패 시 예외 발생
                response.raise_for_status() 
                
                # main.py에서 반환한 {"result": "..."} 값을 파싱
                ai_response = response.json().get("result", "오류: 'result' 키를 찾을 수 없습니다.")

            # (C) AI의 응답을 세션과 화면에 추가
            st.session_state.messages.append({"role": "assistant", "content": ai_response})
            # st.markdown(ai_response) # <-- 스트리밍이 아닐 때
            
            # (선택사항) 타이핑 효과 연출
            response_placeholder = st.empty()
            full_response = ""
            for chunk in ai_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)

        except requests.exceptions.ConnectionError:
            error_msg = "오류: 백엔드 서버에 연결할 수 없습니다. (FastAPI 서버가 실행 중인지 확인하세요)"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        except Exception as e:
            error_msg = f"오류가 발생했습니다: {e}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})