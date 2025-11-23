#!/bin/bash

# 종료 시 백그라운드 프로세스를 정리하는 함수
cleanup() {
    echo -e "\n🛑 스크립트를 종료합니다. 백그라운드 서버를 정리합니다..."

    # 백엔드 서버(uvicorn) 종료
    if [ -n "$BACKEND_PID" ]; then
        kill $BACKEND_PID
        echo "FastAPI 백엔드 서버 (PID: $BACKEND_PID) 종료됨."
    fi

    # 프론트엔드 서버(streamlit) 종료
    if [ -n "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID
        echo "Streamlit 프론트엔드 서버 (PID: $FRONTEND_PID) 종료됨."
    fi
    exit
}

# Ctrl+C (SIGINT) 또는 종료(SIGTERM) 신호를 받으면 cleanup 함수 호출
trap cleanup SIGINT SIGTERM

echo "🚀 FastAPI 백엔드 서버를 시작합니다 (http://localhost:8000)..."
# 1. backend 폴더로 이동하여 uvicorn을 백그라운드로 실행
(cd backend && uvicorn main:app --reload) &
# 백그라운드 프로세스의 PID(Process ID) 저장
BACKEND_PID=$!

echo "🎨 Streamlit 프론트엔드 서버를 시작합니다 (http://localhost:8501)..."
# 2. backend 폴더로 이동하여 streamlit을 백그라운드로 실행
(cd frontend && streamlit run streamlit_app_v2.py) &
# 백그라운드 프로세스의 PID 저장
FRONTEND_PID=$!

echo -e "\n✅ 모든 서버가 실행 중입니다."
echo "백엔드 PID: $BACKEND_PID"
echo "프론트엔드 PID: $FRONTEND_PID"
echo -e "\n👉 중지하려면 Ctrl+C 를 누르세요."

# 스크립트가 바로 종료되지 않도록 백그라운드 프로세스들을 기다림
wait $BACKEND_PID
wait $FRONTEND_PID