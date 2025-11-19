프론트 시작 방법 
---
cd frontend && streamlit run streamlit_app.py

백엔드 시작 방법 
---
cd backend && uvicorn main:app --reload

prompt 설명
---
system_message : LLM에 기본적으로 적용할 프롬프트 (역할, 어투, 번역 등등)

tool_description 
- db_search_tool_description : 언제 어떻게 DB(ES)를 이용해 검색해야할지에 대한 설명