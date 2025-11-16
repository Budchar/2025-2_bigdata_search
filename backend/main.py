from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import Body
from agent import ESAgent

app = FastAPI()
es_agnet = ESAgent()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # React 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/agent/query")
async def read_item(body: Body):
    result = es_agnet.agent_chain.invoke(input=body.message)
    # 결과가 문자열이 아닌 경우 문자열로 변환
    if isinstance(result, dict):
        if 'output' in result:
            return {"result": result['output']}
        else:
            return {"result": str(result)}
    return {"result": str(result)}