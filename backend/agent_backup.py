from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import StructuredTool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List

from .elastic_search import ElasticSearchClient
from .llm import LLMClient
from .web_search import WebSearchClient

class RagSearchInput(BaseModel):
    query: str = Field(..., description="The search query.")

class ESAgent(LLMClient, ElasticSearchClient, WebSearchClient):
    def __init__(self):
        LLMClient.__init__(self)
        ElasticSearchClient.__init__(self)
        WebSearchClient.__init__(self)

        # 1. 도구 정의: 로컬 검색 우선순위는 여전히 유지 (중요!)
        local_paper_tool = StructuredTool(
            name="local_paper_search",
            func=self.paper_search,
            description="""Search the **priority local paper database**. 
            **You MUST use this tool first** for academic queries.""",
            args_schema=RagSearchInput,
        )

        google_scholar_tool = StructuredTool(
            name="google_scholar_search",
            func=self.google_scholar_search,
            description="""Search Google Scholar for general and latest academic papers. 
            **ONLY use this tool if local_paper_search yields poor or no results**.""",
            args_schema=RagSearchInput,
        )

        tools = [local_paper_tool, google_scholar_tool]

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        # 🚀 [수정] 시스템 프롬프트: JSON 구조 단순화 (속도 최적화)
        system_msg = """
너는 전문적인 학술 논문 검색 에이전트이다. 
**최종 답변은 반드시 한국어로 작성하며, 아래 JSON 형식을 엄격히 준수하라.**

### 1. 검색 결과 처리 지침
- **Local/Web 검색 결과의 'Content'나 'Snippet' 내용을 읽고, 핵심 내용을 3문장 내외의 한국어로 요약하여 결과 JSON의 `snippet` 필드에 작성하라.** (단순 복사 금지)
- 저자, 연도, 인용수 정보가 불확실하면 'Unknown' 또는 0으로 표기하라.

### 2. 절대적 제약 사항 (Strict Output Format):
- **반드시 순수 JSON 형식만 출력하라.** (Markdown ```json ... ``` 코드 블록 사용 금지)
- JSON의 Key는 반드시 큰따옴표(")를 사용하라.
- **Python의 `json.loads()`로 파싱 가능한 형태여야 한다.**

### 3. 응답 포맷 (Unified JSON Structure):
{{
  "rag_answer": "사용자 질문에 대한 종합적인 답변 (한국어 서술형)",
  "related_papers": [
     {{
        "title": "논문 제목",
        "authors": "저자 목록",
        "published_year": "연도 (예: 2023)",
        "citation_count": "인용수 (정수)",
        "url": "URL 링크 또는 파일 경로",
        "snippet": "논문 핵심 내용 요약 (반드시 한국어로 번역 및 요약됨)"
     }}
  ]
}}
"""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(self.llm, tools, prompt)

        self.agent_chain = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True, # 디버깅을 위해 켜둠
            handle_parsing_errors=True,
        )

    # --- 심층 요약 및 키워드 추출 ---
    def generate_deep_summary(self, title: str, snippet: str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            당신은 학술 논문 분석가입니다. 주어진 논문 정보를 바탕으로 다음 3가지를 JSON 형식으로 작성하세요.
            
            {{
                "summary_kr": "한국어 요약 (3문장 내외)",
                "summary_en": "English Summary",
                "keywords": ["키워드1", "키워드2", "키워드3", "키워드4", "키워드5"]
            }}
            """),
            ("human", f"Title: {title}\nSnippet: {snippet}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({})

    # --- 논문 비교 및 추천 ---
    def compare_and_recommend(self, papers_info: str, user_goal: str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            당신은 수석 연구원입니다. JSON 형식으로 답변하세요.
            
            {{
                "analysis": "비교 분석 내용 (마크다운)",
                "recommendation": "추천 논문 및 이유 (마크다운)"
            }}
            """),
            ("human", f"사용자 연구 목적: {user_goal}\n\n[비교할 논문 목록]\n{papers_info}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({})