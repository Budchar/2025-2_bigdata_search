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

        # 1. 도구 정의
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

        # 🚀 [수정] 시스템 프롬프트: 요약 강제 및 포맷 엄격화
        system_msg = """
너는 전문적인 학술 논문 검색 에이전트이다. **최종 답변은 반드시 한국어로 작성하라.**

### 1. 검색 결과 처리 지침 (매우 중요)
- **Local 검색이든 Web 검색이든, 검색된 논문의 내용(Content/Snippet)을 읽고 핵심 내용을 3문장 내외의 한국어로 '요약'하여 `snippet` 필드에 채워라.**
- 단순히 원문을 복사하지 말고, 사용자가 이해하기 쉽게 내용을 요약/번역해야 한다.
- 저자, 연도, 인용수가 불확실하면 'Unknown' 또는 0으로 표기하라.

### 2. 절대적 제약 사항 (Strict Output Format):
- **반드시 순수 JSON 형식만 출력하라.** (Markdown ```json ... ``` 코드 블록 절대 사용 금지)
- 불필요한 서론이나 사족을 붙이지 마라. 오직 JSON 데이터만 반환하라.

### 3. 응답 포맷 (Unified JSON Structure):
{{
  "rag_answer": "사용자 질문에 대한 종합적인 답변 및 인사이트 (한국어 서술형)",
  "related_papers": [
     {{
        "title": "논문 제목",
        "authors": "저자 이름들 (콤마로 구분)",
        "published_year": "연도 (예: 2023)",
        "citation_count": "인용수 (정수, 없으면 0)",
        "url": "URL 링크 또는 파일 경로",
        "snippet": "💡 여기에 논문의 핵심 내용을 한국어로 요약해서 작성"
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
            verbose=True,
            handle_parsing_errors=True,
        )

    # (하위 메서드들은 기존 유지)
    def generate_deep_summary(self, title: str, snippet: str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Summary logic..."),
            ("human", f"Title: {title}\nSnippet: {snippet}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({})

    def compare_and_recommend(self, papers_info: str, user_goal: str):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Recommendation logic..."),
            ("human", f"Goal: {user_goal}\nPapers: {papers_info}")
        ])
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({})
