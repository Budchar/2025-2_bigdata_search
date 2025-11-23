from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import StructuredTool
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from elastic_search import ElasticSearchClient
from llm import LLMClient
from web_search import WebSearchClient
from utils import get_prompt


class RagSearchInput(BaseModel):
    query: str = Field(..., description="The search query for academic papers or knowledge base.")


class ESAgent(LLMClient, ElasticSearchClient, WebSearchClient):
    def __init__(self):
        # 부모 클래스 초기화
        LLMClient.__init__(self)
        ElasticSearchClient.__init__(self)
        WebSearchClient.__init__(self)

        # 툴 정의
        local_paper_tool = StructuredTool(
            name="local_paper_search",
            func=self.paper_search,
            description="Search for paper contents stored in the local knowledge base (Elasticsearch). Use this FIRST.",
            args_schema=RagSearchInput
        )

        google_scholar_tool = StructuredTool(
            name="google_scholar_search",
            func=self.google_scholar_search,
            description="Search for external academic papers using Google Scholar. Use this if local search fails or for latest research.",
            args_schema=RagSearchInput
        )

        tools = [local_paper_tool, google_scholar_tool]

        # 메모리 설정
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # 프롬프트 템플릿
        system_msg = get_prompt('system_message').replace("{", "{{").replace("}", "}}")

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # 2. [수정] create_tool_calling_agent 사용
        # OpenAI 최신 모델들은 Function Calling 대신 Tool Calling을 사용합니다.
        agent = create_tool_calling_agent(self.llm, tools, prompt)

        # 에이전트 실행기
        self.agent_chain = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=memory,
            handle_parsing_errors=True
        )


def main():
    es_agent = ESAgent()
    print("Welcome to the Paper Search Agent. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        try:
            response = es_agent.agent_chain.invoke({"input": user_input})
            print("Assistant:", response['output'])
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()