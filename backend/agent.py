from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import StructuredTool
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field

from elastic_search import ElasticSearchClient
from llm import LLMClient
from utils import get_prompt


class RagSearchInput(BaseModel):
    query: str = Field(..., description="The search query for the knowledge base.")


class ESAgent(LLMClient, ElasticSearchClient):
    def __init__(self):
        super().__init__()

        db_search_tool = StructuredTool(
                name="site_search",
                func=self.site_search,
                description=get_prompt('db_search_tool_description'),
                args_schema=RagSearchInput
        )

        tools = [db_search_tool]

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.agent_chain = initialize_agent(
                tools,
                self.llm,
                agent=AgentType.OPENAI_FUNCTIONS,
                memory=memory,
                verbose=True,
                handle_parsing_errors=True,
                system_message=get_prompt('system_message')
        )


# Interactive conversation with the agent
def main():
    es_agent = ESAgent()
    print("Welcome to the chat agent. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        # Update method call to address deprecation warning
        response = es_agent.agent_chain.invoke(input=user_input)
        print("Assistant:", response['output'])


if __name__ == "__main__":
    main()
