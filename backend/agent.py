from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import StructuredTool
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field

from elastic_search import ElasticSearchClient
from llm import LLMClient


class RagSearchInput(BaseModel):
    query: str = Field(..., description="The search query for the knowledge base.")


class ESAgent(LLMClient, ElasticSearchClient):
    def __init__(self):
        """Initialize the ESAgent with the OpenAI client."""
        super().__init__()

        es_status_tool = Tool(
            name="ES_Status",
            func=self.es_ping,
            description="Checks if Elasticsearch is connected.",
        )

        culture_search_tool = StructuredTool(
                name="culture_search",
                func=self.culture_search,
                description=(
                    "Use this tool to search for information about culture from the knowledge base. "
                    "**Input must include a search query.** "
                ),
                args_schema=RagSearchInput
        )

        site_search_tool = StructuredTool(
                name="site_search",
                func=self.site_search,
                description=(
                    "Use this tool to search for information about site from the knowledge base. "
                    "**Input must include a search query.** "
                ),
                args_schema=RagSearchInput
        )

        tools = [es_status_tool, culture_search_tool, site_search_tool]

        # Initialize memory to keep track of the conversation
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.agent_chain = initialize_agent(
                tools,
                self.llm,
                agent=AgentType.OPENAI_FUNCTIONS,
                memory=memory,
                verbose=True,
                handle_parsing_errors=True,
                system_message="""
            You are an AI assistant that helps with questions about place to play using a knowledge base. Be concise, sharp, to the point, and respond in one paragraph.
            You have access to the following tools:

            - **ES_Status**: Checks if Elasticsearch is connected.
            - **culture_search**: Use this to search for culture information in the knowledge base. **Input must include a search query.
            - **site_search**: Use this to search for site information in the knowledge base. **Input must include a search query.

            **Important Instructions:**

            - **Extract proper culture and site info from user using question.**
            - **If the info from knowledge base is too big. filter this by question to user, politely ask them to provide one before proceeding.**

            When you decide to use a tool, use the following format *exactly*:
            Thought: [Your thought process about what you need to do next]
            Action: [The action to take, should be one of [ES_Status, culture_search, site_search]]
            Action Input: {"query": "the search query"}


            If you receive an observation after an action, you should consider it and then decide your next step. If you have enough information to answer the user's question, respond with:
            Thought: [Your thought process]
            Assistant: [Your final answer to the user]

            **Examples:**

            - **User's Question:** "Tell me about the place to play."
              Thought: I need to search for information about the place to play.
              Action: site_search
              Action Input: {"query": "the place to play"}

            - **User's Question:** "What happened during the presidential election?"
              Thought: The user didn't specify a date. I should ask for a date range.
              Assistant: Could you please specify the date or date range for the presidential election you're interested in?

            Always ensure that your output strictly follows one of the above formats, and do not include any additional text or formatting.

            Remember:

            - **Do not** include any text before or after the specified format.
            - **Do not** add extra explanations.
            - **Do not** include markdown, bullet points, or numbered lists unless it is part of the Assistant's final answer.

            Your goal is to assist the user by effectively using the tools when necessary and providing clear and concise answers.
            """
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
