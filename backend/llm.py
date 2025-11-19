from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI
from config import set_open_params
import time

load_dotenv()


class LLMClient:
    def __init__(self):
        super().__init__()
        """Initialize the Agent with the OpenAI client."""
        self.llm = self.get_llm_client()

    def get_llm_client(self):
        """Get the OpenAI client with the specified parameters."""
        openai_params = set_open_params()
        client = ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                ** openai_params
        )
        return client

    def invoke(self, system_message_content, user_prompt_content, verbose=False):
        messages = [
            ('system', system_message_content),
            ('human', user_prompt_content)
        ]

        time_start = time.time()
        retry_count = 3
        for i in range(0, retry_count):
            while True:
                try:
                    response = self.llm.invoke(messages)

                    answer = response.content

                    if verbose:
                        tokens = response.usage_metadata.total_tokens
                        time_end = time.time()
                        print('prompt: %s \n token: %d \n %.1fsec\n\n anwer : \n%s' % (user_prompt_content, tokens,
                                                                                       (time_end - time_start), answer))
                    return answer

                except Exception as error:
                    print(f"API Error: {error}")
                    print(f"Retrying {i + 1} time(s) in 4 seconds...")

                    if i + 1 == retry_count:
                        return user_prompt_content, None, None
                    time.sleep(4)
                    continue
        return None


def main():
    agent = LLMClient()
    print("Welcome to the chat agent. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        # Update method call to address deprecation warning
        response = agent.invoke(system_message_content='answer as korean',
                                user_prompt_content=user_input)
        print("Assistant:", response)


if __name__ == "__main__":
    main()
