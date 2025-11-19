import pathlib

PROMPT_PATH = pathlib.Path(__file__).parent / 'prompt'

def get_prompt(name: str) -> str:
    with open(PROMPT_PATH / name, mode="r", encoding="utf-8") as f:
        content = f.read()
    return content