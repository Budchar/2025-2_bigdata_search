import pathlib

PROMPT_PATH = pathlib.Path(__file__).parent.parent / 'prompt'

def get_prompt(name: str) -> str:
    if not name.endswith(".txt"):
        name += ".txt"
    with open(PROMPT_PATH / name, mode="r", encoding="utf-8") as f:
        content = f.read()
    return content