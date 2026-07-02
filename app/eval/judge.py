import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()


def _judge_config() -> tuple[str, str]:
    model = os.environ.get("JUDGE_MODEL") or os.environ.get("OLLAMA_MODEL", "qwen3:4b")
    base_url = os.environ.get("JUDGE_BASE_URL") or os.environ.get(
        "OLLAMA_BASE_URL", "http://localhost:11434"
    )
    return model, base_url


def get_ragas_judge_llm() -> ChatOllama:
    # JUDGE_MODEL must be a non-thinking model (e.g. qwen3 instruct-2507):
    # thinking judges burn thousands of hidden tokens per verdict, and 2507
    # thinking variants ignore think=false entirely.
    model, base_url = _judge_config()
    return ChatOllama(model=model, base_url=base_url, temperature=0)
