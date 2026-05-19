import os
import logging
from typing import Optional, Tuple, List

from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from langsmith import traceable

from app.agents.react_agent import answer_with_react_agent
from app.tools.retrieval_tool import has_vectorstore

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

logger = logging.getLogger(__name__)


def _get_llm():
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


@traceable(name="answer_question")
def answer_question(question: str, history: Optional[List[dict]] = None) -> Tuple[str, List[str]]:
    logger.info("Received question: %s", question)
    llm = _get_llm()
    return answer_with_react_agent(llm, question, history or [])
