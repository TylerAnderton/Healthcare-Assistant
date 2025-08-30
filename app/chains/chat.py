import os
from typing import List, Tuple

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langchain_ollama import ChatOllama
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./data/processed/vectorstore")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

_system = (
    "You are a careful health data assistant.\n"
    "- Always include a short disclaimer: you are not providing medical advice.\n"
    "- Prefer citing exact numbers, dates, and ranges from retrieved sources.\n"
    "- If uncertain or insufficient data, say so and suggest next steps.\n"
)

_prompt = ChatPromptTemplate.from_messages([
    ("system", _system + "\nContext from documents (if any):\n{context}"),
    ("human", "Question: {question}")
])


def _get_llm():
    return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)


def _get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _load_vectorstore():
    if not os.path.isdir(VECTORSTORE_DIR):
        return None
    try:
        vs = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=_get_embeddings())
        # Touch retriever to ensure index is valid
        _ = vs.as_retriever()
        return vs
    except Exception:
        return None


_VECTORSTORE = _load_vectorstore()


def has_vectorstore() -> bool:
    return _VECTORSTORE is not None


def _retrieve(query: str) -> List[Document]:
    if _VECTORSTORE is None:
        return []
    retriever = _VECTORSTORE.as_retriever(search_kwargs={"k": 4})
    return retriever.invoke(query)


def answer_question(question: str) -> Tuple[str, List[str]]:
    docs = _retrieve(question)
    context = "\n\n".join([d.page_content for d in docs]) if docs else ""
    sources = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source")
        pg = meta.get("page")
        label = src if src else "unknown"
        if pg is not None:
            label += f" (page {pg})"
        sources.append(label)

    llm = _get_llm()
    chain = _prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    # Ensure disclaimer present
    if "medical advice" not in answer.lower():
        answer += "\n\nNote: I am not providing medical advice; use this information for education and discuss with a clinician."

    return answer, sources
