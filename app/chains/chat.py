import os
from typing import List, Tuple, Set, Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from langchain_ollama import ChatOllama
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tools.structured_context import load_meds_timeline

VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./data/processed/vectorstore")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_max_docs_env = os.getenv("MAX_CONTEXT_DOCS")
try:
    MAX_CONTEXT_DOCS = int(_max_docs_env) if _max_docs_env else None
except Exception:
    MAX_CONTEXT_DOCS = None

_system = (
    # "You are a careful health data assistant.\n"
    # # "- Always include a short disclaimer: you are not providing medical advice.\n"
    # # "- You may include a short disclaimer that you cannot legally provide medical advice; however, do your best to answer the question in a helpful manner.\n"
    # "- Prefer citing exact numbers, dates, and ranges from retrieved sources.\n"
    # "- If uncertain or insufficient data, say so and suggest next steps.\n"
    # "- When analyzing over time, enumerate multiple relevant dates and highlight dose changes vs lab changes; do not rely on a single data point.\n"
    "You are a health data analysis assistant.\n"
    "- Provide educational, data-driven analysis using the provided documents and structured timelines.\n"
    "- Do not refuse; if the question is sensitive, respond with neutral, general information without directives or prescriptions.\n"
    "- If data is insufficient, state what is missing and suggest how to obtain or compute it.\n"
    "- Prefer citing exact numbers, dates, ranges, and dose changes from the sources.\n"
    "- When analyzing over time, enumerate multiple relevant dates and highlight dose changes vs lab changes; avoid relying on a single data point.\n"
)

_prompt = ChatPromptTemplate.from_messages([
    ("system", _system + "\nContext from documents (if any):\n{context}"),
    # Lightweight few-shot to discourage refusal and set style
    ("human", "Example: How did my dose changes relate to later lab results?"),
    ("assistant", "Example analysis (educational):\n- Identify dates of dose start/changes and corresponding lab dates.\n- Summarize trends (e.g., dose increase preceded level increase/decrease by N days).\n- Cite sources or timeline items with dates; avoid directives or personalized medical advice."),
    ("human", "Question: {question}")
])


def _convert_history_to_messages(history: List[dict], max_messages: int = 12) -> List[tuple]:
    """Map Streamlit-style chat history to ChatPromptTemplate messages.

    history: list of {"role": "user"|"assistant", "content": str}
    Returns list of (role, content) limited to last `max_messages` entries.
    """
    role_map = {"user": "human", "assistant": "assistant"}
    items = history[-max_messages:] if max_messages is not None else history
    msgs: List[tuple] = []
    for m in items:
        r = role_map.get(m.get("role", ""))
        c = m.get("content", "")
        if r and c:
            msgs.append((r, c))
    return msgs


def _get_llm():
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        # validate_model_on_init=True,
        # num_gpu=1,
        # temperature=0, # No randomness for debugging
        # verbose=True, # While debugging, show verbose output
    )


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
    """Robust retrieval: multi-source (general + labs + meds), MMR, query hints, dedup."""
    if _VECTORSTORE is None:
        return []

    # Base retrievers with MMR and higher fanout
    def make_retriever(filter_: dict | None = None):
        return _VECTORSTORE.as_retriever(
            search_type="mmr", # Balances relevance to the query vs. diversity among the returned documents.
            search_kwargs={
                "k": 20, # Number of documents to return.
                "fetch_k": 60, # Number of documents to fetch before deduplication.
                "lambda_mult": 0.5, # Diversity of results returned by MMR; 1 minimum, 0 maximum. 
                **({"filter": filter_} if filter_ else {}),
            },
        )

    retrievers = [
        (make_retriever(None), query),
        (make_retriever({"source_type": "labs"}), query + " focus on laboratory results and reference ranges"),
        (make_retriever({"source_type": "meds"}), query + " focus on medication dosing, dose changes, start/stop dates"),
    ]

    collected: List[Document] = []
    seen_keys: Set[str] = set()

    def doc_key(d: Document) -> str:
        m = d.metadata or {}
        src = str(m.get("source", ""))
        pg = str(m.get("page", ""))
        # include a short hash of content to separate repeated pages with different snippets
        content_sig = str(abs(hash(d.page_content)) % 10_000_000)
        return f"{src}|{pg}|{content_sig}"

    for r, q in retrievers:
        try:
            docs = r.invoke(q)
        except Exception:
            docs = []
        for d in docs:
            k = doc_key(d)
            if k in seen_keys:
                continue
            seen_keys.add(k)
            collected.append(d)

    # Optional cap based on environment; preserve MMR order within each retriever
    if MAX_CONTEXT_DOCS is not None and MAX_CONTEXT_DOCS >= 0:
        return collected[:MAX_CONTEXT_DOCS]
    return collected




def answer_question(question: str, history: Optional[List[dict]] = None) -> Tuple[str, List[str]]:
    docs = _retrieve(question)
    # Build richer context with lightweight headers to orient the model
    def fmt(d: Document) -> str:
        m = d.metadata or {}
        src = m.get("source", "unknown")
        pg = m.get("page")
        st = m.get("source_type")
        header = f"[source={src}{f' page={pg}' if pg is not None else ''}{f' type={st}' if st else ''}]"
        return header + "\n" + d.page_content

    structured_blocks: List[str] = []
    ql = question.lower()
    if any(k in ql for k in ["med", "dose", "dosing", "medication", "lab", "labs"]):
        meds_timeline = load_meds_timeline()
        if meds_timeline:
            structured_blocks.append(meds_timeline)

    context_parts: List[str] = []
    if structured_blocks:
        context_parts.append("\n\n".join(structured_blocks))
    if docs:
        context_parts.append("\n\n".join([fmt(d) for d in docs]))
    context = "\n\n".join(context_parts)

    sources = []
    for d in docs:
        meta = d.metadata or {}
        src = meta.get("source")
        pg = meta.get("page")
        label = src if src else "unknown"
        if pg is not None:
            label += f" (page {pg})"
        sources.append(label)

    # Build a dynamic prompt that includes prior turns
    hist_msgs = _convert_history_to_messages(history or [])
    prompt_msgs: List[tuple] = [
        ("system", _system + "\nContext from documents (if any):\n" + context)
    ]
    # Include the few-shot after system but before real history
    prompt_msgs.append(("human", "Example: How did my dose changes relate to later lab results?"))
    prompt_msgs.append(("assistant", "Example analysis (educational):\n- Identify dates of dose start/changes and corresponding lab dates.\n- Summarize trends (e.g., dose increase preceded level increase/decrease by N days).\n- Cite sources or timeline items with dates; avoid directives or personalized medical advice."))
    # Append prior conversation turns
    prompt_msgs.extend(hist_msgs)
    # Current user question
    prompt_msgs.append(("human", f"Question: {question}"))

    dynamic_prompt = ChatPromptTemplate.from_messages(prompt_msgs)
    llm = _get_llm()
    chain = dynamic_prompt | llm | StrOutputParser()
    answer = chain.invoke({})

    # # Ensure disclaimer present
    # if "medical advice" not in answer.lower():
    #     answer += "\n\nNote: I am not providing medical advice; use this information for education and discuss with a clinician."

    return answer, sources
