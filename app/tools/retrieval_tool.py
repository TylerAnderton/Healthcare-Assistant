import os
import logging
from typing import List, Optional, Set

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./data/processed/vectorstore")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

try:
    from app.constants import MAX_DOCS_PER_RETRIEVER, MAX_CONTEXT_DOCS
except Exception:
    MAX_DOCS_PER_RETRIEVER = 20
    MAX_CONTEXT_DOCS = None


def _get_embeddings():
    from langchain_huggingface.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def _load_vectorstore():
    if not os.path.isdir(VECTORSTORE_DIR):
        return None
    try:
        from langchain_chroma import Chroma
        vs = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=_get_embeddings())
        _ = vs.as_retriever()
        return vs
    except Exception:
        return None


_VECTORSTORE = _load_vectorstore()


def has_vectorstore() -> bool:
    return _VECTORSTORE is not None


def _retrieve(query: str, source_type: Optional[str] = None) -> List[Document]:
    if _VECTORSTORE is None:
        return []

    def make_retriever(filter_: Optional[dict] = None):
        return _VECTORSTORE.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": MAX_DOCS_PER_RETRIEVER,
                "fetch_k": MAX_DOCS_PER_RETRIEVER * 3,
                "lambda_mult": 0.5,
                **({"filter": filter_} if filter_ else {}),
            },
        )

    if source_type:
        retrievers = [(make_retriever({"source_type": source_type}), query)]
    else:
        retrievers = [
            (make_retriever(None), query),
            (make_retriever({"source_type": "labs"}), query + " focus on laboratory results and reference ranges"),
            (make_retriever({"source_type": "meds"}), query + " focus on medication dosing, dose changes, start/stop dates"),
            (make_retriever({"source_type": "whoop"}), query + " focus on WHOOP metrics: sleep, recovery, workouts, HRV, RHR, respiratory rate, strain"),
        ]

    collected: List[Document] = []
    seen_keys: Set[str] = set()

    def doc_key(d: Document) -> str:
        m = d.metadata or {}
        src = str(m.get("source", ""))
        pg = str(m.get("page", ""))
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

    if MAX_CONTEXT_DOCS is not None and MAX_CONTEXT_DOCS >= 0:
        return collected[:MAX_CONTEXT_DOCS]
    return collected


def _format_docs_for_tool(docs: List[Document]) -> str:
    if not docs:
        return "No relevant documents found."
    parts = []
    for d in docs:
        m = d.metadata or {}
        src = m.get("source", "unknown")
        pg = m.get("page")
        st = m.get("source_type")
        header = f"[source={src}{f' page={pg}' if pg is not None else ''}{f' type={st}' if st else ''}]"
        parts.append(header + "\n" + d.page_content)
    return "\n\n".join(parts)


def _extract_sources(docs: List[Document]) -> List[str]:
    sources: List[str] = []
    seen: Set[str] = set()
    for d in docs:
        m = d.metadata or {}
        src = m.get("source")
        pg = m.get("page")
        if src:
            label = src if pg is None else f"{src} (page {pg})"
            if label not in seen:
                seen.add(label)
                sources.append(label)
    return sources
