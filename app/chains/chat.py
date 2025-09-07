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
from tools.structured_context import load_meds_timeline, load_labs_panel, load_whoop_recent
from chains.prompts import *

VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./data/processed/vectorstore")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
_max_docs_env = os.getenv("MAX_CONTEXT_DOCS")
try:
    MAX_CONTEXT_DOCS = int(_max_docs_env) if _max_docs_env else None
    MAX_DOCS_PER_RETRIEVER = int(os.getenv("MAX_DOCS_PER_RETRIEVER", "12"))
except Exception:
    MAX_CONTEXT_DOCS = None
    MAX_DOCS_PER_RETRIEVER = 12


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
                "k": MAX_DOCS_PER_RETRIEVER, # Number of documents to return.
                "fetch_k": MAX_DOCS_PER_RETRIEVER * 3, # Number of documents to fetch before deduplication.
                "lambda_mult": 0.5, # Diversity of results returned by MMR; 1 minimum, 0 maximum. 
                **({"filter": filter_} if filter_ else {}),
            },
        )

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


def _build_context(question: str, docs: List[Document]) -> str:
    def fmt(d: Document) -> str:
        m = d.metadata or {}
        src = m.get("source", "unknown")
        pg = m.get("page")
        st = m.get("source_type")
        header = f"[source={src}{f' page={pg}' if pg is not None else ''}{f' type={st}' if st else ''}]"
        return header + "\n" + d.page_content

    structured_blocks: List[str] = []
    ql = question.lower()

    # Medication dosing timeline when question touches meds or labs
    if any(k in ql for k in ["med", "dose", "dosing", "medication", "lab", "labs"]):
        print('Loading meds timeline')
        meds_timeline = load_meds_timeline()
        if meds_timeline:
            structured_blocks.append(meds_timeline)

    # Latest labs panel snapshot when question touches labs broadly
    if any(k in ql for k in ["lab", "labs", "panel", "analyte"]):
        print('Loading labs panel')
        labs_panel = load_labs_panel()
        if labs_panel:
            structured_blocks.append(labs_panel)

    # WHOOP recent snapshot when question touches WHOOP or sleep/recovery/workouts
    if any(k in ql for k in [
        "whoop", "sleep", "recovery", "workout", "strain", "hrv", "rhr", "resting heart rate", "respiratory rate", "skin temp"
    ]):
        print('Loading WHOOP recent snapshot')
        whoop_block = load_whoop_recent()
        if whoop_block:
            structured_blocks.append(whoop_block)

    # Analyte-specific summaries if the question mentions them
    try:
        from tools import labs_tool as LT  # local import to avoid heavy deps at module import time
        names = LT.list_analytes()
        print('Checking for analyte-specific queries')
        if names:
            ql_compact = ql
            # direct name substring match
            matches = [n for n in names if n.lower() in ql_compact]
            # if none, try token prefix match for tokens >=3 chars
            if not matches:
                toks = [t.strip(",.;:!?()[]{}") for t in ql_compact.split()]
                cands = set()
                for t in toks:
                    if len(t) >= 3:
                        for n in names:
                            if n.lower().startswith(t):
                                cands.add(n)
                matches = list(cands)
            # limit to top few to keep prompt lean
            matches = matches[:3]
            print('Matched analytes:', matches)

            if matches:
                # Summaries
                lines = ["[structured_labs_summaries]"]
                for n in matches:
                    s = LT.summary(n)
                    if not s:
                        continue
                    dpct = s.get("pct_change_from_prev")
                    dpct_str = f"{dpct:.1f}%" if isinstance(dpct, (int, float)) else ""
                    delta = s.get("delta_from_prev")
                    unit = s.get("unit") or ""
                    ref = (f"{s.get('ref_low')}-{s.get('ref_high')}"
                           if (s.get('ref_low') is not None or s.get('ref_high') is not None) else "")
                    pieces = [
                        f"{s['analyte']}:",
                        f"{s.get('last_value')} {unit}".strip(),
                        f"on {s.get('last_date')}" if s.get('last_date') else "",
                        f"(ref {ref})" if ref else "",
                        f"[flag {s.get('flag')}]" if s.get('flag') else "",
                        (f"Δprev {delta}" + (f" ({dpct_str})" if dpct_str else "")) if (delta is not None) else "",
                        f"[{s.get('out_of_range')}]" if s.get('out_of_range') else "",
                    ]
                    line = " ".join([p for p in pieces if p]).strip()
                    if line:
                        lines.append("- " + line)
                if len(lines) > 1:
                    structured_blocks.append("\n".join(lines))

                # Recent history (chronological) for the same matches
                hist_lines = ["[structured_labs_history]"]
                for n in matches:
                    h = LT.history(n, limit=5, ascending=False)
                    if not h:
                        continue
                    h = list(reversed(h))
                    seq = ", ".join([
                        f"{x.get('date')}: {x.get('value')} {x.get('unit') or ''}".strip()
                        for x in h
                    ])
                    if seq:
                        hist_lines.append(f"- {n}: {seq}")
                if len(hist_lines) > 1:
                    structured_blocks.append("\n".join(hist_lines))

    except Exception:
        # If any issue occurs, silently skip structured labs additions to avoid user disruption
        print('Failed to add analyte-specific summaries to context')
        pass

    context_parts: List[str] = []
    if structured_blocks:
        context_parts.append("\n\n".join(structured_blocks))
    if docs:
        context_parts.append("\n\n".join([fmt(d) for d in docs]))
    context = "\n\n".join(context_parts)
    return context


def answer_question(question: str, history: Optional[List[dict]] = None) -> Tuple[str, List[str]]:
    docs = _retrieve(question)

    context = _build_context(question, docs)

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
        ("system", SYSTEM_BASE + "\nContext from documents (if any):\n" + context)
    ]

    # Include the few-shot after system but before real history
    # prompt_msgs.append(("human", "Example: How did my dose changes relate to later lab results?"))
    # prompt_msgs.append(("assistant", "Example analysis (educational):\n- Identify dates of dose start/changes and corresponding lab dates.\n- Summarize trends (e.g., dose increase preceded level increase/decrease by N days).\n- Cite sources or timeline items with dates; avoid directives or personalized medical advice."))
    for example in few_shot_examples:
        prompt_msgs.append(("human", example["human"]))
        prompt_msgs.append(("assistant", example["assistant"]))

    # Append prior conversation turns
    prompt_msgs.extend(hist_msgs)

    # Current user question
    prompt_msgs.append(("human", f"Question: {question}"))

    dynamic_prompt = ChatPromptTemplate.from_messages(prompt_msgs)
    llm = _get_llm()
    chain = dynamic_prompt | llm | StrOutputParser()
    answer = chain.invoke({})

    return answer, sources
