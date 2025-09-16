import os
import re
import json
from typing import List, Tuple, Set, Optional, Dict, Any
import logging

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from langchain_ollama import ChatOllama
# from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from app.tools.structured_context import load_meds_timeline, load_labs_panel, load_whoop_recent
from .prompts import *

VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "./data/processed/vectorstore")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ENABLE_TOOL_CALLING = os.getenv("ENABLE_TOOL_CALLING", "1") in {"1", "true", "True"}
PREFER_STRUCTURED = os.getenv("PREFER_STRUCTURED", "0") in {"1", "true", "True"}

_max_docs_env = os.getenv("MAX_CONTEXT_DOCS")
try:
    MAX_CONTEXT_DOCS = int(_max_docs_env) if _max_docs_env else None
    MAX_DOCS_PER_RETRIEVER = int(os.getenv("MAX_DOCS_PER_RETRIEVER", "12"))
except Exception:
    MAX_CONTEXT_DOCS = None
    MAX_DOCS_PER_RETRIEVER = 12

logger = logging.getLogger(__name__)

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
    logger.info('Building context from %d docs', len(docs))
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
        logger.info('Loading meds timeline')
        meds_timeline = load_meds_timeline()
        if meds_timeline:
            structured_blocks.append(meds_timeline)

    # Latest labs panel snapshot when question touches labs broadly
    if any(k in ql for k in ["lab", "labs", "panel", "analyte"]):
        logger.info('Loading labs panel')
        labs_panel = load_labs_panel()
        if labs_panel:
            structured_blocks.append(labs_panel)

    # WHOOP recent snapshot when question touches WHOOP or sleep/recovery/workouts
    if any(k in ql for k in [
        "whoop", "sleep", "recovery", "workout", "strain", "hrv", "rhr", "resting heart rate", "respiratory rate", "skin temp"
    ]):
        logger.info('Loading WHOOP recent snapshot')
        whoop_block = load_whoop_recent()
        if whoop_block:
            structured_blocks.append(whoop_block)

    # Analyte-specific summaries if the question mentions them
    try:
        from tools import labs_tool as LT  # local import to avoid heavy deps at module import time
        names = LT.list_analytes()
        logger.info('Checking for analyte-specific queries')
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
            logger.info('Matched analytes: %s', matches)

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
        logger.error('Failed to add analyte-specific summaries to context')
        pass

    context_parts: List[str] = []
    if structured_blocks:
        context_parts.append("\n\n".join(structured_blocks))
    if docs:
        # Explicitly mark docs as background only to reduce factual interference
        bg_header = "[background_docs]\nUse these only for narrative background if structured blocks above are insufficient. For numeric values and dates, rely on structured blocks or tools."
        context_parts.append(bg_header + "\n\n" + "\n\n".join([fmt(d) for d in docs]))
    context = "\n\n".join(context_parts)
    return context


# -----------------------------
# Tool-calling support (optional)
# -----------------------------
def _answer_with_tools(llm: ChatOllama, prompt: ChatPromptTemplate) -> str:
    """Let the model call structured tools to fetch authoritative data.

    We avoid adding long background documents here to reduce interference and rely
    on tools that read the canonical parquet tables.
    """
    try:
        from app.tools import labs_tool as LT
        from app.tools import whoop_tool as WT
        from app.tools import meds_tool as MT  # noqa: F401 (for potential future expansion)
    except Exception:
        LT = None
        WT = None
        MT = None

    # Define callable tools that the model can invoke
    # Labs Tools
    def labs_list_analytes_tool(prefix: Optional[str] = None, table_path: Optional[str] = None) -> List[str]:
        """Return a list of all unique lab analytes.

        Args:
            prefix: Optional prefix to filter analytes by.
            table_path: Optional path to the labs table.
        """
        if LT is None:
            return []
        return LT.list_analytes(prefix=prefix, table_path=table_path)
    
    def labs_latest_value_tool(analyte: str, table_path: Optional[str] = None) -> Optional[Dict]:
        """Return the latest value for a lab analyte.

        Args:
            analyte: Name of the lab analyte.
            table_path: Optional path to the labs table.
        """
        if LT is None:
            return None
        return LT.latest_value(analyte, table_path=table_path)
    
    def labs_history_tool(analyte: str, limit: int = 10, ascending: bool = True) -> List[Dict[str, Any]]:
        """Return recent history for a lab analyte.

        Args:
            analyte: Name of the lab analyte, e.g., "ALT (SGPT)".
            limit: Max number of rows to return.
            ascending: Sort order by date.
        """
        if LT is None:
            return []
        return LT.history(analyte, limit=limit, ascending=ascending)

    def labs_summary_tool(analyte: str) -> Dict[str, Any]:
        """Return a summary for a lab analyte (last value/date, delta, unit, ref range).

        Args:
            analyte: Name of the lab analyte.
        """
        if LT is None:
            return {}
        s = LT.summary(analyte)
        return s or {}

    def labs_value_on_date_tool(analyte: str, date: str) -> Dict[str, Any]:
        """Return the value of an analyte on an exact date (YYYY-MM-DD).

        Args:
            analyte: Name of the lab analyte.
            date: Date in YYYY-MM-DD.
        """
        if LT is None:
            return {}
        rows = LT.history(analyte, ascending=True)
        target = str(date)
        for r in rows:
            if str(r.get("date")) == target:
                return {
                    "analyte": analyte,
                    "date": r.get("date"),
                    "value": r.get("value"),
                    "unit": r.get("unit"),
                    "ref_low": r.get("ref_low"),
                    "ref_high": r.get("ref_high"),
                    "flag": r.get("flag"),
                }
        return {}

    # Meds Tools
    def meds_timeline_tool() -> str:
        """Return a compact medication dosing timeline from structured data."""
        try:
            return load_meds_timeline() or ""
        except Exception:
            return ""

    def meds_history_tool(medication: str, fuzzy: bool = True) -> List[Dict[str, Any]]:
        """Return the chronological dosing history for a medication.

        Args:
            medication: Name of the medication to search for.
            fuzzy: If True, enable fuzzy matching on the medication name.
        """
        if MT is None:
            return []
        try:
            return MT.get_medication_history(medication, fuzzy=fuzzy)
        except Exception:
            return []

    def meds_dosage_on_date_tool(medication: str, date: Optional[str] = None, fuzzy: bool = True) -> Dict[str, Any]:
        """Return the dose/frequency for a medication on a specific date (YYYY-MM-DD). If date is omitted, use today.

        Args:
            medication: Name of the medication to search for.
            date: Date in YYYY-MM-DD to query. If None, defaults to today.
            fuzzy: If True, enable fuzzy matching on the medication name.
        """
        if MT is None:
            return {}
        try:
            return MT.dosage_on_date(medication, date, fuzzy=fuzzy)
        except Exception:
            return {}

    def meds_list_medications_tool() -> List[str]:
        """Return a list of medications."""
        return MT.list_medications()

    def meds_list_current_tool(date: Optional[str|None] = None) -> dict:
        """Return a list of current medications and dosages on a given date."""
        if MT is None:
            return {}
        return MT.list_current(date=date)

    # Whoop Tools
    def whoop_recovery_strain_on_date_tool(date: str) -> Dict[str, Any]:
        """Return WHOOP recovery score and day strain for an exact date (YYYY-MM-DD).

        Args:
            date: Date in YYYY-MM-DD.
        """
        if WT is None:
            return {}
        rec = WT.recovery(start=date, end=date, ascending=True, limit=10)
        for r in rec:
            if str(r.get("date")) == str(date):
                return {
                    "date": r.get("date"),
                    "recovery": r.get("recovery_score"),
                    "strain": r.get("strain"),
                    "rhr_bpm": r.get("rhr_bpm"),
                    "hrv_ms": r.get("hrv_ms"),
                }
        return {}

    tools = [
        # Labs Tools
        labs_list_analytes_tool,
        labs_latest_value_tool,
        labs_history_tool,
        labs_summary_tool,
        labs_value_on_date_tool,
        # Meds Tools
        meds_timeline_tool,
        meds_history_tool,
        meds_dosage_on_date_tool,
        meds_list_medications_tool,
        meds_list_current_tool,
        # Whoop Tools
        whoop_recovery_strain_on_date_tool,
    ]

    llm_tools = llm.bind_tools(tools)
    messages = prompt.format_messages()

    # Simple tool loop (max 3 rounds)
    max_rounds = 3
    for _ in range(max_rounds):
        ai: AIMessage = llm_tools.invoke(messages)  # type: ignore[assignment]
        messages.append(ai)
        if not getattr(ai, "tool_calls", None):
            # Model chose not to call tools; return its answer
            return ai.content if isinstance(ai.content, str) else str(ai.content)

        # Dispatch tool calls
        for tc in ai.tool_calls:
            name = tc.get("name")
            args = tc.get("args", {}) or {}
            try:
                if name == "labs_history_tool":
                    result = labs_history_tool(**args)
                elif name == "labs_list_analytes_tool":
                    result = labs_list_analytes_tool(**args)
                elif name == "labs_latest_value_tool":
                    result = labs_latest_value_tool(**args)
                elif name == "labs_summary_tool":
                    result = labs_summary_tool(**args)
                elif name == "labs_value_on_date_tool":
                    result = labs_value_on_date_tool(**args)

                elif name == "meds_history_tool":
                    result = meds_history_tool(**args)
                elif name == "meds_dosage_on_date_tool":
                    result = meds_dosage_on_date_tool(**args)
                elif name == "meds_list_medications_tool":
                    result = meds_list_medications_tool(**args)
                elif name == "meds_list_current_tool":
                    result = meds_list_current_tool(**args)

                elif name == "whoop_recovery_strain_on_date_tool":
                    result = whoop_recovery_strain_on_date_tool(**args)

                else:
                    result = {"error": f"Unknown tool: {name}"}

            except Exception as e:
                logger.exception("Tool execution failed: %s", name)
                result = {"error": f"Exception in {name}: {e}"}

            # Feed tool result back to the model
            messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=tc.get("id", ""),
                )
            )

    # If exceeded rounds, return last AI content
    last = messages[-1]
    if isinstance(last, AIMessage):
        return last.content if isinstance(last.content, str) else str(last.content)
    return "I'm unable to complete the tool-assisted answer right now."



def answer_question(question: str, history: Optional[List[dict]] = None) -> Tuple[str, List[str]]:
    logger.info('Received question: %s', question)
    docs = _retrieve(question)
    logger.info('Retrieved %d docs', len(docs))

    # Optionally suppress background docs for structured/date-specific questions
    def _mentions_date(s: str) -> bool:
        # Match YYYY-MM-DD or YYYY-MM or YYYY patterns likely used in queries
        return bool(re.search(r"\b\d{4}-(\d{2})(?:-\d{2})?\b", s)) or bool(re.search(r"\b\d{4}\b", s))

    ql = question.lower()
    structured_keywords = [
        "lab", "labs", "panel", "analyte", "med", "dose", "dosing", "medication",
        "whoop", "sleep", "recovery", "workout", "strain", "hrv", "rhr", "respiratory rate"
    ]
    is_structured_query = any(k in ql for k in structured_keywords)
    if PREFER_STRUCTURED and (is_structured_query or _mentions_date(ql)):
        logger.info('Suppressing retrieved docs due to structured/date-specific query')
        docs_for_context = []
    else:
        docs_for_context = docs

    context = _build_context(question, docs_for_context)

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
        ("system", SYSTEM_BASE + "\nContext (structured first):\n" + context)
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

    if ENABLE_TOOL_CALLING:
        # Run a lightweight tool-calling loop using structured tools
        answer = _answer_with_tools(llm, dynamic_prompt)
    else:
        chain = dynamic_prompt | llm | StrOutputParser()
        answer = chain.invoke({})

    return answer, sources
