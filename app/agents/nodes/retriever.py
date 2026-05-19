import logging
from langchain_core.messages import ToolMessage
from app.tools.retrieval_tool import _retrieve, _format_docs_for_tool, _extract_sources

logger = logging.getLogger(__name__)


def _retriever_node(state: dict) -> dict:
    last = state["messages"][-1]
    retrieval_calls = [
        tc for tc in (getattr(last, "tool_calls", None) or [])
        if tc["name"] == "retrieve_documents"
    ]

    messages = []
    sources: list = []
    for tc in retrieval_calls:
        args = tc["args"]
        query = args.get("query", "")
        source_type = args.get("source_type")
        docs = _retrieve(query=query, source_type=source_type)
        text = _format_docs_for_tool(docs)
        sources.extend(_extract_sources(docs))
        messages.append(ToolMessage(content=text, tool_call_id=tc["id"]))

    return {"messages": messages, "sources": sources}
