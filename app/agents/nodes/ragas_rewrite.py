from langchain_core.messages import SystemMessage

from app.constants import RAGAS_FAITHFULNESS_THRESHOLD

_FEEDBACK_TEMPLATE = (
    "RAGAS EVALUATION FEEDBACK: Your previous answer did not meet the faithfulness threshold.\n"
    "Faithfulness score: {faithfulness_score:.2f} (minimum required: {threshold})\n\n"
    "Your answer contained claims that were not directly supported by the retrieved documents.\n"
    "You MUST call retrieve_documents again with a more specific query to gather better evidence.\n"
    "Original query: {query}\n"
    "Do NOT answer from memory or general knowledge. Only use retrieved document content."
)


def ragas_rewrite_node(state: dict) -> dict:
    scores = state.get("ragas_scores") or {}
    message = SystemMessage(
        content=_FEEDBACK_TEMPLATE.format(
            faithfulness_score=scores.get("faithfulness") or 0.0,
            threshold=RAGAS_FAITHFULNESS_THRESHOLD,
            query=state.get("query", ""),
        )
    )
    return {
        "messages": [message],
        "ragas_rewrite_count": state.get("ragas_rewrite_count", 0) + 1,
    }
