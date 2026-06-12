import asyncio
import logging

from langchain_core.messages import AIMessage

from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, ResponseRelevancy
# Classic ContextPrecision requires a ground-truth reference, which doesn't exist
# during a live graph run — the reference-free variant scores retrieved contexts
# against the generated answer instead.
from ragas.metrics import LLMContextPrecisionWithoutReference as ContextPrecision

from app.eval.judge import get_ragas_judge_llm
from app.tools.retrieval_tool import _get_embeddings

logger = logging.getLogger(__name__)

_ZERO_SCORES = {"faithfulness": 0.0, "relevance": 0.0, "context_precision": 0.0}


def _extract_answer(messages: list) -> str:
    for msg in reversed(messages):
        if (
            isinstance(msg, AIMessage)
            and not getattr(msg, "tool_calls", None)
            and isinstance(msg.content, str)
            and msg.content.strip()
        ):
            return msg.content
    return ""


def _dedup_contexts(docs: list) -> list[str]:
    seen = set()
    contexts = []
    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            contexts.append(doc.page_content)
    return contexts


def _clamp(score):
    if score is None:
        return None
    return max(0.0, min(1.0, float(score)))


async def _score_all(metrics: dict, sample) -> dict:
    scores = {}
    for name, metric in metrics.items():
        try:
            scores[name] = _clamp(await metric.single_turn_ascore(sample))
        except Exception:
            logger.exception("RAGAS metric '%s' failed", name)
            scores[name] = None
    return scores


def ragas_evaluator_node(state: dict) -> dict:
    answer = _extract_answer(state["messages"])
    if not answer:
        return {"ragas_scores": dict(_ZERO_SCORES)}

    sample = SingleTurnSample(
        user_input=state.get("query", ""),
        response=answer,
        retrieved_contexts=_dedup_contexts(state.get("retrieved_docs") or []),
    )

    # format="json": every RAGAS prompt expects JSON; small instruct judges
    # otherwise drift into prose (seen with context_precision) and fail parsing.
    judge = LangchainLLMWrapper(get_ragas_judge_llm().model_copy(update={"format": "json"}))
    embeddings = LangchainEmbeddingsWrapper(_get_embeddings())
    metrics = {
        "faithfulness": Faithfulness(llm=judge),
        "relevance": ResponseRelevancy(llm=judge, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=judge),
    }

    return {"ragas_scores": asyncio.run(_score_all(metrics, sample))}
