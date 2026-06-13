"""Offline RAGAS evaluation suite against the curated 15-case dataset.

Runs the real agent + judge LLM via Ollama. Skipped unless --run-eval is passed.

Entries with non-null expected_values are scored with AnswerCorrectness against a
constructed ground-truth string. Entries with null expected_values are validated
via topic coverage (>=50% of expected_topics present in the answer).
"""
import asyncio
import json
import os
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from app.agents.react_agent import build_react_agent
from app.chains.prompts import SYSTEM_BASE
from app.constants import AGENT_MAX_ITERATIONS, RAGAS_ANSWER_CORRECTNESS_THRESHOLD

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

DATASET_PATH = Path(__file__).parent / "fixtures" / "ragas_dataset.json"

with open(DATASET_PATH) as f:
    DATASET = json.load(f)

CORRECTNESS_CASES = [c for c in DATASET if c["expected_values"] is not None]
TOPIC_CASES = [c for c in DATASET if c["expected_values"] is None]


@pytest.fixture(autouse=True, scope="module")
def _require_run_eval(request):
    if not request.config.getoption("--run-eval"):
        pytest.skip("evaluation tests skipped; pass --run-eval to enable")


@pytest.fixture(scope="module")
def agent():
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    return build_react_agent(llm)


def _run_agent(agent, query: str) -> dict:
    return agent.invoke(
        {
            "messages": [SystemMessage(content=SYSTEM_BASE), HumanMessage(content=query)],
            "sources": [],
            "retrieved_docs": [],
            "tool_outputs": [],
            "tool_validation_errors": [],
            "query": query,
            "grader_verdict": "",
            "grader_feedback": "",
            "rewrite_count": 0,
            "iteration_count": 0,
            "ragas_scores": None,
            "ragas_rewrite_count": 0,
        },
        config={"recursion_limit": AGENT_MAX_ITERATIONS * 3},
    )


def _ground_truth(expected_values: dict) -> str:
    key = expected_values["key"]
    value = expected_values["value"]
    unit = expected_values["unit"]
    date = expected_values["date"]
    return f"{key}: {value} {unit or ''} (date: {date})".strip()


@pytest.mark.parametrize("case", CORRECTNESS_CASES, ids=[c["id"] for c in CORRECTNESS_CASES])
def test_rag_eval_answer_correctness(agent, case):
    # Arrange
    from ragas.dataset_schema import SingleTurnSample
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import AnswerCorrectness, AnswerSimilarity

    from app.eval.judge import get_ragas_judge_llm
    from app.tools.retrieval_tool import _get_embeddings

    final_state = _run_agent(agent, case["query"])
    answer = final_state["messages"][-1].content
    assert isinstance(answer, str) and answer.strip(), "agent returned empty answer"

    sample = SingleTurnSample(
        user_input=case["query"],
        response=answer,
        reference=_ground_truth(case["expected_values"]),
    )
    # format="json": RAGAS statement-extraction prompts expect JSON; small
    # instruct judges drift into prose otherwise (same as evaluator node).
    judge = LangchainLLMWrapper(
        get_ragas_judge_llm().model_copy(update={"format": "json"})
    )
    embeddings = LangchainEmbeddingsWrapper(_get_embeddings())
    # answer_similarity must be set explicitly: ragas only auto-builds it
    # inside evaluate()'s init, not when calling single_turn_ascore directly.
    metric = AnswerCorrectness(
        llm=judge,
        embeddings=embeddings,
        answer_similarity=AnswerSimilarity(embeddings=embeddings),
    )

    # Act
    score = asyncio.run(metric.single_turn_ascore(sample))

    # Assert
    assert score >= RAGAS_ANSWER_CORRECTNESS_THRESHOLD, (
        f"{case['id']}: AnswerCorrectness {score:.2f} < {RAGAS_ANSWER_CORRECTNESS_THRESHOLD}\n"
        f"Query: {case['query']}\nGround truth: {_ground_truth(case['expected_values'])}\n"
        f"Answer: {answer}"
    )


@pytest.mark.parametrize("case", TOPIC_CASES, ids=[c["id"] for c in TOPIC_CASES])
def test_rag_eval_topic_coverage(agent, case):
    # Arrange
    final_state = _run_agent(agent, case["query"])
    answer = final_state["messages"][-1].content
    assert isinstance(answer, str) and answer.strip(), "agent returned empty answer"

    # Act
    answer_lower = answer.lower()
    hits = [t for t in case["expected_topics"] if t.lower() in answer_lower]

    # Assert — majority of expected topics must be present
    required = len(case["expected_topics"]) / 2
    assert len(hits) >= required, (
        f"{case['id']}: only {len(hits)}/{len(case['expected_topics'])} topics covered "
        f"(need >={required:.1f})\nMissing: {set(case['expected_topics']) - set(hits)}\n"
        f"Query: {case['query']}\nAnswer: {answer}"
    )
