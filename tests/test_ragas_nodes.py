"""Unit tests for RAGAS evaluator node, rewrite node, and graph routing.

All judge LLM calls and RAGAS metrics are mocked — no network required.
"""
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.nodes.ragas_evaluator import ragas_evaluator_node
from app.agents.nodes.ragas_rewrite import ragas_rewrite_node
from app.agents.react_agent import _route_after_ragas
from app.constants import MAX_RAGAS_REWRITES, RAGAS_FAITHFULNESS_THRESHOLD
from langgraph.graph import END


def _make_state(**overrides):
    state = {
        "messages": [
            HumanMessage(content="What is my HbA1c?"),
            AIMessage(content="Your most recent HbA1c is 5.4% from 2025-03-27."),
        ],
        "query": "What is my HbA1c?",
        "retrieved_docs": [Document(page_content="HbA1c: 5.4% on 2025-03-27")],
        "ragas_scores": None,
        "ragas_rewrite_count": 0,
    }
    state.update(overrides)
    return state


def _mock_metric(score=0.9):
    metric = MagicMock()
    metric.single_turn_ascore = AsyncMock(return_value=score)
    return metric


EVALUATOR_MODULE = "app.agents.nodes.ragas_evaluator"


class TestRagasEvaluatorNode:
    def _patch_all(self, faithfulness=0.9, relevance=0.8, precision=0.7):
        """Patch judge LLM, embeddings, and the three metric classes."""
        return (
            patch(f"{EVALUATOR_MODULE}.get_ragas_judge_llm", return_value=MagicMock()),
            patch(f"{EVALUATOR_MODULE}._get_embeddings", return_value=MagicMock()),
            patch(f"{EVALUATOR_MODULE}.Faithfulness", return_value=_mock_metric(faithfulness)),
            patch(f"{EVALUATOR_MODULE}.ResponseRelevancy", return_value=_mock_metric(relevance)),
            patch(f"{EVALUATOR_MODULE}.ContextPrecision", return_value=_mock_metric(precision)),
        )

    def test_ragas_evaluator_returns_scores_dict(self):
        # Arrange
        state = _make_state()
        p_judge, p_emb, p_faith, p_rel, p_prec = self._patch_all(0.9, 0.8, 0.7)

        # Act
        with p_judge, p_emb, p_faith, p_rel, p_prec:
            result = ragas_evaluator_node(state)

        # Assert
        assert result["ragas_scores"] == {
            "faithfulness": 0.9,
            "relevance": 0.8,
            "context_precision": 0.7,
        }

    def test_ragas_evaluator_with_no_ai_message_returns_zero_scores(self):
        # Arrange
        state = _make_state(messages=[HumanMessage(content="What is my HbA1c?")])
        p_judge, p_emb, p_faith, p_rel, p_prec = self._patch_all()

        # Act
        with p_judge, p_emb, p_faith, p_rel, p_prec:
            result = ragas_evaluator_node(state)

        # Assert
        assert result["ragas_scores"] == {
            "faithfulness": 0.0,
            "relevance": 0.0,
            "context_precision": 0.0,
        }

    def test_ragas_evaluator_ignores_ai_messages_with_tool_calls(self):
        # Arrange — last AI message has tool_calls; earlier AI message is the answer
        answer_msg = AIMessage(content="Your HbA1c is 5.4%.")
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"name": "labs_latest_value", "args": {"analyte": "HbA1c"}, "id": "tc1"}],
        )
        state = _make_state(
            messages=[HumanMessage(content="What is my HbA1c?"), answer_msg, tool_call_msg]
        )
        faith_metric = _mock_metric(0.9)
        with (
            patch(f"{EVALUATOR_MODULE}.get_ragas_judge_llm", return_value=MagicMock()),
            patch(f"{EVALUATOR_MODULE}._get_embeddings", return_value=MagicMock()),
            patch(f"{EVALUATOR_MODULE}.Faithfulness", return_value=faith_metric),
            patch(f"{EVALUATOR_MODULE}.ResponseRelevancy", return_value=_mock_metric(0.8)),
            patch(f"{EVALUATOR_MODULE}.ContextPrecision", return_value=_mock_metric(0.7)),
            patch(f"{EVALUATOR_MODULE}.SingleTurnSample") as mock_sample,
        ):
            # Act
            ragas_evaluator_node(state)

        # Assert — sample built from the non-tool-call answer
        assert mock_sample.call_args.kwargs["response"] == "Your HbA1c is 5.4%."

    def test_ragas_evaluator_handles_metric_exception_gracefully(self):
        # Arrange
        state = _make_state()
        failing_metric = MagicMock()
        failing_metric.single_turn_ascore = AsyncMock(side_effect=RuntimeError("judge down"))
        with (
            patch(f"{EVALUATOR_MODULE}.get_ragas_judge_llm", return_value=MagicMock()),
            patch(f"{EVALUATOR_MODULE}._get_embeddings", return_value=MagicMock()),
            patch(f"{EVALUATOR_MODULE}.Faithfulness", return_value=failing_metric),
            patch(f"{EVALUATOR_MODULE}.ResponseRelevancy", return_value=_mock_metric(0.8)),
            patch(f"{EVALUATOR_MODULE}.ContextPrecision", return_value=_mock_metric(0.7)),
        ):
            # Act
            result = ragas_evaluator_node(state)

        # Assert — failed metric is None, others still scored
        assert result["ragas_scores"]["faithfulness"] is None
        assert result["ragas_scores"]["relevance"] == 0.8
        assert result["ragas_scores"]["context_precision"] == 0.7

    def test_ragas_evaluator_deduplicates_retrieved_docs(self):
        # Arrange
        state = _make_state(
            retrieved_docs=[
                Document(page_content="HbA1c: 5.4%"),
                Document(page_content="HbA1c: 5.4%"),
                Document(page_content="ALT: 38 IU/L"),
            ]
        )
        with (
            patch(f"{EVALUATOR_MODULE}.get_ragas_judge_llm", return_value=MagicMock()),
            patch(f"{EVALUATOR_MODULE}._get_embeddings", return_value=MagicMock()),
            patch(f"{EVALUATOR_MODULE}.Faithfulness", return_value=_mock_metric()),
            patch(f"{EVALUATOR_MODULE}.ResponseRelevancy", return_value=_mock_metric()),
            patch(f"{EVALUATOR_MODULE}.ContextPrecision", return_value=_mock_metric()),
            patch(f"{EVALUATOR_MODULE}.SingleTurnSample") as mock_sample,
        ):
            # Act
            ragas_evaluator_node(state)

        # Assert
        contexts = mock_sample.call_args.kwargs["retrieved_contexts"]
        assert sorted(contexts) == ["ALT: 38 IU/L", "HbA1c: 5.4%"]

    def test_ragas_evaluator_clamps_scores_to_valid_range(self):
        # Arrange
        state = _make_state()
        p_judge, p_emb, p_faith, p_rel, p_prec = self._patch_all(
            faithfulness=1.5, relevance=-0.2, precision=0.5
        )

        # Act
        with p_judge, p_emb, p_faith, p_rel, p_prec:
            result = ragas_evaluator_node(state)

        # Assert
        assert result["ragas_scores"]["faithfulness"] == 1.0
        assert result["ragas_scores"]["relevance"] == 0.0
        assert result["ragas_scores"]["context_precision"] == 0.5

    def test_ragas_evaluator_uses_judge_model_env_var(self, monkeypatch):
        # Arrange
        monkeypatch.setenv("JUDGE_MODEL", "custom-judge:1b")
        monkeypatch.setenv("JUDGE_BASE_URL", "http://judge-host:11434")
        state = _make_state()
        with (
            patch("app.eval.judge.ChatOllama") as mock_chat,
            patch(f"{EVALUATOR_MODULE}._get_embeddings", return_value=MagicMock()),
            patch(f"{EVALUATOR_MODULE}.Faithfulness", return_value=_mock_metric()),
            patch(f"{EVALUATOR_MODULE}.ResponseRelevancy", return_value=_mock_metric()),
            patch(f"{EVALUATOR_MODULE}.ContextPrecision", return_value=_mock_metric()),
        ):
            # Act
            ragas_evaluator_node(state)

        # Assert
        mock_chat.assert_called_once_with(
            model="custom-judge:1b", base_url="http://judge-host:11434", temperature=0
        )


class TestRagasRewriteNode:
    def _rewrite_state(self, faithfulness=0.4, count=0):
        return _make_state(
            ragas_scores={"faithfulness": faithfulness, "relevance": 0.8, "context_precision": 0.7},
            ragas_rewrite_count=count,
        )

    def test_ragas_rewrite_increments_rewrite_count(self):
        # Arrange
        state = self._rewrite_state(count=0)

        # Act
        result = ragas_rewrite_node(state)

        # Assert
        assert result["ragas_rewrite_count"] == 1

    def test_ragas_rewrite_appends_system_message(self):
        # Arrange
        state = self._rewrite_state()

        # Act
        result = ragas_rewrite_node(state)

        # Assert
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], SystemMessage)

    def test_ragas_rewrite_message_includes_faithfulness_score(self):
        # Arrange
        state = self._rewrite_state(faithfulness=0.42)

        # Act
        result = ragas_rewrite_node(state)

        # Assert
        assert "0.42" in result["messages"][0].content

    def test_ragas_rewrite_message_includes_original_query(self):
        # Arrange
        state = self._rewrite_state()

        # Act
        result = ragas_rewrite_node(state)

        # Assert
        assert "What is my HbA1c?" in result["messages"][0].content

    def test_ragas_rewrite_instructs_agent_to_retrieve(self):
        # Arrange
        state = self._rewrite_state()

        # Act
        result = ragas_rewrite_node(state)

        # Assert
        assert "retrieve_documents" in result["messages"][0].content


class TestRagasRouting:
    def test_route_after_ragas_low_faithfulness_triggers_rewrite(self):
        # Arrange
        state = {
            "ragas_scores": {"faithfulness": RAGAS_FAITHFULNESS_THRESHOLD - 0.1},
            "ragas_rewrite_count": 0,
        }

        # Act / Assert
        assert _route_after_ragas(state) == "ragas_rewrite"

    def test_route_after_ragas_high_faithfulness_routes_to_end(self):
        # Arrange
        state = {
            "ragas_scores": {"faithfulness": RAGAS_FAITHFULNESS_THRESHOLD + 0.1},
            "ragas_rewrite_count": 0,
        }

        # Act / Assert
        assert _route_after_ragas(state) == END

    def test_route_after_ragas_exhausted_budget_routes_to_end(self):
        # Arrange
        state = {
            "ragas_scores": {"faithfulness": 0.0},
            "ragas_rewrite_count": MAX_RAGAS_REWRITES,
        }

        # Act / Assert
        assert _route_after_ragas(state) == END

    def test_route_after_ragas_none_faithfulness_logs_warning_and_routes_to_end(self, caplog):
        # Arrange
        state = {
            "ragas_scores": {"faithfulness": None, "relevance": 0.8, "context_precision": 0.7},
            "ragas_rewrite_count": 0,
        }

        # Act
        with caplog.at_level(logging.WARNING, logger="app.agents.react_agent"):
            route = _route_after_ragas(state)

        # Assert
        assert route == END
        assert any(
            "faithfulness" in rec.message.lower() and rec.levelno == logging.WARNING
            for rec in caplog.records
        )

    def test_route_after_ragas_missing_scores_routes_to_end(self):
        # Arrange
        state = {"ragas_scores": None, "ragas_rewrite_count": 0}

        # Act / Assert
        assert _route_after_ragas(state) == END
