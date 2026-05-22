"""
TDD red phase: tests for the LangGraph ReAct agent.
All imports reference modules that do not yet exist — ImportError is expected
until implementation is complete.
"""

import os
import operator
import logging
import inspect
import pytest
from unittest.mock import MagicMock, patch
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.tools import StructuredTool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.documents import Document
from langgraph.prebuilt import ToolNode
from langgraph.graph import END

try:
    from langgraph.errors import GraphRecursionError
except ImportError:
    GraphRecursionError = RecursionError

load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

from app.agents.react_agent import (
    build_react_agent,
    answer_with_react_agent,
    AgentState,
    _route_after_agent,
    _history_to_messages,
    NODE_TOOLS,
    ALL_TOOLS,
    retrieve_documents,
    labs_panel,
    whoop_recent,
    # existing tools:
    labs_list_analytes,
    labs_latest_value,
    labs_history,
    labs_summary,
    labs_value_on_date,
    meds_timeline,
    meds_history,
    meds_dosage_on_date,
    meds_list_medications,
    meds_list_current,
    whoop_sleeps_on_date,
    whoop_recovery_metrics_on_date,
    whoop_workouts_on_date,
)
from app.agents.nodes.retriever import _retriever_node
from app.tools.retrieval_tool import has_vectorstore
from app.constants import AGENT_MAX_ITERATIONS

# New imports for grader/rewrite — protected so existing tests survive the red phase
try:
    from app.agents.react_agent import _route_after_grader
    from app.agents.nodes.grader import build_grader_node
    from app.agents.nodes.rewrite import build_rewrite_node
    from app.constants import MAX_REWRITES
    _GRADER_AVAILABLE = True
except ImportError:
    _route_after_grader = None  # type: ignore
    build_grader_node = None    # type: ignore
    build_rewrite_node = None   # type: ignore
    MAX_REWRITES = None         # type: ignore
    _GRADER_AVAILABLE = False


# ---------------------------------------------------------------------------
# TestGraphStructure
# ---------------------------------------------------------------------------


class TestGraphStructure:
    def test_graph_has_agent_tools_retriever_nodes(self):
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        agent = build_react_agent(llm)
        node_keys = set(agent.get_graph().nodes.keys())
        assert "agent" in node_keys
        assert "tools" in node_keys
        assert "retriever" in node_keys

    def test_tools_node_excludes_retrieve_documents(self):
        tool_names = [t.name for t in NODE_TOOLS]
        assert "retrieve_documents" not in tool_names
        assert len(tool_names) == 15

    def test_retrieve_documents_in_all_tools(self):
        tool_names = [t.name for t in ALL_TOOLS]
        assert "retrieve_documents" in tool_names
        assert len(tool_names) == 16

    def test_all_tools_bound_to_llm(self):
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        # Should not raise
        bound = llm.bind_tools(ALL_TOOLS)
        assert bound is not None


# ---------------------------------------------------------------------------
# TestAgentState
# ---------------------------------------------------------------------------


class TestAgentState:
    def test_state_has_messages_key(self):
        import typing
        hints = typing.get_type_hints(AgentState, include_extras=True)
        assert "messages" in hints

    def test_state_has_sources_key(self):
        import typing
        hints = typing.get_type_hints(AgentState, include_extras=True)
        assert "sources" in hints

    def test_state_sources_reducer_is_add(self):
        import typing
        hints = typing.get_type_hints(AgentState, include_extras=True)
        sources_hint = hints["sources"]
        # Annotated[list[str], operator.add] — metadata[0] is operator.add
        metadata = getattr(sources_hint, "__metadata__", None)
        assert metadata is not None, "sources annotation has no __metadata__"
        assert metadata[0] is operator.add


# ---------------------------------------------------------------------------
# TestRouteAfterAgent
# ---------------------------------------------------------------------------


class TestRouteAfterAgent:
    def test_routes_to_grader_when_no_tool_calls(self):
        msg = AIMessage(content="done")
        state = {"messages": [msg], "sources": []}
        result = _route_after_agent(state)
        assert result == "grader"

    def test_routes_to_tools_for_standard_tool_call(self):
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "labs_latest_value", "args": {}, "id": "x", "type": "tool_call"}
            ],
        )
        state = {"messages": [msg], "sources": []}
        result = _route_after_agent(state)
        assert result == "tools"

    def test_routes_to_retriever_for_retrieve_documents_call(self):
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "retrieve_documents",
                    "args": {"query": "test"},
                    "id": "x",
                    "type": "tool_call",
                }
            ],
        )
        state = {"messages": [msg], "sources": []}
        result = _route_after_agent(state)
        assert result == "retriever"

    def test_routes_to_retriever_when_both_categories_present(self):
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "retrieve_documents",
                    "args": {"query": "test"},
                    "id": "call_1",
                    "type": "tool_call",
                },
                {
                    "name": "labs_latest_value",
                    "args": {},
                    "id": "call_2",
                    "type": "tool_call",
                },
            ],
        )
        state = {"messages": [msg], "sources": []}
        result = _route_after_agent(state)
        assert result == "retriever"


# ---------------------------------------------------------------------------
# TestRetrieverNode
# ---------------------------------------------------------------------------


class TestRetrieverNode:
    def _make_retrieve_state(self, query="test query", source_type=None):
        args = {"query": query}
        if source_type is not None:
            args["source_type"] = source_type
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "retrieve_documents",
                    "args": args,
                    "id": "call_ret",
                    "type": "tool_call",
                }
            ],
        )
        return {"messages": [msg], "sources": []}

    def test_retriever_returns_tool_message_for_retrieve_call(self, monkeypatch):
        monkeypatch.setattr("app.agents.nodes.retriever._retrieve", lambda **kwargs: [])
        monkeypatch.setattr("app.agents.nodes.retriever._retrieve", lambda **kwargs: [])
        state = self._make_retrieve_state()
        result = _retriever_node(state)
        assert "messages" in result
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) > 0
        assert isinstance(result["messages"][0], ToolMessage)
        assert "sources" in result
        assert isinstance(result["sources"], list)

    def test_retriever_passes_query_to_retrieve(self, monkeypatch):
        calls = []

        def mock_retrieve(**kwargs):
            calls.append(kwargs)
            return []

        monkeypatch.setattr("app.agents.nodes.retriever._retrieve", mock_retrieve)
        monkeypatch.setattr("app.agents.nodes.retriever._retrieve", mock_retrieve)
        state = self._make_retrieve_state(query="my specific query")
        _retriever_node(state)
        assert len(calls) >= 1
        assert calls[0].get("query") == "my specific query"

    def test_retriever_passes_source_type_to_retrieve(self, monkeypatch):
        calls = []

        def mock_retrieve(**kwargs):
            calls.append(kwargs)
            return []

        monkeypatch.setattr("app.agents.nodes.retriever._retrieve", mock_retrieve)
        monkeypatch.setattr("app.agents.nodes.retriever._retrieve", mock_retrieve)
        state = self._make_retrieve_state(query="labs test", source_type="labs")
        _retriever_node(state)
        assert len(calls) >= 1
        assert calls[0].get("source_type") == "labs"

    def test_retriever_handles_no_docs_gracefully(self, monkeypatch):
        monkeypatch.setattr("app.agents.nodes.retriever._retrieve", lambda **kwargs: [])
        monkeypatch.setattr("app.agents.nodes.retriever._retrieve", lambda **kwargs: [])
        state = self._make_retrieve_state()
        result = _retriever_node(state)
        tool_msg = result["messages"][0]
        assert isinstance(tool_msg.content, str)
        assert len(tool_msg.content) > 0
        assert result["sources"] == []

    def test_retriever_extracts_sources_from_docs(self, monkeypatch):
        doc = Document(
            page_content="some lab content",
            metadata={"source": "test.pdf", "page": 1},
        )

        monkeypatch.setattr("app.agents.nodes.retriever._retrieve", lambda **kwargs: [doc])
        monkeypatch.setattr("app.agents.nodes.retriever._retrieve", lambda **kwargs: [doc])
        state = self._make_retrieve_state()
        result = _retriever_node(state)
        assert "sources" in result
        sources = result["sources"]
        assert any("test.pdf" in s for s in sources)
        assert any("1" in s for s in sources)


# ---------------------------------------------------------------------------
# TestNewTools
# ---------------------------------------------------------------------------


class TestNewTools:
    def test_labs_panel_is_structured_tool(self):
        assert isinstance(labs_panel, StructuredTool)

    def test_whoop_recent_is_structured_tool(self):
        assert isinstance(whoop_recent, StructuredTool)

    def test_retrieve_documents_is_structured_tool(self):
        assert isinstance(retrieve_documents, StructuredTool)

    def test_whoop_recent_accepts_days_arg(self):
        assert "days" in whoop_recent.args

    def test_retrieve_documents_accepts_source_type(self):
        assert "source_type" in retrieve_documents.args

    def test_retrieve_documents_accepts_query(self):
        assert "query" in retrieve_documents.args


# ---------------------------------------------------------------------------
# TestHistoryToMessages
# ---------------------------------------------------------------------------


class TestHistoryToMessages:
    def test_converts_user_role_to_human_message(self):
        result = _history_to_messages([{"role": "user", "content": "hi"}])
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "hi"

    def test_converts_assistant_role_to_ai_message(self):
        result = _history_to_messages([{"role": "assistant", "content": "hello"}])
        assert len(result) == 1
        assert isinstance(result[0], AIMessage)
        assert result[0].content == "hello"

    def test_ignores_unknown_role(self):
        result = _history_to_messages([{"role": "system", "content": "x"}])
        assert result == []

    def test_preserves_order(self):
        history = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ]
        result = _history_to_messages(history)
        assert len(result) == 3
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "first"
        assert isinstance(result[1], AIMessage)
        assert result[1].content == "second"
        assert isinstance(result[2], HumanMessage)
        assert result[2].content == "third"


# ---------------------------------------------------------------------------
# TestAnswerWithReactAgent
# ---------------------------------------------------------------------------


class TestAnswerWithReactAgent:
    def test_returns_tuple_of_str_and_list(self, monkeypatch):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="answer")],
            "sources": ["src1"],
        }
        monkeypatch.setattr(
            "app.agents.react_agent.build_react_agent", lambda llm: mock_agent
        )
        llm = MagicMock()
        result = answer_with_react_agent(llm, "question", [])
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == "answer"
        assert result[1] == ["src1"]

    def test_system_prompt_prepended(self, monkeypatch):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="answer")],
            "sources": [],
        }
        monkeypatch.setattr(
            "app.agents.react_agent.build_react_agent", lambda llm: mock_agent
        )
        llm = MagicMock()
        answer_with_react_agent(llm, "question", [])
        call_args = mock_agent.invoke.call_args
        # First positional arg is the state dict
        state = call_args[0][0] if call_args[0] else call_args.args[0]
        messages = state["messages"]
        assert len(messages) > 0
        assert isinstance(messages[0], SystemMessage)

    def test_history_in_initial_messages(self, monkeypatch):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="answer")],
            "sources": [],
        }
        monkeypatch.setattr(
            "app.agents.react_agent.build_react_agent", lambda llm: mock_agent
        )
        llm = MagicMock()
        history = [{"role": "user", "content": "prior"}]
        answer_with_react_agent(llm, "new question", history)
        call_args = mock_agent.invoke.call_args
        state = call_args[0][0] if call_args[0] else call_args.args[0]
        messages = state["messages"]
        contents = [m.content for m in messages]
        assert "prior" in contents

    def test_question_as_last_human_message(self, monkeypatch):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="answer")],
            "sources": [],
        }
        monkeypatch.setattr(
            "app.agents.react_agent.build_react_agent", lambda llm: mock_agent
        )
        llm = MagicMock()
        answer_with_react_agent(llm, "the question", [])
        call_args = mock_agent.invoke.call_args
        state = call_args[0][0] if call_args[0] else call_args.args[0]
        messages = state["messages"]
        last_msg = messages[-1]
        assert isinstance(last_msg, HumanMessage)
        assert last_msg.content == "the question"

    def test_recursion_limit_in_config(self, monkeypatch):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="answer")],
            "sources": [],
        }
        monkeypatch.setattr(
            "app.agents.react_agent.build_react_agent", lambda llm: mock_agent
        )
        llm = MagicMock()
        answer_with_react_agent(llm, "question", [])
        call_args = mock_agent.invoke.call_args
        # config is passed as a keyword argument
        kwargs = call_args[1] if call_args[1] else call_args.kwargs
        assert "config" in kwargs
        assert kwargs["config"]["recursion_limit"] == AGENT_MAX_ITERATIONS

    def test_graceful_fallback_on_recursion_error(self, monkeypatch):
        mock_agent = MagicMock()
        mock_agent.invoke.side_effect = GraphRecursionError("exceeded")
        monkeypatch.setattr(
            "app.agents.react_agent.build_react_agent", lambda llm: mock_agent
        )
        llm = MagicMock()
        result = answer_with_react_agent(llm, "question", [])
        assert isinstance(result, tuple)
        assert len(result) == 2
        answer_text, sources = result
        assert isinstance(answer_text, str)
        assert isinstance(sources, list)
        lower = answer_text.lower()
        assert "wasn't able" in lower or "reasoning steps" in lower


# ---------------------------------------------------------------------------
# TestChatIntegration
# ---------------------------------------------------------------------------


class TestChatIntegration:
    def test_answer_question_has_correct_signature(self):
        from app.chains.chat import answer_question
        sig = inspect.signature(answer_question)
        assert "question" in sig.parameters
        assert "history" in sig.parameters

    def test_chat_does_not_define_retrieve_internally(self):
        import app.chains.chat as c
        assert not hasattr(c, "_retrieve")

    def test_chat_does_not_define_build_context(self):
        import app.chains.chat as c
        assert not hasattr(c, "_build_context")

    def test_has_vectorstore_importable_from_chat(self):
        from app.chains.chat import has_vectorstore
        assert callable(has_vectorstore)


# ---------------------------------------------------------------------------
# TestAgentStateExpanded
# ---------------------------------------------------------------------------


class TestAgentStateExpanded:
    def test_state_has_grader_verdict_key(self):
        import typing
        hints = typing.get_type_hints(AgentState, include_extras=True)
        assert "grader_verdict" in hints

    def test_state_has_grader_feedback_key(self):
        import typing
        hints = typing.get_type_hints(AgentState, include_extras=True)
        assert "grader_feedback" in hints

    def test_state_has_rewrite_count_key(self):
        import typing
        hints = typing.get_type_hints(AgentState, include_extras=True)
        assert "rewrite_count" in hints


# ---------------------------------------------------------------------------
# TestGraphStructureExpanded
# ---------------------------------------------------------------------------


class TestGraphStructureExpanded:
    def test_graph_has_grader_node(self):
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        agent = build_react_agent(llm)
        node_keys = set(agent.get_graph().nodes.keys())
        assert "grader" in node_keys

    def test_graph_has_rewrite_node(self):
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        agent = build_react_agent(llm)
        node_keys = set(agent.get_graph().nodes.keys())
        assert "rewrite" in node_keys


# ---------------------------------------------------------------------------
# TestRouteAfterGrader
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _GRADER_AVAILABLE, reason="grader not yet implemented")
class TestRouteAfterGrader:
    def _make_state(self, verdict: str, rewrite_count: int) -> dict:
        return {
            "messages": [AIMessage(content="answer")],
            "sources": [],
            "grader_verdict": verdict,
            "grader_feedback": "some feedback",
            "rewrite_count": rewrite_count,
        }

    def test_pass_verdict_routes_to_end(self):
        state = self._make_state("pass", 0)
        assert _route_after_grader(state) == END

    def test_fail_with_count_zero_routes_to_rewrite(self):
        state = self._make_state("fail", 0)
        assert _route_after_grader(state) == "rewrite"

    def test_fail_with_count_one_routes_to_rewrite(self):
        state = self._make_state("fail", 1)
        assert _route_after_grader(state) == "rewrite"

    def test_fail_at_cap_routes_to_end(self):
        state = self._make_state("fail", MAX_REWRITES)
        assert _route_after_grader(state) == END

    def test_fail_over_cap_routes_to_end(self):
        state = self._make_state("fail", MAX_REWRITES + 1)
        assert _route_after_grader(state) == END


# ---------------------------------------------------------------------------
# TestGraderNode
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _GRADER_AVAILABLE, reason="grader not yet implemented")
class TestGraderNode:
    def _make_state(self, messages: list) -> dict:
        return {
            "messages": messages,
            "sources": [],
            "grader_verdict": "",
            "grader_feedback": "",
            "rewrite_count": 0,
        }

    def _mock_llm(self, verdict: str, reason: str = "ok"):
        mock_result = MagicMock()
        mock_result.verdict = verdict
        mock_result.reason = reason
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_result
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_chain
        return mock_llm, mock_chain

    def test_returns_pass_for_relevant_complete_answer(self):
        mock_llm, _ = self._mock_llm("pass", "Relevant and complete.")
        grader = build_grader_node(mock_llm)
        state = self._make_state([
            HumanMessage(content="What is my latest HRV?"),
            AIMessage(content="Your latest HRV is 45ms as of 2024-01-15."),
        ])
        result = grader(state)
        assert result["grader_verdict"] == "pass"

    def test_returns_fail_with_reason_for_off_topic_answer(self):
        mock_llm, _ = self._mock_llm("fail", "Answer does not address HRV.")
        grader = build_grader_node(mock_llm)
        state = self._make_state([
            HumanMessage(content="What is my latest HRV?"),
            AIMessage(content="The weather today is sunny."),
        ])
        result = grader(state)
        assert result["grader_verdict"] == "fail"
        assert len(result["grader_feedback"]) > 0

    def test_returns_fail_for_incomplete_answer(self):
        mock_llm, _ = self._mock_llm("fail", "Missing sleep data for the second part of the question.")
        grader = build_grader_node(mock_llm)
        state = self._make_state([
            HumanMessage(content="What is my HRV and how did I sleep last night?"),
            AIMessage(content="Your HRV is 45ms."),
        ])
        result = grader(state)
        assert result["grader_verdict"] == "fail"

    def test_uses_last_human_message_not_first(self):
        mock_llm, mock_chain = self._mock_llm("pass")
        grader = build_grader_node(mock_llm)
        state = self._make_state([
            HumanMessage(content="FIRST QUESTION"),
            AIMessage(content="first answer"),
            HumanMessage(content="LAST QUESTION"),
            AIMessage(content="last answer"),
        ])
        grader(state)
        invoke_args = mock_chain.invoke.call_args
        messages_sent = invoke_args[0][0]
        combined = " ".join(str(m.content) for m in messages_sent)
        assert "LAST QUESTION" in combined

    def test_uses_last_ai_message_as_candidate(self):
        mock_llm, mock_chain = self._mock_llm("pass")
        grader = build_grader_node(mock_llm)
        state = self._make_state([
            HumanMessage(content="What is my HRV?"),
            AIMessage(content="FIRST ANSWER"),
            HumanMessage(content="What is my HRV?"),
            AIMessage(content="LAST ANSWER"),
        ])
        grader(state)
        invoke_args = mock_chain.invoke.call_args
        messages_sent = invoke_args[0][0]
        combined = " ".join(str(m.content) for m in messages_sent)
        assert "LAST ANSWER" in combined


# ---------------------------------------------------------------------------
# TestRewriteNode
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _GRADER_AVAILABLE, reason="grader not yet implemented")
class TestRewriteNode:
    def _make_state(self, feedback: str = "Missing dates.", count: int = 0) -> dict:
        return {
            "messages": [
                HumanMessage(content="What is my HRV?"),
                AIMessage(content="Your HRV is high."),
            ],
            "sources": [],
            "grader_verdict": "fail",
            "grader_feedback": feedback,
            "rewrite_count": count,
        }

    def test_returns_dict_with_messages_key(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="rewritten answer")
        rewrite = build_rewrite_node(mock_llm)
        result = rewrite(self._make_state())
        assert "messages" in result
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) > 0
        assert isinstance(result["messages"][0], AIMessage)

    def test_increments_rewrite_count(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="rewritten")
        rewrite = build_rewrite_node(mock_llm)
        result = rewrite(self._make_state(count=1))
        assert result["rewrite_count"] == 2

    def test_injects_grader_feedback_into_prompt(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="rewritten")
        rewrite = build_rewrite_node(mock_llm)
        rewrite(self._make_state(feedback="Answer did not cite specific HRV values."))
        invoke_args = mock_llm.invoke.call_args
        messages_sent = invoke_args[0][0]
        combined = " ".join(str(m.content) for m in messages_sent)
        assert "Answer did not cite specific HRV values." in combined

    def test_returned_ai_message_has_no_tool_calls(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="clean rewritten answer")
        rewrite = build_rewrite_node(mock_llm)
        result = rewrite(self._make_state())
        returned_msg = result["messages"][0]
        tool_calls = getattr(returned_msg, "tool_calls", None)
        assert not tool_calls
