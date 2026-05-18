"""Tests for LangGraph ReAct agent build and flow."""

import os
import inspect
import pytest
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool

load_dotenv()

from app.tools.react_agent import (
    build_react_agent,
    answer_with_react_agent,
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

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# All 13 tools that should be in the agent
ALL_TOOLS = [
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
]


class TestReActAgentBuild:
    """Verify ReAct agent builds correctly with all tools."""

    def test_agent_builds_successfully(self):
        """Agent instantiates without errors."""
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        agent = build_react_agent(llm)
        assert agent is not None

    def test_agent_has_invoke_method(self):
        """Agent has invoke method for execution."""
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        agent = build_react_agent(llm)
        assert callable(agent.invoke)

    def test_all_13_tools_are_structured_tools(self):
        """All 13 tools are properly decorated as StructuredTool."""
        for tool in ALL_TOOLS:
            assert isinstance(tool, StructuredTool), f"{tool.name} is not a StructuredTool"

    def test_all_tools_have_names(self):
        """All tools have unique, meaningful names."""
        names = [tool.name for tool in ALL_TOOLS]
        assert len(names) == 13
        assert len(set(names)) == 13, "Tool names not unique"
        for name in names:
            assert len(name) > 0
            assert not name.startswith("_")


class TestReActAgentFlow:
    """Verify ReAct agent can be invoked and returns responses."""

    def test_agent_accepts_and_processes_messages(self):
        """Agent accepts messages and returns result dict with messages."""
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        agent = build_react_agent(llm)

        messages = [{"role": "user", "content": "What medications are available?"}]
        result = agent.invoke({"messages": messages})

        assert result is not None
        assert "messages" in result
        assert len(result["messages"]) > 0

    def test_answer_with_react_agent_formats_prompt_to_messages(self):
        """answer_with_react_agent converts ChatPromptTemplate to messages for agent."""
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a health data assistant."),
            ("human", "What labs are in the database?")
        ])

        response = answer_with_react_agent(llm, prompt)
        assert isinstance(response, str)
        assert len(response) > 0

    def test_answer_with_react_agent_handles_errors_gracefully(self):
        """answer_with_react_agent returns error message instead of crashing."""
        llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)

        # Create a prompt that might cause issues
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a health data assistant."),
            ("human", "")  # Empty question
        ])

        response = answer_with_react_agent(llm, prompt)
        assert isinstance(response, str)


class TestToolProperties:
    """Verify tools are properly defined."""

    def test_all_tools_have_docstrings(self):
        """All tools have docstrings describing their purpose."""
        for tool in ALL_TOOLS:
            assert tool.description is not None
            assert len(tool.description) > 0

    def test_tool_descriptions_are_concise(self):
        """Tool descriptions are brief (under 150 chars)."""
        for tool in ALL_TOOLS:
            desc_len = len(tool.description)
            assert desc_len < 150, f"{tool.name} description too long: {desc_len} chars"

    def test_tools_have_arguments(self):
        """Tools have documented arguments (except those with no params)."""
        # Some tools take no arguments, that's ok
        for tool in ALL_TOOLS:
            # Each tool should have either no args or documented args
            assert hasattr(tool, "args_schema")

    def test_lab_tools_distinct_from_med_tools(self):
        """Lab tools are named distinctly from medication tools."""
        tool_names = [t.name for t in ALL_TOOLS]
        lab_names = [n for n in tool_names if "labs_" in n]
        med_names = [n for n in tool_names if "meds_" in n]
        whoop_names = [n for n in tool_names if "whoop_" in n]

        assert len(lab_names) == 5, "Should have 5 lab tools"
        assert len(med_names) == 5, "Should have 5 medication tools"
        assert len(whoop_names) == 3, "Should have 3 WHOOP tools"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
