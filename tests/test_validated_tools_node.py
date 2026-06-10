"""
TDD red phase: tests for ValidatedToolNode in app/agents/nodes/validated_tools_node.py.
All tests will fail (ImportError) until implementation is complete.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, ValidationError

try:
    from app.agents.nodes.validated_tools_node import ValidatedToolNode
    _NODE_AVAILABLE = True
except ImportError:
    ValidatedToolNode = None  # type: ignore
    _NODE_AVAILABLE = False

try:
    from app.constants import MAX_TOOL_VALIDATION_RETRIES
    _CONSTANTS_AVAILABLE = True
except (ImportError, AttributeError):
    MAX_TOOL_VALIDATION_RETRIES = None  # type: ignore
    _CONSTANTS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _NODE_AVAILABLE,
    reason="app/agents/nodes/validated_tools_node.py not yet implemented",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_call(name: str, args: dict, call_id: str = "call_001"):
    return {"name": name, "args": args, "id": call_id}


def _make_ai_message(*tool_calls):
    msg = AIMessage(content="")
    msg.tool_calls = list(tool_calls)
    return msg


def _make_state(tool_calls=None, tool_validation_errors=None):
    return {
        "messages": [_make_ai_message(*(tool_calls or []))],
        "tool_validation_errors": tool_validation_errors or [],
    }


def _make_mock_tool(name: str, return_value=None):
    mock = MagicMock()
    mock.name = name
    mock.invoke.return_value = return_value if return_value is not None else {"result": "ok"}
    return mock


# ---------------------------------------------------------------------------
# Input Validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_valid_args_invokes_tool(self):
        mock_tool = _make_mock_tool("labs_latest_value", {"analyte": "Glucose", "value": 90.0})
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        tc = _make_tool_call("labs_latest_value", {"analyte": "Glucose"})
        node(_make_state([tc]))
        mock_tool.invoke.assert_called_once()

    def test_invalid_args_returns_error_tool_message(self):
        mock_tool = _make_mock_tool("labs_latest_value")
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        tc = _make_tool_call("labs_latest_value", {"analyte": ""})  # empty analyte
        result = node(_make_state([tc]))
        messages = result["messages"]
        assert len(messages) == 1
        assert isinstance(messages[0], ToolMessage)
        assert "INPUT_VALIDATION_ERROR" in messages[0].content

    def test_tool_not_called_on_input_error(self):
        mock_tool = _make_mock_tool("labs_latest_value")
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        tc = _make_tool_call("labs_latest_value", {"analyte": ""})
        node(_make_state([tc]))
        mock_tool.invoke.assert_not_called()

    def test_error_message_contains_tool_name(self):
        mock_tool = _make_mock_tool("labs_latest_value")
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        tc = _make_tool_call("labs_latest_value", {"analyte": ""})
        result = node(_make_state([tc]))
        assert "labs_latest_value" in result["messages"][0].content

    def test_error_message_tool_call_id_matches(self):
        mock_tool = _make_mock_tool("labs_latest_value")
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        tc = _make_tool_call("labs_latest_value", {"analyte": ""}, call_id="xyz_999")
        result = node(_make_state([tc]))
        assert result["messages"][0].tool_call_id == "xyz_999"

    def test_multiple_calls_validated_independently(self):
        mock_a = _make_mock_tool("labs_latest_value", {"analyte": "A", "value": 1.0})
        mock_b = _make_mock_tool("labs_summary")
        node = ValidatedToolNode({
            "labs_latest_value": mock_a,
            "labs_summary": mock_b,
        })
        tc_good = _make_tool_call("labs_latest_value", {"analyte": "Glucose"}, call_id="c1")
        tc_bad = _make_tool_call("labs_summary", {"analyte": ""}, call_id="c2")
        result = node(_make_state([tc_good, tc_bad]))
        messages = result["messages"]
        assert len(messages) == 2
        ids = {m.tool_call_id for m in messages}
        assert ids == {"c1", "c2"}

    def test_second_call_succeeds_when_first_fails(self):
        mock_a = _make_mock_tool("labs_latest_value")
        mock_b = _make_mock_tool("labs_summary", {"analyte": "HbA1c", "count": 3})
        node = ValidatedToolNode({
            "labs_latest_value": mock_a,
            "labs_summary": mock_b,
        })
        tc_bad = _make_tool_call("labs_latest_value", {"analyte": ""}, call_id="c1")
        tc_good = _make_tool_call("labs_summary", {"analyte": "HbA1c"}, call_id="c2")
        result = node(_make_state([tc_bad, tc_good]))
        good_msg = next(m for m in result["messages"] if m.tool_call_id == "c2")
        assert "INPUT_VALIDATION_ERROR" not in good_msg.content
        mock_b.invoke.assert_called_once()

    def test_date_format_error_in_message(self):
        mock_tool = _make_mock_tool("whoop_recovery_metrics_on_date")
        node = ValidatedToolNode({"whoop_recovery_metrics_on_date": mock_tool})
        tc = _make_tool_call("whoop_recovery_metrics_on_date", {"date": "2024/01/15"})
        result = node(_make_state([tc]))
        assert "INPUT_VALIDATION_ERROR" in result["messages"][0].content
        assert "YYYY-MM-DD" in result["messages"][0].content or "date" in result["messages"][0].content


# ---------------------------------------------------------------------------
# Output Validation
# ---------------------------------------------------------------------------

class TestOutputValidation:
    def test_valid_dict_output_produces_tool_message(self):
        return_val = {
            "analyte": "Glucose",
            "value": 90.0,
            "unit": "mg/dL",
            "date": "2024-01-15",
        }
        mock_tool = _make_mock_tool("labs_latest_value", return_val)
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        tc = _make_tool_call("labs_latest_value", {"analyte": "Glucose"})
        result = node(_make_state([tc]))
        assert len(result["messages"]) == 1
        assert "OUTPUT_VALIDATION_ERROR" not in result["messages"][0].content

    def test_missing_required_output_field_produces_error(self):
        # labs_latest_value output requires "analyte" field
        return_val = {"value": 90.0}  # missing analyte
        mock_tool = _make_mock_tool("labs_latest_value", return_val)
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        tc = _make_tool_call("labs_latest_value", {"analyte": "Glucose"})
        result = node(_make_state([tc]))
        assert "OUTPUT_VALIDATION_ERROR" in result["messages"][0].content

    def test_str_output_bypasses_output_validation(self):
        mock_tool = _make_mock_tool("meds_timeline", "Metformin: started 2023-01-01")
        node = ValidatedToolNode({"meds_timeline": mock_tool})
        tc = _make_tool_call("meds_timeline", {})
        result = node(_make_state([tc]))
        assert "OUTPUT_VALIDATION_ERROR" not in result["messages"][0].content

    def test_none_output_handled_gracefully(self):
        mock_tool = _make_mock_tool("labs_latest_value", None)
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        tc = _make_tool_call("labs_latest_value", {"analyte": "Glucose"})
        result = node(_make_state([tc]))
        # None result: no crash, returns a ToolMessage
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], ToolMessage)

    def test_list_output_validates_each_item(self):
        # Both items have only optional fields — all should pass
        return_val = [
            {"date": "2024-01-15", "value": 5.6, "unit": "%"},
            {"date": "2024-02-01", "value": "<0.01", "flag": "L"},  # str value also valid
        ]
        mock_tool = _make_mock_tool("labs_history", return_val)
        node = ValidatedToolNode({"labs_history": mock_tool})
        tc = _make_tool_call("labs_history", {"analyte": "HbA1c"})
        result = node(_make_state([tc]))
        # All optional fields — should succeed
        assert "OUTPUT_VALIDATION_ERROR" not in result["messages"][0].content


# ---------------------------------------------------------------------------
# Tool Execution Errors
# ---------------------------------------------------------------------------

class TestToolExecutionErrors:
    def test_execution_exception_returns_error_tool_message(self):
        mock_tool = _make_mock_tool("labs_latest_value")
        mock_tool.invoke.side_effect = RuntimeError("parquet file missing")
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        tc = _make_tool_call("labs_latest_value", {"analyte": "Glucose"})
        result = node(_make_state([tc]))
        assert "TOOL_EXECUTION_ERROR" in result["messages"][0].content

    def test_execution_error_message_has_call_id(self):
        mock_tool = _make_mock_tool("labs_latest_value")
        mock_tool.invoke.side_effect = ValueError("bad value")
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        tc = _make_tool_call("labs_latest_value", {"analyte": "Glucose"}, call_id="exec_err_01")
        result = node(_make_state([tc]))
        assert result["messages"][0].tool_call_id == "exec_err_01"


# ---------------------------------------------------------------------------
# Retry Budget
# ---------------------------------------------------------------------------

class TestRetryBudget:
    @pytest.mark.skipif(not _CONSTANTS_AVAILABLE, reason="MAX_TOOL_VALIDATION_RETRIES not available")
    def test_exceeding_max_retries_produces_terminal_message(self):
        mock_tool = _make_mock_tool("labs_latest_value")
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        tc = _make_tool_call("labs_latest_value", {"analyte": ""}, call_id="retry_01")
        # Simulate state where this call_id has already hit the retry limit
        prior_errors = [
            {"tool": "labs_latest_value", "call_id": "retry_01", "type": "input", "error": "err"}
            for _ in range(MAX_TOOL_VALIDATION_RETRIES)
        ]
        result = node(_make_state([tc], tool_validation_errors=prior_errors))
        assert "MAX_RETRIES_EXCEEDED" in result["messages"][0].content

    @pytest.mark.skipif(not _CONSTANTS_AVAILABLE, reason="MAX_TOOL_VALIDATION_RETRIES not available")
    def test_terminal_message_does_not_invoke_tool(self):
        mock_tool = _make_mock_tool("labs_latest_value")
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        tc = _make_tool_call("labs_latest_value", {"analyte": ""}, call_id="retry_02")
        prior_errors = [
            {"tool": "labs_latest_value", "call_id": "retry_02", "type": "input", "error": "err"}
            for _ in range(MAX_TOOL_VALIDATION_RETRIES)
        ]
        node(_make_state([tc], tool_validation_errors=prior_errors))
        mock_tool.invoke.assert_not_called()

    def test_below_retry_limit_still_validates(self):
        mock_tool = _make_mock_tool("labs_latest_value")
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        tc = _make_tool_call("labs_latest_value", {"analyte": ""}, call_id="retry_03")
        # Only 1 prior error — should still attempt validation (and fail on empty analyte)
        prior_errors = [
            {"tool": "labs_latest_value", "call_id": "retry_03", "type": "input", "error": "err"}
        ]
        result = node(_make_state([tc], tool_validation_errors=prior_errors))
        assert "INPUT_VALIDATION_ERROR" in result["messages"][0].content

    def test_retry_count_is_per_call_id(self):
        mock_tool = _make_mock_tool("labs_latest_value")
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        # High error count for a different call_id should NOT affect this call
        tc = _make_tool_call("labs_latest_value", {"analyte": ""}, call_id="new_call")
        prior_errors = [
            {"tool": "labs_latest_value", "call_id": "other_call", "type": "input", "error": "err"}
            for _ in range(10)
        ]
        result = node(_make_state([tc], tool_validation_errors=prior_errors))
        # Should fail on empty analyte, not on MAX_RETRIES_EXCEEDED
        assert "INPUT_VALIDATION_ERROR" in result["messages"][0].content
        assert "MAX_RETRIES_EXCEEDED" not in result["messages"][0].content


# ---------------------------------------------------------------------------
# State Output
# ---------------------------------------------------------------------------

class TestStateOutput:
    def test_returns_messages_key(self):
        mock_tool = _make_mock_tool("labs_latest_value", {"analyte": "G", "value": 1.0})
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        tc = _make_tool_call("labs_latest_value", {"analyte": "Glucose"})
        result = node(_make_state([tc]))
        assert "messages" in result

    def test_returns_tool_outputs_key_on_success(self):
        mock_tool = _make_mock_tool("labs_latest_value", {"analyte": "Glucose", "value": 90.0})
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        tc = _make_tool_call("labs_latest_value", {"analyte": "Glucose"})
        result = node(_make_state([tc]))
        assert "tool_outputs" in result
        assert len(result["tool_outputs"]) == 1

    def test_tool_outputs_empty_on_validation_failure(self):
        mock_tool = _make_mock_tool("labs_latest_value")
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        tc = _make_tool_call("labs_latest_value", {"analyte": ""})
        result = node(_make_state([tc]))
        assert result.get("tool_outputs", []) == []

    def test_returns_tool_validation_errors_key(self):
        mock_tool = _make_mock_tool("labs_latest_value")
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        tc = _make_tool_call("labs_latest_value", {"analyte": ""})
        result = node(_make_state([tc]))
        assert "tool_validation_errors" in result

    def test_validation_error_entry_has_required_fields(self):
        mock_tool = _make_mock_tool("labs_latest_value")
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        tc = _make_tool_call("labs_latest_value", {"analyte": ""}, call_id="chk_001")
        result = node(_make_state([tc]))
        assert len(result["tool_validation_errors"]) == 1
        entry = result["tool_validation_errors"][0]
        assert "tool" in entry
        assert "call_id" in entry
        assert "type" in entry
        assert "error" in entry

    def test_validation_error_type_is_input(self):
        mock_tool = _make_mock_tool("labs_latest_value")
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        tc = _make_tool_call("labs_latest_value", {"analyte": ""})
        result = node(_make_state([tc]))
        assert result["tool_validation_errors"][0]["type"] == "input"

    def test_no_errors_on_success(self):
        mock_tool = _make_mock_tool("labs_latest_value", {"analyte": "Glucose", "value": 90.0})
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        tc = _make_tool_call("labs_latest_value", {"analyte": "Glucose"})
        result = node(_make_state([tc]))
        assert result["tool_validation_errors"] == []

    def test_tool_output_entry_has_name_and_args(self):
        return_val = {"analyte": "Glucose", "value": 90.0}
        mock_tool = _make_mock_tool("labs_latest_value", return_val)
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        tc = _make_tool_call("labs_latest_value", {"analyte": "Glucose"})
        result = node(_make_state([tc]))
        entry = result["tool_outputs"][0]
        assert entry["name"] == "labs_latest_value"
        assert "args" in entry
        assert "output" in entry


# ---------------------------------------------------------------------------
# Enhanced Input Error Messages
# ---------------------------------------------------------------------------

class TestEnhancedInputErrorMessages:
    def test_passed_value_shown_in_error(self):
        mock_tool = _make_mock_tool("whoop_recovery_metrics_on_date")
        node = ValidatedToolNode({"whoop_recovery_metrics_on_date": mock_tool})
        tc = _make_tool_call("whoop_recovery_metrics_on_date", {"date": "2024/01/15"})
        result = node(_make_state([tc]))
        assert "2024/01/15" in result["messages"][0].content

    def test_date_error_shows_expected_format_example(self):
        mock_tool = _make_mock_tool("whoop_recovery_metrics_on_date")
        node = ValidatedToolNode({"whoop_recovery_metrics_on_date": mock_tool})
        tc = _make_tool_call("whoop_recovery_metrics_on_date", {"date": "2024/01/15"})
        result = node(_make_state([tc]))
        content = result["messages"][0].content
        assert "YYYY-MM-DD" in content
        assert "2024-01-15" in content  # example of correct format

    def test_date_error_has_reformat_instruction(self):
        mock_tool = _make_mock_tool("whoop_recovery_metrics_on_date")
        node = ValidatedToolNode({"whoop_recovery_metrics_on_date": mock_tool})
        tc = _make_tool_call("whoop_recovery_metrics_on_date", {"date": "01/15/2024"})
        result = node(_make_state([tc]))
        content = result["messages"][0].content
        assert "Reformat" in content or "reformat" in content

    def test_empty_string_error_has_provide_instruction(self):
        mock_tool = _make_mock_tool("labs_latest_value")
        node = ValidatedToolNode({"labs_latest_value": mock_tool})
        tc = _make_tool_call("labs_latest_value", {"analyte": ""})
        result = node(_make_state([tc]))
        content = result["messages"][0].content
        assert "Provide" in content or "provide" in content

    def test_action_required_line_names_tool(self):
        mock_tool = _make_mock_tool("whoop_recovery_metrics_on_date")
        node = ValidatedToolNode({"whoop_recovery_metrics_on_date": mock_tool})
        tc = _make_tool_call("whoop_recovery_metrics_on_date", {"date": "bad-date"})
        result = node(_make_state([tc]))
        content = result["messages"][0].content
        assert "Action required" in content
        assert "whoop_recovery_metrics_on_date" in content

    def test_action_required_names_failing_field(self):
        mock_tool = _make_mock_tool("whoop_recovery_metrics_on_date")
        node = ValidatedToolNode({"whoop_recovery_metrics_on_date": mock_tool})
        tc = _make_tool_call("whoop_recovery_metrics_on_date", {"date": "2024/01/15"})
        result = node(_make_state([tc]))
        content = result["messages"][0].content
        assert "'date'" in content  # field name appears in action-required line

    def test_non_date_error_still_shows_passed_value(self):
        mock_tool = _make_mock_tool("labs_history")
        node = ValidatedToolNode({"labs_history": mock_tool})
        tc = _make_tool_call("labs_history", {"analyte": "HbA1c", "limit": -1})
        result = node(_make_state([tc]))
        assert "-1" in result["messages"][0].content
