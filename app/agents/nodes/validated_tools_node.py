"""
ValidatedToolNode: replaces LangGraph ToolNode with Pydantic-validated tool execution.

For each tool call:
  1. Check retry budget (per call_id).
  2. Validate input args against the tool's InputModel.
  3. Execute the tool.
  4. Validate output against the tool's OutputModel (Dict-returning tools only;
     List[Dict] tools validate each item).

Validation failures produce a structured ToolMessage with error feedback so
the LLM can correct and retry via the normal agent → tools → agent loop.
"""

import logging
from typing import Any

from langchain_core.messages import ToolMessage
from pydantic import BaseModel, ValidationError

from app.tools.schemas import TOOL_SCHEMAS
from app.constants import MAX_TOOL_VALIDATION_RETRIES

logger = logging.getLogger(__name__)


def _retry_count(state: dict, call_id: str) -> int:
    errors = state.get("tool_validation_errors", [])
    return sum(1 for e in errors if e.get("call_id") == call_id)


def _error_message(
    call_id: str,
    tool_name: str,
    error_type: str,
    detail: str,
    failed_fields: list[str] | None = None,
) -> ToolMessage:
    fields_str = ", ".join(f"'{f}'" for f in failed_fields) if failed_fields else "the arguments"
    content = (
        f"[TOOL_ERROR:{error_type}] Tool '{tool_name}' failed.\n"
        f"Detail: {detail}\n\n"
        f"Action required: Fix {fields_str} and call '{tool_name}' again."
    )
    return ToolMessage(content=content, tool_call_id=call_id)


def _terminal_message(call_id: str, tool_name: str, count: int) -> ToolMessage:
    content = (
        f"[TOOL_ERROR:MAX_RETRIES_EXCEEDED] Tool '{tool_name}' failed {count} times. "
        f"Proceeding without this data."
    )
    return ToolMessage(content=content, tool_call_id=call_id)


def _format_input_error(tool_name: str, raw_args: dict, exc: ValidationError) -> str:
    lines = [f"Input validation failed for '{tool_name}':"]
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"]) if err["loc"] else "value"
        msg = err["msg"]
        passed_val = raw_args.get(field, "<not provided>")

        if "YYYY-MM-DD" in msg:
            lines.append(
                f"  - {field}: You passed {passed_val!r}. "
                f"Expected: YYYY-MM-DD format (example: '2024-01-15').\n"
                f"    → Reformat your date to YYYY-MM-DD and retry this tool call."
            )
        elif "non-empty" in msg or "at least 1 character" in msg:
            lines.append(
                f"  - {field}: You passed {passed_val!r}. Expected: a non-empty string.\n"
                f"    → Provide a valid {field} value and retry this tool call."
            )
        else:
            lines.append(
                f"  - {field}: You passed {passed_val!r}. {msg}.\n"
                f"    → Correct the '{field}' argument and retry this tool call."
            )
    return "\n".join(lines)


def _format_validation_error(model_name: str, exc: ValidationError) -> str:
    lines = [f"Validation failed for '{model_name}':"]
    for err in exc.errors():
        field = ".".join(str(p) for p in err["loc"]) if err["loc"] else "value"
        lines.append(f"  - {field}: {err['msg']}")
    return "\n".join(lines)


def _validate_output(result: Any, output_model: type[BaseModel]) -> None:
    """Validate tool output. Raises ValidationError on first failure."""
    if isinstance(result, list):
        for i, item in enumerate(result):
            if isinstance(item, dict):
                output_model(**item)
    elif isinstance(result, dict):
        output_model(**result)


class ValidatedToolNode:
    """Validated replacement for LangGraph ToolNode.

    Validates inputs before execution and outputs after. Errors are returned
    as structured ToolMessages so the LLM can self-correct and retry.
    """

    def __init__(self, tools_by_name: dict):
        self.tools_by_name = tools_by_name

    def __call__(self, state: dict) -> dict:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", [])
        messages = []
        tool_outputs = []
        tool_validation_errors = []

        for tc in tool_calls:
            name = tc["name"]
            raw_args = tc.get("args", {})
            call_id = tc.get("id", "")

            tool = self.tools_by_name.get(name)
            if tool is None:
                messages.append(_error_message(call_id, name, "UNKNOWN_TOOL", f"No tool named '{name}'"))
                continue

            # 1. Check retry budget
            if _retry_count(state, call_id) >= MAX_TOOL_VALIDATION_RETRIES:
                count = _retry_count(state, call_id)
                logger.warning("Tool '%s' exceeded max retries (%d) for call_id=%s", name, count, call_id)
                messages.append(_terminal_message(call_id, name, count))
                continue

            # 2. Input validation
            schema = TOOL_SCHEMAS.get(name)
            input_model_cls, output_model_cls = schema if schema else (None, None)
            invoke_args = raw_args

            if input_model_cls is not None:
                try:
                    validated = input_model_cls(**raw_args)
                    invoke_args = validated.model_dump(exclude_none=False)
                except ValidationError as exc:
                    failed_fields = [".".join(str(p) for p in e["loc"]) for e in exc.errors()]
                    detail = _format_input_error(name, raw_args, exc)
                    logger.debug("Input validation failed for tool '%s': %s", name, detail)
                    messages.append(_error_message(call_id, name, "INPUT_VALIDATION_ERROR", detail, failed_fields))
                    tool_validation_errors.append({
                        "tool": name,
                        "call_id": call_id,
                        "type": "input",
                        "error": detail,
                    })
                    continue

            # 3. Execute tool
            try:
                result = tool.invoke(invoke_args)
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"
                logger.exception("Tool execution failed for '%s': %s", name, detail)
                messages.append(_error_message(call_id, name, "TOOL_EXECUTION_ERROR", detail))
                tool_validation_errors.append({
                    "tool": name,
                    "call_id": call_id,
                    "type": "execution",
                    "error": detail,
                })
                continue

            # 4. Output validation (Dict/List[Dict] tools only; None result skipped)
            if output_model_cls is not None and result is not None:
                try:
                    _validate_output(result, output_model_cls)
                except ValidationError as exc:
                    detail = _format_validation_error(f"{name} output", exc)
                    logger.warning("Output validation failed for tool '%s': %s", name, detail)
                    messages.append(_error_message(call_id, name, "OUTPUT_VALIDATION_ERROR", detail))
                    tool_validation_errors.append({
                        "tool": name,
                        "call_id": call_id,
                        "type": "output",
                        "error": detail,
                    })
                    continue

            messages.append(ToolMessage(content=str(result), tool_call_id=call_id))
            tool_outputs.append({"name": name, "args": raw_args, "output": result})

        return {
            "messages": messages,
            "tool_outputs": tool_outputs,
            "tool_validation_errors": tool_validation_errors,
        }
