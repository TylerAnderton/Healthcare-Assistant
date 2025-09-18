from typing import List, Tuple, Set, Optional, Dict, Any
import logging
import json
import pandas as pd

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from dotenv import load_dotenv
load_dotenv()

from app.tools import labs_tool as LT
from app.tools import whoop_tool as WT
from app.tools import meds_tool as MT  # noqa: F401 (for potential future expansion)

from app.tools.structured_context import load_meds_timeline

logger = logging.getLogger(__name__)

# -----------------------------
# Tool-calling support (optional)
# -----------------------------
def answer_with_tools(llm: ChatOllama, prompt: ChatPromptTemplate) -> str:
    """Let the model call structured tools to fetch authoritative data.

    We avoid adding long background documents here to reduce interference and rely
    on tools that read the canonical parquet tables.
    """
    # Define callable tools that the model can invoke
    # Labs Tools
    def labs_list_analytes_tool(prefix: Optional[str] = None, table_path: Optional[str] = None) -> List[str]:
        """Return a list of all unique lab analytes.

        Args:
            prefix: Optional prefix to filter analytes by.
            table_path: Optional path to the labs table.
        """
        return LT.list_analytes(prefix=prefix, table_path=table_path)
    
    def labs_latest_value_tool(analyte: str, table_path: Optional[str] = None) -> Optional[Dict]:
        """Return the latest value for a lab analyte.

        Args:
            analyte: Name of the lab analyte.
            table_path: Optional path to the labs table.
        """
        return LT.latest_value(analyte, table_path=table_path)
    
    def labs_history_tool(analyte: str, limit: int = 10, ascending: bool = True) -> List[Dict[str, Any]]:
        """Return recent history for a lab analyte.

        Args:
            analyte: Name of the lab analyte, e.g., "ALT (SGPT)".
            limit: Max number of rows to return.
            ascending: Sort order by date.
        """
        return LT.history(analyte, limit=limit, ascending=ascending)

    def labs_summary_tool(analyte: str) -> Dict[str, Any]:
        """Return a summary for a lab analyte (last value/date, delta, unit, ref range).

        Args:
            analyte: Name of the lab analyte.
        """
        s = LT.summary(analyte)
        return s or {}

    def labs_value_on_date_tool(analyte: str, date: str) -> Dict[str, Any]:
        """Return the value of an analyte on an exact date (YYYY-MM-DD).

        Args:
            analyte: Name of the lab analyte.
            date: Date in YYYY-MM-DD.
        """
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
        return load_meds_timeline() or ""

    def meds_history_tool(medication: str, fuzzy: bool = True) -> List[Dict[str, Any]]:
        """Return the chronological dosing history for a medication.

        Args:
            medication: Name of the medication to search for.
            fuzzy: If True, enable fuzzy matching on the medication name.
        """
        return MT.get_medication_history(medication, fuzzy=fuzzy)

    def meds_dosage_on_date_tool(medication: str, date: Optional[str] = None, fuzzy: bool = True) -> Dict[str, Any]:
        """Return the dose/frequency for a medication on a specific date (YYYY-MM-DD). If date is omitted, use today.

        Args:
            medication: Name of the medication to search for.
            date: Date in YYYY-MM-DD to query. If None, defaults to today.
            fuzzy: If True, enable fuzzy matching on the medication name.
        """
        return MT.dosage_on_date(medication, date, fuzzy=fuzzy)

    def meds_list_medications_tool() -> List[str]:
        """Return a list of medications."""
        return MT.list_medications()

    def meds_list_current_tool(date: Optional[str|None] = None) -> List[Dict[str, Any]]:
        """Return a list of current medications and dosages on a given date."""
        return MT.list_current(date=date)

    # Whoop Tools
    def whoop_sleeps_on_date_tool(date: str) -> List[Dict[str, Any]]:
        """Return a list of WHOOP sleeps for an exact date (YYYY-MM-DD),
            including sleep onset time, sleep performance score,
            duration of time spend in bed, asleep, and each sleep stage,
            sleep efficiency, sleep consistency, and whether the sleep was a nap.

        Args:
            date: Date in YYYY-MM-DD.
        """
        rec: pd.DataFrame = WT.sleeps(start=date, end=date, ascending=True, limit=10)
        if len(rec) == 0:
            logger.warning(f"No sleep records found for date {date}")
        sleeps = []
        for r in rec.iterrows():
            if str(r.get("date")) == str(date):
                sleeps.append(r.to_dict())
        return sleeps

    def whoop_recovery_metrics_on_date_tool(date: str) -> Dict[str, Any]:
        """Return a dictionary of WHOOP recovery metrics for an exact date (YYYY-MM-DD),
            including recovery score, strain, RHR, and HRV.

        Args:
            date: Date in YYYY-MM-DD.
        """
        rec: pd.DataFrame = WT.recovery(start=date, end=date, ascending=True, limit=10)
        if len(rec) > 1:
            logger.warning(f"Multiple recovery records found for date {date}")
        elif len(rec) == 0:
            logger.warning(f"No recovery records found for date {date}")
        for r in rec.iterrows():
            if str(r.get("date")) == str(date):
                return r.to_dict()
        return {}
    
    def whoop_workouts_on_date_tool(date: str) -> List[Dict[str, Any]]:
        """Return a list of WHOOP workouts for an exact date (YYYY-MM-DD),
            including activity, duration, strain, average and maximum heart rates,
            calories, and time spent in each heart rate zone.

        Args:
            date: Date in YYYY-MM-DD.
        """
        rec: pd.DataFrame = WT.workouts(start=date, end=date, ascending=True, limit=10)
        workouts = []
        if len(rec) == 0:
            logger.info(f"No workout records found for date {date}")
        for r in rec.iterrows():
            if str(r.get("date")) == str(date):
                workouts.append(r.to_dict())
        return workouts


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
        whoop_sleeps_on_date_tool,
        whoop_recovery_metrics_on_date_tool,
        whoop_workouts_on_date_tool,
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

                elif name == "whoop_sleeps_on_date_tool":
                    result = whoop_sleeps_on_date_tool(**args)
                elif name == "whoop_recovery_metrics_on_date_tool":
                    result = whoop_recovery_metrics_on_date_tool(**args)
                elif name == "whoop_workouts_on_date_tool":
                    result = whoop_workouts_on_date_tool(**args)

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
    logger.warning("Last message is not an AIMessage")
    return "I'm unable to complete the tool-assisted answer right now."
