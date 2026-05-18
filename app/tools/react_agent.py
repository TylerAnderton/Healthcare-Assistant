from typing import List, Optional, Dict, Any
import logging

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from dotenv import load_dotenv
load_dotenv()

from app.tools import labs_tool as LT
from app.tools import whoop_tool as WT
from app.tools import meds_tool as MT
from app.tools.structured_context import load_meds_timeline

logger = logging.getLogger(__name__)

# ============================================================================
# Labs Tools
# ============================================================================

@tool
def labs_list_analytes(prefix: Optional[str] = None, table_path: Optional[str] = None) -> List[str]:
    """List all unique lab analyte names in the database."""
    return LT.list_analytes(prefix=prefix, table_path=table_path)


@tool
def labs_latest_value(analyte: str, table_path: Optional[str] = None) -> Optional[Dict]:
    """Get the most recent result for a lab analyte (analyte, value, unit, date, etc.)."""
    return LT.latest_value(analyte, table_path=table_path)


@tool
def labs_history(analyte: str, limit: int = 10, ascending: bool = True) -> List[Dict[str, Any]]:
    """Get chronological history of results for a lab analyte. Use limit to cap rows."""
    return LT.history(analyte, limit=limit, ascending=ascending)


@tool
def labs_summary(analyte: str) -> Dict[str, Any]:
    """Get aggregate statistics for a lab analyte: latest value/date, min/max/mean, reference range, flags."""
    s = LT.summary(analyte)
    return s or {}


@tool
def labs_value_on_date(analyte: str, date: str) -> Dict[str, Any]:
    """Get the exact result for a lab analyte on a specific date (YYYY-MM-DD format)."""
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


# ============================================================================
# Medications Tools
# ============================================================================

@tool
def meds_timeline() -> str:
    """Get a compact text timeline of all medication dosing changes and events."""
    return load_meds_timeline() or ""


@tool
def meds_history(medication: str, fuzzy: bool = True) -> List[Dict[str, Any]]:
    """Get chronological dosing history for a medication (all dose/frequency/date changes)."""
    return MT.get_medication_history(medication, fuzzy=fuzzy)


@tool
def meds_dosage_on_date(medication: str, date: Optional[str] = None, fuzzy: bool = True) -> Dict[str, Any]:
    """Get dose and frequency for a medication on a specific date (YYYY-MM-DD), or today if no date given."""
    return MT.dosage_on_date(medication, date, fuzzy=fuzzy)


@tool
def meds_list_medications() -> List[str]:
    """List all unique medication names in the database."""
    return MT.list_medications()


@tool
def meds_list_current(date: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get all active medications and their dosages on a specific date (YYYY-MM-DD), or today if no date given."""
    return MT.list_current(date=date)


# ============================================================================
# WHOOP Tools
# ============================================================================

@tool
def whoop_sleeps_on_date(date: str) -> List[Dict[str, Any]]:
    """Get all sleep sessions for a date (YYYY-MM-DD): sleep score, duration, onset time, nap status, sleep stages."""
    rec = WT.sleeps(start=date, end=date, ascending=True, limit=10)
    if len(rec) == 0:
        logger.warning(f"No sleep records found for date {date}")
    sleeps = []
    for _, r in rec.iterrows():
        if str(r.get("date")) == str(date):
            sleeps.append(r.to_dict())
    return sleeps


@tool
def whoop_recovery_metrics_on_date(date: str) -> Dict[str, Any]:
    """Get recovery metrics for a date (YYYY-MM-DD): recovery score, strain, resting heart rate, heart rate variability."""
    rec = WT.recovery(start=date, end=date, ascending=True, limit=10)
    if len(rec) > 1:
        logger.warning(f"Multiple recovery records found for date {date}")
    elif len(rec) == 0:
        logger.warning(f"No recovery records found for date {date}")
    for _, r in rec.iterrows():
        if str(r.get("date")) == str(date):
            return r.to_dict()
    return {}


@tool
def whoop_workouts_on_date(date: str) -> List[Dict[str, Any]]:
    """Get all workouts for a date (YYYY-MM-DD): activity type, duration, strain, avg/max heart rates, calories, heart rate zones."""
    rec = WT.workouts(start=date, end=date, ascending=True, limit=10)
    workouts = []
    if len(rec) == 0:
        logger.info(f"No workout records found for date {date}")
    for _, r in rec.iterrows():
        if str(r.get("date")) == str(date):
            workouts.append(r.to_dict())
    return workouts


# ============================================================================
# ReAct Agent Builder
# ============================================================================

def build_react_agent(llm):
    """Create a ReAct agent with all available tools."""
    tools = [
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
    return create_react_agent(llm, tools=tools)


def answer_with_react_agent(llm, prompt) -> str:
    """Answer a question using a ReAct agent with tool calling."""
    agent = build_react_agent(llm)
    messages = prompt.format_messages()

    try:
        result = agent.invoke({"messages": messages})
        # Extract the final text response from the agent output
        final_message = result.get("messages", [])[-1] if result.get("messages") else None
        if final_message and hasattr(final_message, "content"):
            return final_message.content if isinstance(final_message.content, str) else str(final_message.content)
        return "Unable to generate a response."
    except Exception as e:
        logger.exception("Error in ReAct agent: %s", e)
        return f"An error occurred while processing your question: {e}"
