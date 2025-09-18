"""
Helpers to build structured context blocks from tabular data (e.g., meds timeline)
that can be prepended to RAG context for better recall and reasoning.
"""
from __future__ import annotations

import os
from typing import List, Optional
import pandas as pd
from collections import defaultdict
from datetime import datetime, UTC, timedelta
import logging

from .whoop_tool import recent as whoop_recent

logger = logging.getLogger(__name__)


def _safe_str(x) -> str:
    try:
        import math
        if x is None:
            return ""
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return ""
        return str(x)
    except Exception:
        return ""


def load_meds_timeline(
    max_meds: int = 12, # TODO: Set max meds and events in env var
    max_events: int = 24,
    processed_dir: Optional[str] = None,
    table_path: Optional[str] = None,
) -> str:
    """Build a compact medication dosing timeline from tables/meds.parquet.

    Returns a text block suitable for inclusion in the RAG context.
    """
    # Tool-first refactor: reuse meds_tool for event construction
    try:
        from app.tools import meds_tool as MT
    except Exception:
        return ""

    meds = MT.list_medications()
    if not meds:
        return ""

    grouped = defaultdict(list)
    for med_name in meds[:max_meds]:
        try:
            hist = MT.get_medication_history(med_name, fuzzy=False)
        except Exception:
            hist = []
        if not hist:
            continue
        # Ensure chronological order and trim
        hist = sorted(hist, key=lambda e: str(e.get("date")))
        grouped[med_name] = hist[-max_events:]

    lines: List[str] = ["[structured_meds_timeline]"]
    for med_name in list(grouped.keys())[:max_meds]:
        lines.append(f"- Drug: {med_name}")
        for e in grouped[med_name]:
            dose_s = f"{e['dose']} {e['dose_unit']}".strip()
            freq_s = f"{e['freq']} {e['freq_unit']}".strip()
            details = "; ".join([
                s for s in [
                    dose_s if dose_s.strip() else "",
                    freq_s if freq_s.strip() else "",
                    f"current={e['current']}" if e['current'] else "",
                ] if s
            ])
            lines.append(f"  - {e['date']} {e['kind']}{(': ' + details) if details else ''}")

    return "\n".join(lines)


def load_labs_panel(
    max_rows: int = 40,
    processed_dir: Optional[str] = None,
    table_path: Optional[str] = None,
) -> str:
    """Build a compact snapshot of the most recent lab panel from tables/labs.parquet.

    Returns a text block suitable for inclusion in the RAG context.
    """
    if table_path is None:
        if processed_dir is None:
            processed_dir = os.getenv("PROCESSED_DIR", "./data/processed")
        table_path = os.path.join(processed_dir, "tables", "labs.parquet")

    if not os.path.exists(table_path):
        return ""

    try:
        df = pd.read_parquet(table_path)
    except Exception:
        return ""

    if df is None or df.empty or "date" not in df.columns:
        return ""

    dff = df[df["date"].notna()].copy()
    if dff.empty:
        return ""

    # Determine latest date (ISO sorts lexicographically)
    latest_date = sorted(dff["date"].astype(str).unique())[-1]
    logger.info(f"Latest lab panel date: {latest_date}")
    panel = dff[dff["date"] == latest_date].copy()
    if panel.empty:
        return ""

    # Prioritize out-of-range rows first
    def _to_float(x):
        try:
            import math
            if x is None:
                return None
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                return None
            return float(x)
        except Exception:
            return None

    def _oor(row) -> int:
        v = _to_float(row.get("value"))
        rl = _to_float(row.get("ref_low"))
        rh = _to_float(row.get("ref_high"))
        if v is None:
            return 0
        if rl is not None and v < rl:
            return 1
        if rh is not None and v > rh:
            return 1
        return 0

    panel = panel.copy()
    panel["_oor"] = panel.apply(_oor, axis=1)
    panel = panel.sort_values(["_oor", "analyte"], ascending=[False, True])
    if max_rows is not None and max_rows > 0:
        panel = panel.head(max_rows)

    lines: List[str] = ["[structured_labs_panel]", f"- Date: {latest_date}"]
    for _, r in panel.fillna("").iterrows():
        name = _safe_str(r.get("analyte"))
        val = _safe_str(r.get("value"))
        unit = _safe_str(r.get("unit"))
        rl = _safe_str(r.get("ref_low"))
        rh = _safe_str(r.get("ref_high"))
        flag = _safe_str(r.get("flag"))
        ref = f"ref {rl}-{rh}" if (rl or rh) else ""
        pieces = [name + ":", val, unit, f"({ref})" if ref else "", f"[{flag}]" if flag else ""]
        line = " ".join([p for p in pieces if p]).strip()
        if line:
            lines.append("  - " + line)

    return "\n".join(lines)


def _fmt_minutes_to_hm(min):
    try:
        if min is None:
            return ""
        s = int(float(min))
        if s < 0:
            return ""
        h, rem = divmod(s, 60)
        if h > 0:
            return f"{h}h {rem}m"
        return f"{rem}m"
    except Exception:
        return ""


def load_whoop_recent(
    days: int = 7,
) -> str:
    """Build a compact WHOOP snapshot block for recent days using whoop_tool.recent().

    This delegates all column mapping and date parsing to tools/whoop_tool.py
    to avoid code duplication. processed_dir is accepted for API compatibility,
    but whoop_tool handles path resolution internally.
    """
    try:
        data = whoop_recent(days=days)
    except Exception:
        return ""

    lines: List[str] = []

    # Sleeps
    sleeps = data.get("sleeps") or []
    if sleeps:
        lines.append("[whoop_recent_sleeps]")
        for r in sorted(sleeps, key=lambda x: str(x.get("date"))):
            date = str(r.get("date"))
            score = r.get("sleep_score")
            sleep_start_time = r.get("sleep_start_time")
            inbed_dur_min = _fmt_minutes_to_hm(r.get("inbed_duration_min"))
            asleep_dur_min = _fmt_minutes_to_hm(r.get("asleep_duration_min"))
            light_sleep_dur_min = _fmt_minutes_to_hm(r.get("light_sleep_duration_min"))
            deep_sleep_dur_min = _fmt_minutes_to_hm(r.get("deep_sleep_duration_min"))
            rem_sleep_dur_min = _fmt_minutes_to_hm(r.get("rem_sleep_duration_min"))
            awake_dur_min = _fmt_minutes_to_hm(r.get("awake_duration_min"))
            efficiency = r.get("efficiency_pct")
            consistency = r.get("consistency_pct")
            nap = r.get("nap")

            parts = [
                f"- {date}:",
                f"sleep score {score}" if score not in (None, "") else "",
                f"sleep start {sleep_start_time}" if sleep_start_time not in (None, "") else "",
                f"inbed duration {inbed_dur_min}" if inbed_dur_min else "",
                f"asleep duration {asleep_dur_min}" if asleep_dur_min else "",
                f"light sleep duration {light_sleep_dur_min}" if light_sleep_dur_min else "",
                f"deep sleep duration {deep_sleep_dur_min}" if deep_sleep_dur_min else "",
                f"rem sleep duration {rem_sleep_dur_min}" if rem_sleep_dur_min else "",
                f"awake duration {awake_dur_min}" if awake_dur_min else "",
                f"efficiency {efficiency}%" if efficiency not in (None, "") else "",
                f"consistency {consistency}%" if consistency not in (None, "") else "",
                f"nap {nap}" if nap not in (None, "") else "",
            ]
            s = " ".join([p for p in parts if p]).strip()
            if s:
                lines.append(s)

    # Recovery
    recovery = data.get("recovery") or []
    if recovery:
        lines.append("[whoop_recent_recovery]")
        for r in sorted(recovery, key=lambda x: str(x.get("date"))):
            date = str(r.get("date"))
            strain = r.get("strain")
            rec = r.get("recovery_score")
            rhr = r.get("rhr_bpm")
            hrv = r.get("hrv_ms")

            parts = [
                f"- {date}:",
                f"strain {strain}" if strain not in (None, "") else "",
                f"recovery {rec}" if rec not in (None, "") else "",
                f"RHR {rhr} bpm" if rhr not in (None, "") else "",
                f"HRV {hrv} ms" if hrv not in (None, "") else "",
            ]
            s = " ".join([p for p in parts if p]).strip()
            if s:
                lines.append(s)

    # Workouts
    workouts = data.get("workouts") or []
    if workouts:
        lines.append("[whoop_recent_workouts]")
        for r in sorted(workouts, key=lambda x: str(x.get("date"))):
            date = str(r.get("date"))
            activity = r.get("activity")
            dur_str = _fmt_minutes_to_hm(r.get("duration_min"))
            strain = r.get("strain")
            avg_hr = r.get("avg_hr_bpm")
            max_hr = r.get("max_hr_bpm")
            cal = r.get("calories")
            hr_zone_1 = r.get("hr_zone_1")
            hr_zone_2 = r.get("hr_zone_2")
            hr_zone_3 = r.get("hr_zone_3")
            hr_zone_4 = r.get("hr_zone_4")
            hr_zone_5 = r.get("hr_zone_5")

            parts = [
                f"- {date}:",
                f"{activity}" if activity not in (None, "") else "workout",
                f"duration {dur_str}" if dur_str else "",
                f"strain {strain}" if strain not in (None, "") else "",
                f"avg HR {avg_hr} bpm" if avg_hr not in (None, "") else "",
                f"max HR {max_hr} bpm" if max_hr not in (None, "") else "",
                f"{cal} kcal" if cal not in (None, "") else "",
                f"HR Zone 1 {hr_zone_1}%" if hr_zone_1 not in (None, "") else "",
                f"HR Zone 2 {hr_zone_2}%" if hr_zone_2 not in (None, "") else "",
                f"HR Zone 3 {hr_zone_3}%" if hr_zone_3 not in (None, "") else "",
                f"HR Zone 4 {hr_zone_4}%" if hr_zone_4 not in (None, "") else "",
                f"HR Zone 5 {hr_zone_5}%" if hr_zone_5 not in (None, "") else "",
            ]
            s = " ".join([p for p in parts if p]).strip()
            if s:
                lines.append(s)

    if not lines:
        return ""
    header = f"[structured_whoop_recent] (last {days} days)"
    return "\n".join([header] + lines)
