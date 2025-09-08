import os
import math
from datetime import datetime, timedelta, UTC
import pandas as pd
from typing import Dict, List, Optional

BASE = os.path.join(os.getenv("PROCESSED_DIR", "./data/processed"), "tables")


def quick_summary() -> Dict[str, int]:
    out = {}
    for name in ["sleeps", "workouts", "physiological_cycles", "journal_entries"]:
        fp = os.path.join(BASE, f"whoop_{name}.parquet")
        if os.path.exists(fp):
            try:
                df = pd.read_parquet(fp)
                out[name] = len(df)
            except Exception:
                pass
    return out


def pick_column(row: pd.Series, candidates: List[str]):
    for c in candidates:
        if c in row and pd.notna(row[c]):
            return row[c]
    return None


def parse_date_any(row: pd.Series, candidates):
    val = pick_column(row, candidates)
    if val is None:
        return ""
    try:
        s = str(val)
        if "T" in s:
            s = s.split("T", 1)[0]
        # simple normalization; if not ISO, return prefix
        try:
            dt = datetime.fromisoformat(s)
            return dt.date().isoformat()
        except Exception:
            return s[:10]
    except Exception:
        return ""

# Internal helpers
def _load_df(table_path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(table_path):
        return None
    try:
        df = pd.read_parquet(table_path)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
    except Exception:
        return None


def _norm_date_str(val) -> str:
    try:
        s = str(val)
        if not s:
            return ""
        if "T" in s:
            s = s.split("T", 1)[0]
        try:
            dt = datetime.fromisoformat(s)
            return dt.date().isoformat()
        except Exception:
            return s[:10]
    except Exception:
        return ""


def _ensure_date_col(df: pd.DataFrame, candidates: List[str]) -> pd.DataFrame:
    if "date" not in df.columns:
        df = df.copy()
        df["date"] = df.apply(lambda r: _norm_date_str(pick_column(r, candidates)), axis=1)
    return df


def _filter_date_range(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    dff = df.copy()
    if start:
        dff = dff[dff["date"].astype(str) >= str(start)]
    if end:
        dff = dff[dff["date"].astype(str) <= str(end)]
    return dff


# Public API
SLEEPS_PATH = os.path.join(BASE, "whoop_sleeps.parquet")
RECOVERY_PATH = os.path.join(BASE, "whoop_physiological_cycles.parquet")
WORKOUTS_PATH = os.path.join(BASE, "whoop_workouts.parquet")


def sleeps(
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: Optional[int] = None,
    ascending: bool = True,
    table_path: Optional[str] = None,
) -> List[Dict]:
    """Return key sleep metrics per date.

    Fields: date, sleep_score, rhr_bpm, hrv_ms, duration_sec, efficiency_pct
    """
    tp = table_path or SLEEPS_PATH
    df = _load_df(tp)
    if df is None:
        return []
    df = _ensure_date_col(df, ["date", "day", "Cycle start time"])  # type: ignore
    dff = df[df["date"].astype(str) != ""].copy()
    dff = _filter_date_range(dff, start, end)
    if dff.empty:
        return []
    dff = dff.sort_values("date", ascending=ascending)
    if limit is not None and limit > 0:
        dff = dff.head(limit) if ascending else dff.tail(limit)
    out: List[Dict] = []
    for _, r in dff.iterrows():
        out.append({
            "date": str(r.get("date")),
            "sleep_score": pick_column(r, ["sleep_performance", "sleep_score", "score", "Sleep performance %"]),
            # TODO: Apply time zone conversion with "Cycle timezone" column
            "sleep_start_time": pick_column(r, ["sleep_start", "start_time", "Sleep onset"]),
            "inbed_duration_min": pick_column(r, ["In bed duration (min)"]),
            "asleep_duration_min": pick_column(r, ["Asleep duration (min)"]),
            "light_sleep_duration_min": pick_column(r, ["Light sleep duration (min)"]),
            "deep_sleep_duration_min": pick_column(r, ["Deep (SWS) duration (min)"]),
            "rem_sleep_duration_min": pick_column(r, ["REM duration (min)"]),
            "awake_duration_min": pick_column(r, ["Awake duration (min)"]),
            "efficiency_pct": pick_column(r, ["sleep_efficiency", "efficiency", "Sleep efficiency %"]),
            "consistency_pct": pick_column(r, ["sleep_consistency", "consistency", "Sleep consistency %"]),
            "nap": pick_column(r, ["nap", "Nap"]),
        })
    return out


# Physiological cycles
def recovery(
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: Optional[int] = None,
    ascending: bool = True,
    table_path: Optional[str] = None,
) -> List[Dict]:
    """Return key recovery metrics per date.

    Fields: date, recovery_score, rhr_bpm, hrv_ms
    """
    tp = table_path or RECOVERY_PATH
    df = _load_df(tp)
    if df is None:
        return []
    df = _ensure_date_col(df, ["date", "cycle_date", "start", "day", "Cycle start time"])  # type: ignore
    dff = df[df["date"].astype(str) != ""].copy()
    dff = _filter_date_range(dff, start, end)
    if dff.empty:
        return []
    dff = dff.sort_values("date", ascending=ascending)
    if limit is not None and limit > 0:
        dff = dff.head(limit) if ascending else dff.tail(limit)
    out: List[Dict] = []
    for _, r in dff.iterrows():
        out.append({
            "date": str(r.get("date")),
            "strain": pick_column(r, ["strain", "Day Strain"]),
            "recovery_score": pick_column(r, ["recovery_score", "recovery", "score", "Recovery score %"]),
            "rhr_bpm": pick_column(r, ["resting_heart_rate", "rhr", "Resting heart rate (bpm)"]),
            "hrv_ms": pick_column(r, ["hrv", "hrv_rmssd_milli", "rmssd", "Heart rate variability (ms)"]),
        })
    return out


def workouts(
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: Optional[int] = None,
    ascending: bool = True,
    table_path: Optional[str] = None,
) -> List[Dict]:
    """Return basic workout metrics per date.

    Fields: date, activity, duration_min, strain, avg_hr_bpm, calories
    """
    tp = table_path or WORKOUTS_PATH
    df = _load_df(tp)
    if df is None:
        return []
    df = _ensure_date_col(df, ["date", "start", "workout_start", "start_time", "Cycle start time"])  # type: ignore
    dff = df[df["date"].astype(str) != ""].copy()
    dff = _filter_date_range(dff, start, end)
    if dff.empty:
        return []
    dff = dff.sort_values("date", ascending=ascending)
    if limit is not None and limit > 0:
        dff = dff.head(limit) if ascending else dff.tail(limit)
    out: List[Dict] = []
    for _, r in dff.iterrows():
        out.append({
            "date": str(r.get("date")),
            "activity": pick_column(r, ["sport", "activity", "type", "Activity name"]),
            "duration_min": pick_column(r, ["duration", "workout_duration", "Duration (minutes)"]),
            "strain": pick_column(r, ["strain", "Activity Strain"]),
            "avg_hr_bpm": pick_column(r, ["average_heart_rate", "avg_hr", "Average HR (bpm)"]),
            "max_hr_bpm": pick_column(r, ["max_heart_rate", "max_hr", "Max HR (bpm)"]),
            "calories": pick_column(r, ["calories", "kcal", "Energy burned (cal)"]),
            "hr_zone_1": pick_column(r, ["hr_zone_1", "HR Zone 1 %"]),
            "hr_zone_2": pick_column(r, ["hr_zone_2", "HR Zone 2 %"]),
            "hr_zone_3": pick_column(r, ["hr_zone_3", "HR Zone 3 %"]),
            "hr_zone_4": pick_column(r, ["hr_zone_4", "HR Zone 4 %"]),
            "hr_zone_5": pick_column(r, ["hr_zone_5", "HR Zone 5 %"]),
        })
    return out


def recent(days: int = 7) -> Dict[str, List[Dict]]:
    """Return recent sleeps, recovery, and workouts within the last `days`."""
    today = datetime.now(UTC).date()
    start = (today - timedelta(days=max(0, days - 1))).isoformat()
    return {
        "sleeps": sleeps(start=start, ascending=True),
        "recovery": recovery(start=start, ascending=True),
        "workouts": workouts(start=start, ascending=True),
    }
