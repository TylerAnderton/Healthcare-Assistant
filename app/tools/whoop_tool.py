import os
import math
from datetime import datetime, timedelta, UTC
import pandas as pd
from typing import Dict, List, Optional

from app.constants import (
    WHOOP_TABLE_FILES,
    WHOOP_SLEEPS_RAW_COLS,
    WHOOP_SLEEPS_PROCESSED_COLS,
    WHOOP_RECOVERY_RAW_COLS,
    WHOOP_RECOVERY_PROCESSED_COLS,
    WHOOP_WORKOUTS_RAW_COLS,
    WHOOP_WORKOUTS_PROCESSED_COLS,
)

BASE = os.path.join(os.getenv("PROCESSED_DIR", "./data/processed"), "tables")


def quick_summary() -> Dict[str, int]:
    out: Dict[str, int] = {}
    for key, parquet in WHOOP_TABLE_FILES.items():
        fp = os.path.join(BASE, parquet)
        if os.path.exists(fp):
            try:
                df = pd.read_parquet(fp)
                out[key] = len(df)
            except Exception:
                pass
    return out


# Deterministic mapping helper
def _map_row(row: pd.Series, raw_cols: List[str], processed_cols: List[str]) -> Dict[str, Optional[object]]:
    out: Dict[str, Optional[object]] = {}
    for i, pcol in enumerate(processed_cols):
        try:
            rcol = raw_cols[i]
        except IndexError:
            rcol = None
        out[pcol] = row.get(rcol) if rcol is not None else None
    # Ensure date is serialized to string
    if "date" in out:
        out["date"] = None if out["date"] is None else str(out["date"])  
    return out

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


def _filter_date_range(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    dff = df.copy()
    if start:
        dff = dff[dff["date"].astype(str) >= str(start)]
    if end:
        dff = dff[dff["date"].astype(str) <= str(end)]
    return dff


# Public API (derive from centralized mapping)
SLEEPS_PATH = os.path.join(BASE, WHOOP_TABLE_FILES.get("sleeps", "whoop_sleeps.parquet"))
RECOVERY_PATH = os.path.join(BASE, WHOOP_TABLE_FILES.get("physiological_cycles", "whoop_physiological_cycles.parquet"))
WORKOUTS_PATH = os.path.join(BASE, WHOOP_TABLE_FILES.get("workouts", "whoop_workouts.parquet"))


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
    dff = df[df["date"].astype(str) != ""].copy()
    dff = _filter_date_range(dff, start, end)
    if dff.empty:
        return []
    dff = dff.sort_values("date", ascending=ascending)
    if limit is not None and limit > 0:
        dff = dff.head(limit) if ascending else dff.tail(limit)
    out: List[Dict] = []
    for _, r in dff.iterrows():
        mapped = _map_row(r, WHOOP_SLEEPS_RAW_COLS, WHOOP_SLEEPS_PROCESSED_COLS)
        out.append(mapped)
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
    dff = df[df["date"].astype(str) != ""].copy()
    dff = _filter_date_range(dff, start, end)
    if dff.empty:
        return []
    dff = dff.sort_values("date", ascending=ascending)
    if limit is not None and limit > 0:
        dff = dff.head(limit) if ascending else dff.tail(limit)
    out: List[Dict] = []
    for _, r in dff.iterrows():
        mapped = _map_row(r, WHOOP_RECOVERY_RAW_COLS, WHOOP_RECOVERY_PROCESSED_COLS)
        out.append(mapped)
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
    dff = df[df[WHOOP_WORKOUTS_PROCESSED_COLS[0]].astype(str) != ""].copy()
    dff = _filter_date_range(dff, start, end)
    if dff.empty:
        return []
    dff = dff.sort_values(WHOOP_WORKOUTS_PROCESSED_COLS[0], ascending=ascending)
    if limit is not None and limit > 0:
        dff = dff.head(limit) if ascending else dff.tail(limit)
    out: List[Dict] = []
    for _, r in dff.iterrows():
        mapped = _map_row(r, WHOOP_WORKOUTS_RAW_COLS, WHOOP_WORKOUTS_PROCESSED_COLS)
        out.append(mapped)
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
