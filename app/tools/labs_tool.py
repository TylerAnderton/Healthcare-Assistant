from typing import Optional, List, Dict
import os
import math
import pandas as pd
import logging

TABLE_PATH = os.path.join(os.getenv("PROCESSED_DIR", "./data/processed"), "tables", "labs.parquet")

logger = logging.getLogger(__name__)

from app.constants import LABS_PROCESSED_COLS

def _load_df(table_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    tp = table_path or TABLE_PATH
    if not os.path.exists(tp):
        logger.warning(f'Labs table not found at {tp}')
        return None
    try:
        df = pd.read_parquet(tp)
        if df is None or df.empty:
            logger.warning(f'Labs table is empty at {tp}')
            return None
        return df
    except Exception:
        logger.error(f'Failed to load labs table at {tp}')
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


def list_analytes(prefix: Optional[str] = None, table_path: Optional[str] = None) -> List[str]:
    """
    Return a list of all unique lab analytes.

    Args:
        prefix: Optional prefix to filter analytes by.
        table_path: Optional path to the labs table.
    """
    df = _load_df(table_path)
    if df is None or "analyte" not in df.columns:
        logger.warning(f'No valid labs table found at {table_path}')
        return []
    names = [str(x) for x in df["analyte"].dropna().unique()]
    names.sort(key=lambda s: s.lower())
    if prefix:
        pl = prefix.lower()
        names = [n for n in names if n.lower().startswith(pl)]
    return names


def latest_value(analyte: str, table_path: Optional[str] = None) -> Optional[Dict]:
    """
    Return the latest value for a lab analyte.

    Args:
        analyte: Name of the lab analyte.
        table_path: Optional path to the labs table.
    """
    df = _load_df(table_path)
    if df is None or "analyte" not in df.columns or "value" not in df.columns:
        logger.warning(f'No valid labs table found at {table_path}')
        return None
    dff = df[df["analyte"].str.lower() == analyte.lower()].copy()
    if dff.empty:
        logger.warning(f'Labs table at {table_path} is empty')
        return None
    dff = dff.sort_values("date", ascending=False)
    r = dff.iloc[0]
    # TODO: Use LABS_PROCESSED_COLS
    return {
        "analyte": str(r.get("analyte", analyte)),
        "value": r.get("value"),
        "unit": r.get("unit"),
        "date": r.get("date"),
        "vendor": r.get("vendor"),
        "ref_low": r.get("ref_low"),
        "ref_high": r.get("ref_high"),
        "flag": r.get("flag"),
        "source": r.get("source"),
        "page": r.get("page"),
    }


def history(analyte: str, limit: Optional[int] = None, ascending: bool = True, table_path: Optional[str] = None) -> List[Dict]:
    """
    Return recent history for a lab analyte.

    Args:
        analyte: Name of the lab analyte, e.g., "ALT (SGPT)".
        limit: Max number of rows to return.
        ascending: Sort order by date.
        table_path: Optional path to the labs table.
    """
    df = _load_df(table_path)
    if df is None or "analyte" not in df.columns:
        logger.warning(f'No valid labs table found at {table_path}')
        return []
    dff = df[df["analyte"].str.lower() == analyte.lower()].copy()
    if dff.empty:
        logger.warning(f'Labs table at {table_path} is empty')
        return []
    dff = dff.sort_values("date", ascending=ascending)
    if limit is not None and limit > 0:
        dff = dff.head(limit) if ascending else dff.tail(limit)
    cols = [c for c in ["date", "value", "unit", "ref_low", "ref_high", "flag", "vendor", "source", "page"] if c in dff.columns]
    return dff[cols].to_dict(orient="records")


def summary(analyte: str, table_path: Optional[str] = None) -> Optional[Dict]:
    """
    Return a summary for a lab analyte (last value/date, delta, unit, ref range).

    Args:
        analyte: Name of the lab analyte.
        table_path: Optional path to the labs table.
    """
    logger.info(f'Building summary for {analyte}')
    df = _load_df(table_path)

    if df is None or "analyte" not in df.columns or "value" not in df.columns:
        logger.warning(f'No valid labs table found at {table_path}')
        return None
    dff = df[df["analyte"].str.lower() == analyte.lower()].copy()
    if dff.empty:
        logger.warning(f'No valid data found for {analyte}')
        return None
    dff = dff.sort_values("date")
    vals = pd.to_numeric(dff["value"], errors="coerce")
    vals = vals.dropna()
    if vals.empty:
        logger.warning(f'No valid data found for {analyte}')
        return None
    
    logger.info(f'{len(vals)} rows found for {analyte}')

    last = dff.iloc[-1]
    last_val = _safe_float(last.get("value"))
    prev_val = _safe_float(dff.iloc[-2].get("value")) if len(dff) >= 2 else None
    delta = (last_val - prev_val) if (last_val is not None and prev_val is not None) else None
    pct = (delta / prev_val * 100.0) if (delta is not None and prev_val not in (None, 0)) else None
    rl = _safe_float(last.get("ref_low"))
    rh = _safe_float(last.get("ref_high"))
    out_of_range = None

    if last_val is not None and (rl is not None or rh is not None):
        if rl is not None and last_val < rl:
            out_of_range = "LOW"
        if rh is not None and last_val > rh:
            out_of_range = "HIGH"

    # TODO: Use LABS_PROCESSED_COLS
    return {
        "analyte": str(last.get("analyte", analyte)),
        "count": int(len(dff)),
        "first_date": dff.iloc[0].get("date"),
        "last_date": last.get("date"),
        "last_value": last.get("value"),
        "unit": last.get("unit"),
        "min": float(vals.min()) if len(vals) else None,
        "max": float(vals.max()) if len(vals) else None,
        "mean": float(vals.mean()) if len(vals) else None,
        "delta_from_prev": delta,
        "pct_change_from_prev": pct,
        "ref_low": rl,
        "ref_high": rh,
        "flag": last.get("flag"),
        "out_of_range": out_of_range,
        "vendor": last.get("vendor"),
    }


def latest_panel(limit: int = 15, table_path: Optional[str] = None) -> List[Dict]:
    """Return rows from the most recent lab date as a simple snapshot."""
    df = _load_df(table_path)
    if df is None or "date" not in df.columns:
        return []
    dff = df.copy()
    if dff.empty:
        return []
    # Keep rows with a date and select the lexicographically max date (ISO-friendly)
    dff = dff[dff["date"].notna()]
    if dff.empty:
        return []
    latest_date = sorted(dff["date"].astype(str).unique())[-1]
    panel = dff[dff["date"] == latest_date].copy()
    if "analyte" in panel.columns:
        panel = panel.sort_values("analyte")
    if limit is not None and limit > 0:
        panel = panel.head(limit)
    # TODO: Use LABS_PROCESSED_COLS
    # Reuse centralized schema to select known columns in a consistent order
    preferred_cols = [
        "analyte", "value", "unit", "date", "ref_low", "ref_high", "flag", "vendor", "source", "page"
    ]
    # Ensure we only select columns that exist; fall back to LABS_PROCESSED_COLS order if needed
    cols = [c for c in preferred_cols if c in panel.columns]
    if not cols:
        cols = [c for c in LABS_PROCESSED_COLS if c in panel.columns]
    return panel[cols].to_dict(orient="records")
