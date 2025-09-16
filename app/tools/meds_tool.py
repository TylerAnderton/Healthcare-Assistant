from typing import Optional, List, Dict, Tuple
import os
import pandas as pd
import logging
import difflib

logger = logging.getLogger(__name__)

TABLE_PATH = os.path.join(os.getenv("PROCESSED_DIR", "./data/processed"), "tables", "meds.parquet")


def _load_df() -> Optional[pd.DataFrame]:
    if not os.path.exists(TABLE_PATH):
        logger.warning(f'Meds table not found at {TABLE_PATH}')
        return None
    try:
        df = pd.read_parquet(TABLE_PATH)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        logger.warning(f'Failed to load meds table at {TABLE_PATH}')
        return None


def _detect_cols(df: pd.DataFrame) -> Dict[str, Optional[str] | List[str]]:
    name_col = None
    for c in df.columns:
        lc = str(c).lower()
        if lc in ["name", "medication", "drug"]:
            name_col = c
            break
    dose_cols = [c for c in df.columns if str(c).lower() in ["dose", "dosage"]]
    dose_unit_cols = [c for c in df.columns if str(c).lower() in ["dose_unit"]]
    freq_cols = [c for c in df.columns if str(c).lower() in ["frequency", "freq", "dose_frequency"]]
    freq_unit_cols = [c for c in df.columns if str(c).lower() in ["frequency_unit", "dose_frequency_unit"]]
    start_cols = [c for c in df.columns if str(c).lower() in ["start_date", "date_start", "start"]]
    updated_cols = [c for c in df.columns if str(c).lower() in ["dose_updated", "date_updated", "updated", "date_changed", "change_date"]]
    end_cols = [c for c in df.columns if str(c).lower() in ["end_date", "date_stop", "end"]]
    current_cols = [c for c in df.columns if str(c).lower() in ["current", "is_current"]]
    return {
        "name_col": name_col,
        "dose_cols": dose_cols,
        "dose_unit_cols": dose_unit_cols,
        "freq_cols": freq_cols,
        "freq_unit_cols": freq_unit_cols,
        "start_cols": start_cols,
        "updated_cols": updated_cols,
        "end_cols": end_cols,
        "current_cols": current_cols,
    }


def _norm(s: str) -> str:
    try:
        return "".join(ch for ch in str(s).lower() if ch.isalnum())
    except Exception:
        return ""


def _best_name_match(names: List[str], query: str) -> Tuple[str, float]:
    if not names:
        return "", 0.0
    # Use difflib for lightweight fuzzy match
    scores = [(n, difflib.SequenceMatcher(None, _norm(n), _norm(query)).ratio()) for n in names]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0]

# ------ External Tools ------
def list_current(date: Optional[str|None] = None) -> dict:
    """
    Input:
        date: Date to query for current medications
    Returns:
        A dictionary of current mediactions and dosages on a given date.
    """
    logger.info(f'Searching for current medications on date {date}')

    df = _load_df()
    if df is None:
        return {}
    
    # Parse target date (default to today if missing)
    if not date:
        try:
            logger.info(f'No date supplied for current medications. Defaulting to current date.')
            date = pd.Timestamp.now().strftime("%Y-%m-%d")
        except Exception:
            logger.error('Failed to get current date')
            return {}

    try:
        dt = pd.to_datetime(date)
        mask = (
            (df['date_start'] <= dt)
            & (
                (df['date_stop'] >= dt)
                | (df['date_stop'].isna())
                | ((df['date_updated'] >= dt))
                | (df['date_updated'].isna())
            )
        )
        df = df[mask].copy()
    except Exception:
        logger.warning(f'Failed to parse date when loading current medications: {date}')
        return {}
    
    # Keep 'current' in outputs so downstream tools can surface it
    drop_cols = [
        'date_stop',
        'date_updated',
        '__source_file',
    ]
    df_dict = df.drop(columns=drop_cols).to_dict(orient='records')

    logger.info(f'Found {len(df_dict)} current medications on date {date}')
    return df_dict


def list_medications() -> List[str]:
    """Return the unique list of medication names from the table."""
    df = _load_df()
    if df is None:
        return []
    cols = _detect_cols(df)
    name_col = cols["name_col"]  # type: ignore[index]
    if not name_col:
        logger.warning('No name column found in meds table')
        return []
    return sorted({str(x).strip() for x in df[name_col].dropna().unique() if str(x).strip()})


def get_medication_history(medication: str, fuzzy: bool = True, threshold: float = 0.6) -> List[Dict]:
    """Return the chronological dosing history for a medication.

    Args:
        medication: Name of the medication to search for.
        fuzzy: If True, use fuzzy matching to resolve the medication name.
    """
    logger.info(f'Searching for medication history for {medication} (fuzzy={fuzzy})')
    df = _load_df()
    if df is None:
        return []
    cols = _detect_cols(df)
    name_col = cols["name_col"]  # type: ignore[index]
    if not name_col:
        logger.warning('No name column found in meds table')
        return []

    # Resolve name (possibly fuzzy)
    target = medication
    meds = list_medications()
    if fuzzy and meds:
        best, score = _best_name_match(meds, medication)
        if score >= threshold:
            logger.info(f'Found best match for {medication}: {best} (score={score})')
            target = best
        else:
            logger.warning(f'No good match found for {medication} (score={score})')
            return []

    dff = df[df[name_col].astype(str).str.strip().str.lower() == str(target).strip().lower()].copy()
    if dff.empty:
        logger.warning(f'No medication history found for {medication}')
        return []

    events: List[Dict] = []
    def add_event(date_val, kind, row):
        try:
            date_s = str(pd.to_datetime(date_val).date())
        except Exception:
            date_s = str(date_val) if date_val is not None else ""
        if not date_s:
            return
        events.append({
            "name": str(row.get(name_col)),
            "date": date_s,
            "kind": kind,
            "dose": str(row.get(cols["dose_cols"][0])) if cols["dose_cols"] else "",
            "dose_unit": str(row.get(cols["dose_unit_cols"][0])) if cols["dose_unit_cols"] else "",
            "freq": str(row.get(cols["freq_cols"][0])) if cols["freq_cols"] else "",
            "freq_unit": str(row.get(cols["freq_unit_cols"][0])) if cols["freq_unit_cols"] else "",
            "current": str(row.get(cols["current_cols"][0])) if cols["current_cols"] else "",
        })

    for _, r in dff.iterrows():
        # Start events
        for c in cols["start_cols"]:
            add_event(r.get(c), "start", r)
        # Dose change / updated
        for c in cols["updated_cols"]:
            add_event(r.get(c), "dose_change", r)
        # Stop
        for c in cols["end_cols"]:
            add_event(r.get(c), "stop", r)

    if not events:
        return []
    events.sort(key=lambda e: e["date"])  # type: ignore
    return events


def dosage_on_date(medication: str, date: Optional[str] = None, fuzzy: bool = True) -> Dict:
    """Return the dose/frequency in effect for a medication on a given date.

    Args:
        medication: Name of the medication to search for.
        date: Date in YYYY-MM-DD to query. If None, defaults to today's date.
        fuzzy: If True, use fuzzy matching to resolve the medication name.
    """
    logger.info(f'Looking up dosage on {date} for {medication} (fuzzy={fuzzy})')

    # Reuse list_current() to get the active rows for the date
    rows = list_current(date=date) or []
    if not rows:
        return {}

    # Build a small DataFrame to reuse column detection
    try:
        dff = pd.DataFrame(rows)
    except Exception:
        return {}

    cols_map = _detect_cols(dff)
    name_col = cols_map.get('name_col')
    if not name_col or name_col not in dff.columns:
        return {}

    # Resolve name (possibly fuzzy) from the active set
    present_names = sorted({str(x).strip() for x in dff[name_col].dropna().unique() if str(x).strip()})
    target = medication
    if fuzzy and present_names:
        best, score = _best_name_match(present_names, medication)
        if score >= 0.6:
            target = best

    cand = dff[dff[name_col].astype(str).str.strip().str.lower() == str(target).strip().lower()].copy()
    if cand.empty:
        return {}

    # If multiple entries, choose the one with the latest start date if available
    start_col = None
    for c in ['start_date', 'date_start', 'start']:
        if c in cand.columns:
            start_col = c
            break
    if start_col:
        def _to_ts(x):
            try:
                return pd.to_datetime(x)
            except Exception:
                return pd.NaT
        cand['_sort_key'] = cand[start_col].apply(_to_ts)
        cand = cand.sort_values(['_sort_key'], ascending=[False])

    row = cand.iloc[0]

    def _safe_str(x):
        try:
            return '' if x is None else str(x)
        except Exception:
            return ''

    # Map fields from detected columns
    dose = _safe_str(row.get(cols_map['dose_cols'][0])) if cols_map['dose_cols'] else ''
    dose_unit = _safe_str(row.get(cols_map['dose_unit_cols'][0])) if cols_map['dose_unit_cols'] else ''
    freq = _safe_str(row.get(cols_map['freq_cols'][0])) if cols_map['freq_cols'] else ''
    freq_unit = _safe_str(row.get(cols_map['freq_unit_cols'][0])) if cols_map['freq_unit_cols'] else ''
    current = _safe_str(row.get(cols_map['current_cols'][0])) if cols_map['current_cols'] else ''

    return {
        'name': _safe_str(row.get(name_col)),
        'date': date,
        'dose': dose,
        'dose_unit': dose_unit,
        'freq': freq,
        'freq_unit': freq_unit,
        'current': current,
    }