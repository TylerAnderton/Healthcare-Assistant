from typing import Optional, List, Dict, Tuple
import os
import pandas as pd
import logging
import difflib

from app.constants import MEDS_DATE_COLS, MEDS_TABLE_FILE, MEDS_PROCESSED_COLS

logger = logging.getLogger(__name__)

TABLE_PATH = os.path.join(os.getenv("PROCESSED_DIR", "./data/processed"), "tables", MEDS_TABLE_FILE)


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
            (df[MEDS_DATE_COLS[0]] <= dt)
            & (
                (df[MEDS_DATE_COLS[1]] >= dt)
                | (df[MEDS_DATE_COLS[1]].isna())
                | ((df[MEDS_DATE_COLS[2]] >= dt))
                | (df[MEDS_DATE_COLS[2]].isna())
            )
        )
        df = df[mask].copy()
    except Exception:
        logger.warning(f'Failed to parse date when loading current medications: {date}')
        return {}
    
    # Keep 'current' in outputs so downstream tools can surface it
    drop_cols = [
        MEDS_DATE_COLS[1],
        MEDS_DATE_COLS[2],
        MEDS_PROCESSED_COLS[-1],
    ]
    df_dict = df.drop(columns=drop_cols).to_dict(orient='records')

    logger.info(f'Found {len(df_dict)} current medications on date {date}')
    return df_dict


def list_medications() -> List[str]:
    """Return the unique list of medication names from the table."""
    df = _load_df()
    if df is None:
        return []
    if MEDS_PROCESSED_COLS[0] not in df.columns:
        logger.warning('No canonical name column found in meds table')
        return []
    return sorted({str(x).strip() for x in df[MEDS_PROCESSED_COLS[0]].dropna().unique() if str(x).strip()})


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
    if MEDS_PROCESSED_COLS[0] not in df.columns:
        logger.warning('No canonical name column found in meds table')
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

    dff = df[df[MEDS_PROCESSED_COLS[0]].astype(str).str.strip().str.lower() == str(target).strip().lower()].copy()
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
        event = {
            "date": date_s,
            "kind": kind,
        }
        for col in MEDS_PROCESSED_COLS[:-1]:
            event[col] = str(row.get(col) or "")
        events.append(event)

    for _, r in dff.iterrows():
        add_event(r.get(MEDS_DATE_COLS[0]), "start", r)
        add_event(r.get(MEDS_DATE_COLS[1]), "dose_change", r)
        add_event(r.get(MEDS_DATE_COLS[2]), "stop", r)

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

    # Build a small DataFrame and assume canonical columns
    try:
        dff = pd.DataFrame(rows)
    except Exception:
        return {}
    if MEDS_PROCESSED_COLS[0] not in dff.columns:
        return {}

    # Resolve name (possibly fuzzy) from the active set
    present_names = sorted({str(x).strip() for x in dff[MEDS_PROCESSED_COLS[0]].dropna().unique() if str(x).strip()})
    target = medication
    if fuzzy and present_names:
        best, score = _best_name_match(present_names, medication)
        if score >= 0.6:
            target = best

    cand = dff[dff[MEDS_PROCESSED_COLS[0]].astype(str).str.strip().str.lower() == str(target).strip().lower()].copy()
    if cand.empty:
        return {}

    # If multiple entries, choose the one with the latest start date if available
    start_col = None
    start_col = MEDS_DATE_COLS[0] if MEDS_DATE_COLS[0] in cand.columns else None
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

    return_dict = {'lookup_date': date}
    for col in MEDS_PROCESSED_COLS:
        return_dict[col] = _safe_str(row.get(col))

    return return_dict