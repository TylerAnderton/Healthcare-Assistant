from typing import Optional
import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

TABLE_PATH = os.path.join(os.getenv("PROCESSED_DIR", "./data/processed"), "tables", "meds.parquet")


def list_current(date: Optional[str|None] = None) -> dict:
    """
    Input:
        date: Date to query for current medications
    Returns:
        A dictionary of current mediactions and dosages on a given date.
    """
    logger.info(f'Searching for current medications on date {date}')

    if not os.path.exists(TABLE_PATH):
        logger.warning(f'Meds table not found at {TABLE_PATH}')
        return {}
    try:
        df = pd.read_parquet(TABLE_PATH)
    except Exception:
        logger.warning(f'Failed to load meds table at {TABLE_PATH}')
        return {}
    
    if not date:
        logger.info(f'No date supplied for curernt mediations. Defaulting to current date.')
        date = pd.Timestamp.now().strftime("%Y-%m-%d")

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
    
    drop_cols = [
        'date_stop',
        'date_updated',
        'current',
        '__source_file',
    ]
    df_dict = df.drop(columns=drop_cols).to_dict(orient='records')

    logger.info(f'Found {len(df_dict)} current medications on date {date}')
    return df_dict