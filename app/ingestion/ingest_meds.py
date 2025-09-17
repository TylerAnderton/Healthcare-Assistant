import argparse
import os
import glob
import pandas as pd
from datetime import datetime, UTC
import logging

from dotenv import load_dotenv
load_dotenv()

from app.constants import MEDS_PROCESSED_COLS, MEDS_DATE_COLS, MEDS_TABLE_FILE, MEDS_CORPUS_FILE, MEDS_COL_MAP
from typing import List, Optional

logger = logging.getLogger(__name__)

def ensure_dirs(base_out: str):
    os.makedirs(os.path.join(base_out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(base_out, "corpus"), exist_ok=True)


def load_any(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    raise ValueError("Unsupported file type: " + path)


def normalize_meds_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize an input medications dataframe to the canonical MEDS_PROCESSED_COLS schema
    using MEDS_RAW_COLS to resolve source column names.
    Date fields are parsed to pandas datetime.
    """
    # out = pd.DataFrame()
    # # Map canonical columns
    # for i, canon in enumerate(MEDS_PROCESSED_COLS):
    #     if canon == "__source_file":
    #         continue
    #     src = MEDS_RAW_COLS[i]
    #     if src in df.columns:
    #         out[canon] = df[src]
    #     else:
    #         logger.warning(f"Column {src} not found in input dataframe")
    #         out[canon] = pd.NA
    
    out = df.rename(columns=MEDS_COL_MAP)

    # Parse date columns
    for dc in MEDS_DATE_COLS:
        if dc in out.columns:
            try:
                out[dc] = pd.to_datetime(out[dc], errors="coerce")
            except Exception:
                logger.warning(f"Failed to parse date in {dc}")
                pass

    return out


def main(src: str, out: str):
    ensure_dirs(out)
    tables: List[pd.DataFrame] = []
    corpus_rows: List[Dict] = []

    for fp in sorted(glob.glob(os.path.join(src, "*"))):
        if not fp.lower().endswith((".csv", ".xlsx", ".xls")):
            continue
        try:
            raw_df = load_any(fp)
        except Exception as e:
            logger.error(f"Failed to read {fp}: {e}")
            continue
        df = normalize_meds_df(raw_df)
        df["__source_file"] = os.path.relpath(fp)
        tables.append(df)

        # Lightweight corpus rows
        for _, r in df.fillna("").iterrows():
            text = "; ".join([f"{col}: {r.get(col) or 'None'}" for col in MEDS_PROCESSED_COLS])
            corpus_rows.append(
                {
                    "text": text,
                    "source": os.path.relpath(fp),
                    "source_type": "meds",
                    "ingested_at": datetime.now(UTC).isoformat(timespec='seconds') + "Z",
                }
            )

    if tables:
        all_df = pd.concat(tables, ignore_index=True)
        all_df.to_parquet(os.path.join(out, "tables", MEDS_TABLE_FILE))
        logger.info(f"Wrote meds table: {len(all_df)} rows")
    else:
        logger.warning("No medications files found.")

    if corpus_rows:
        cdf = pd.DataFrame(corpus_rows)
        cdf.to_parquet(os.path.join(out, "corpus", MEDS_CORPUS_FILE))
        logger.info(f"Wrote meds corpus: {len(cdf)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_src = os.path.join(os.getenv("DATA_DIR", "./data"), "medications")
    parser.add_argument("--src", default=default_src)
    parser.add_argument("--out", default=os.getenv("PROCESSED_DIR", "./data/processed"))
    args = parser.parse_args()
    main(args.src, args.out)
