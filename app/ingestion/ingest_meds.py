import argparse
import os
import glob
import pandas as pd
from datetime import datetime, UTC
import logging

from dotenv import load_dotenv
load_dotenv()

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


def main(src: str, out: str):
    ensure_dirs(out)
    tables = []
    corpus_rows = []

    for fp in sorted(glob.glob(os.path.join(src, "*"))):
        if not fp.lower().endswith((".csv", ".xlsx", ".xls")):
            continue
        try:
            df = load_any(fp)
        except Exception as e:
            logger.error(f"Failed to read {fp}: {e}")
            continue
        df["__source_file"] = os.path.relpath(fp)
        tables.append(df)

        # Lightweight corpus rows
        for _, r in df.fillna("").iterrows():
            name = str(r.get("name") or r.get("medication") or r.get("drug") or "medication")
            dose = str(r.get("dose") or r.get("dosage") or "")
            dose_unit = str(r.get("dose_unit") or "")
            freq = str(r.get("frequency") or r.get("freq") or r.get("dose_frequency") or "")
            freq_unit = str(r.get("frequency_unit") or r.get("dose_frequency_unit") or "")
            start = str(r.get("start_date") or r.get("date_start") or r.get("start") or "")
            updated = str(r.get("dose_updated") or r.get("date_updated") or r.get("updated") or r.get("date_changed") or "")
            end = str(r.get("end_date") or r.get("date_stop") or r.get("end") or "")
            current = str(r.get("current") or "")

            text = f"Medication: {name}; Dose: {dose} {dose_unit}; Frequency: {freq} {freq_unit}; Start: {start}; Updated: {updated}; End: {end}; Current: {current}."
            corpus_rows.append({
                "text": text,
                "source": os.path.relpath(fp),
                "source_type": "meds",
                "ingested_at": datetime.now(UTC).isoformat(timespec='seconds') + "Z",
            })

    if tables:
        all_df = pd.concat(tables, ignore_index=True)
        all_df.to_parquet(os.path.join(out, "tables", "meds.parquet"))
        logger.info(f"Wrote meds table: {len(all_df)} rows")
    else:
        logger.warning("No medications files found.")

    if corpus_rows:
        cdf = pd.DataFrame(corpus_rows)
        cdf.to_parquet(os.path.join(out, "corpus", "meds_corpus.parquet"))
        logger.info(f"Wrote meds corpus: {len(cdf)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_src = os.path.join(os.getenv("DATA_DIR", "./data"), "medications")
    parser.add_argument("--src", default=default_src)
    parser.add_argument("--out", default=os.getenv("PROCESSED_DIR", "./data/processed"))
    args = parser.parse_args()
    main(args.src, args.out)
