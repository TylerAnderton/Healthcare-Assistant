import argparse
import os
import glob
import pandas as pd
from datetime import datetime, UTC

from dotenv import load_dotenv
load_dotenv()


def ensure_dirs(base_out: str):
    os.makedirs(os.path.join(base_out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(base_out, "corpus"), exist_ok=True)


def main(src: str, out: str):
    ensure_dirs(out)
    corpus_rows = []

    # Load known CSVs if present
    targets = [
        "sleeps.csv",
        "workouts.csv",
        "physiological_cycles.csv",
        "journal_entries.csv",
    ]

    found_any = False
    for root, _, files in os.walk(src):
        for name in files:
            if name in targets:
                found_any = True
                fp = os.path.join(root, name)
                try:
                    df = pd.read_csv(fp)
                except Exception as e:
                    print(f"Failed to read {fp}: {e}")
                    continue
                # Save tables as-is for now
                out_fp = os.path.join(out, "tables", f"whoop_{os.path.splitext(name)[0]}.parquet")
                df.to_parquet(out_fp)

                # Lightweight corpus entries
                text = f"WHOOP file {name} with {len(df)} rows; columns: {', '.join(map(str, df.columns[:10]))}"
                corpus_rows.append({
                    "text": text,
                    "source": os.path.relpath(fp),
                    "source_type": "whoop",
                    "ingested_at": datetime.now(UTC).isoformat(timespec='seconds') + "Z",
                })

    if not found_any:
        print("No WHOOP CSVs found.")

    if corpus_rows:
        cdf = pd.DataFrame(corpus_rows)
        cdf.to_parquet(os.path.join(out, "corpus", "whoop_corpus.parquet"))
        print(f"Wrote WHOOP corpus: {len(cdf)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_src = os.path.join(os.getenv("DATA_DIR", "./data"), "whoop")
    parser.add_argument("--src", default=default_src)
    parser.add_argument("--out", default=os.getenv("PROCESSED_DIR", "./data/processed"))
    args = parser.parse_args()
    main(args.src, args.out)
