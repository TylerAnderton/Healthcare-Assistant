import argparse
import os
import glob
import pandas as pd
from datetime import datetime, UTC

from dotenv import load_dotenv
load_dotenv()

from app.ingestion.utils_pdf import extract_pdf_pages


def ensure_dirs(base_out: str):
    os.makedirs(os.path.join(base_out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(base_out, "corpus"), exist_ok=True)


def main(src: str, out: str):
    ensure_dirs(out)
    rows = []
    for fp in sorted(glob.glob(os.path.join(src, "*.pdf"))):
        try:
            pages = extract_pdf_pages(fp)
        except Exception as e:
            print(f"Failed to read {fp}: {e}")
            continue
        for p in pages:
            rows.append({
                "text": p.get("text", ""),
                "source": os.path.relpath(fp),
                "page": p.get("page"),
                "source_type": "labs",
                "ingested_at": datetime.now(UTC).isoformat(timespec='seconds') + "Z",
            })

    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(os.path.join(out, "corpus", "labs_corpus.parquet"))
        print(f"Wrote corpus: {len(df)} rows")
    else:
        print("No lab PDFs found or text extracted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_src = os.path.join(os.getenv("DATA_DIR", "./data"), "labs")
    parser.add_argument("--src", default=default_src)
    parser.add_argument("--out", default=os.getenv("PROCESSED_DIR", "./data/processed"))
    args = parser.parse_args()
    main(args.src, args.out)
