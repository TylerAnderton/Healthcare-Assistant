import argparse
import os
import glob
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime, UTC
import logging

import fitz  # PyMuPDF
from dotenv import load_dotenv
load_dotenv()

from app.ingestion.utils_pdf import extract_pdf_pages
from app.ingestion.vendors import labcorp as v_labcorp
from app.ingestion.vendors import letsgetchecked as v_lgc
from app.ingestion.vendors import ways2well as v_w2w

logger = logging.getLogger(__name__)

def ensure_dirs(base_out: str):
    os.makedirs(os.path.join(base_out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(base_out, "corpus"), exist_ok=True)


def detect_vendor(filepath: str, first_page_text: str) -> Optional[str]:
    name = os.path.basename(filepath).lower()
    if name.startswith("labcorp_") or "date collected" in first_page_text.lower(): # "date collected" might not be the best to use
        logger.info(f"Detected LabCorp: {filepath}")
        return "labcorp"
    if "letsgetchecked" in name or "lets_get_checked" in name:
        logger.info(f"Detected LetsGetChecked: {filepath}")
        return "letsgetchecked"
    if "ways2well" in name or "ways2well" in first_page_text.lower():
        logger.info(f"Detected Ways2Well: {filepath}")
        return "ways2well"
    return None


def main(src: str, out: str):
    ensure_dirs(out)
    corpus_rows: List[Dict] = []
    structured_rows: List[Dict] = []
    for fp in sorted(glob.glob(os.path.join(src, "*.pdf"))):
        # Always extract corpus text
        try:
            pages = extract_pdf_pages(fp)
        except Exception as e:
            logger.error(f"Failed to read {fp}: {e}")
            continue
        for p in pages:
            corpus_rows.append({
                "text": p.get("text", ""),
                "source": os.path.relpath(fp),
                "page": p.get("page"),
                "source_type": "labs",
                "ingested_at": datetime.now(UTC).isoformat(timespec='seconds') + "Z",
            })

        # Vendor-specific structured extraction
        try:
            with fitz.open(fp) as doc:
                fp_text = (doc[0].get_text("text") or "") if doc.page_count else ""
                vendor = detect_vendor(fp, fp_text) or "unknown"
                vendor_rows: List[Dict] = []
                if vendor == "labcorp":
                    vendor_rows = v_labcorp.extract_rows(doc, fp)
                elif vendor == "letsgetchecked":
                    # vendor_rows = v_lgc.extract_rows(doc, fp)
                    vendor_rows = [] # TODO: improve LetsGetChecked parsing
                # TODO: ways2well
                elif vendor == "ways2well":
                    vendor_rows = v_w2w.extract_rows(doc, fp)
                else:
                    vendor_rows = []
                # Normalize and attach common fields
                for r in vendor_rows:
                    r["source"] = os.path.relpath(fp)
                    r["source_type"] = "labs"
                structured_rows.extend(vendor_rows)
        except Exception as e:
            logger.error(f"Failed to parse structured rows for {fp}: {e}")

    # Write corpus
    if corpus_rows:
        df = pd.DataFrame(corpus_rows)
        df.to_parquet(os.path.join(out, "corpus", "labs_corpus.parquet"))
        logger.info(f"Wrote corpus: {len(df)} rows")
    else:
        logger.warning("No lab PDFs found or text extracted.")

    # Write structured table
    if structured_rows:
        tdf = pd.DataFrame(structured_rows)
        # Ensure columns exist
        for col in [
            "analyte",
            "value",
            "unit",
            "ref_low",
            "ref_high",
            "date",
            "source",
            "page",
            "source_type",
            "vendor",
            "flag",
        ]:
            if col not in tdf.columns:
                tdf[col] = None
        tdf.to_parquet(os.path.join(out, "tables", "labs.parquet"))
        logger.info(f"Wrote labs table: {len(tdf)} rows")
    else:
        logger.warning("No structured labs parsed (non-supported vendors or parsing failed).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_src = os.path.join(os.getenv("DATA_DIR", "./data/raw"), "labs")
    parser.add_argument("--src", default=default_src)
    parser.add_argument("--out", default=os.getenv("PROCESSED_DIR", "./data/processed"))
    args = parser.parse_args()
    main(args.src, args.out)
