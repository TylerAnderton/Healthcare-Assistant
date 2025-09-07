import argparse
import os
import glob
import pandas as pd
from datetime import datetime, UTC

from app.tools.whoop_tool import pick_column, parse_date_any

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

                # Lightweight file corpus entry (kept for provenance)
                text = f"WHOOP file {name} with {len(df)} rows; columns: {', '.join(map(str, df.columns[:10]))}"
                corpus_rows.append({
                    "text": text,
                    "source": os.path.relpath(fp),
                    "source_type": "whoop",
                    "ingested_at": datetime.now(UTC).isoformat(timespec='seconds') + "Z",
                })

                # Enriched per-day key metrics for sleeps and recovery; still pretty minimal for the corpus
                nlow = name.lower()
                try:
                    if "sleep" in nlow:
                        dff = df.copy()
                        # unify date
                        if "date" not in dff.columns:
                            dff["date"] = dff.apply(lambda r: parse_date_any(r, [
                                "date", "day", "Cycle start time"
                            ]), axis=1)
                        for _, r in dff.iterrows():
                            date = str(r.get("date")) if pd.notna(r.get("date")) else ""
                            if not date:
                                continue
                            score = pick_column(r, ["sleep_performance", "sleep_score", "score", "Sleep score %"])  # percent/score
                            parts = [
                                f"WHOOP sleep {date}:",
                                f"score {score}" if score not in (None, "") else "",
                            ]
                            line = " ".join([p for p in parts if p]).strip()
                            if line and line != f"WHOOP sleep {date}:":
                                corpus_rows.append({
                                    "text": line,
                                    "source": os.path.relpath(fp),
                                    "source_type": "whoop",
                                    "whoop_type": "sleep",
                                    "date": date,
                                    "ingested_at": datetime.now(UTC).isoformat(timespec='seconds') + "Z",
                                })

                    if "physiological_cycles" in nlow or "physiological" in nlow or "recovery" in nlow:
                        dff = df.copy()
                        if "date" not in dff.columns:
                            dff["date"] = dff.apply(lambda r: parse_date_any(r, [
                                "date", "cycle_date", "start", "day", "Cycle start time"
                            ]), axis=1)
                        for _, r in dff.iterrows():
                            date = str(r.get("date")) if pd.notna(r.get("date")) else ""
                            if not date:
                                continue
                            rec = pick_column(r, ["recovery_score", "recovery", "score", "Recovery score %"])  # percent/score
                            # rhr = pick_column(r, ["resting_heart_rate", "rhr", "Resting heart rate (bpm)"])  # bpm
                            hrv = pick_column(r, ["hrv", "hrv_rmssd_milli", "rmssd", "Heart rate variability (ms)"])  # ms
                            parts = [
                                f"WHOOP recovery {date}:",
                                f"score {rec}" if rec not in (None, "") else "",
                                # f"RHR {rhr} bpm" if rhr not in (None, "") else "",
                                f"HRV {hrv} ms" if hrv not in (None, "") else "",
                            ]
                            line = " ".join([p for p in parts if p]).strip()
                            if line and line != f"WHOOP recovery {date}:":
                                corpus_rows.append({
                                    "text": line,
                                    "source": os.path.relpath(fp),
                                    "source_type": "whoop",
                                    "whoop_type": "recovery",
                                    "date": date,
                                    "ingested_at": datetime.now(UTC).isoformat(timespec='seconds') + "Z",
                                })
                except Exception as _:
                    # keep ingestion resilient; skip enriched corpus on any error
                    pass

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
