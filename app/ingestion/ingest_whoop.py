import argparse
import os
import glob
import pandas as pd
from datetime import datetime, UTC
import logging

from app.tools.whoop_tool import pick_column, parse_date_any

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

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
                    logger.error(f"Failed to read {fp}: {e}")
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
                        logger.info(f"Processing sleeps from {fp}")
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
                            score = pick_column(r, ["sleep_performance", "sleep_score", "score", "Sleep performance %"])  # percent/score
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
                        logger.info(f"Processing physiological cycles from {fp}")
                        dff = df.copy()
                        if "date" not in dff.columns:
                            dff["date"] = dff.apply(lambda r: parse_date_any(r, [
                                "date", "cycle_date", "start", "day", "Cycle start time"
                            ]), axis=1)
                        for _, r in dff.iterrows():
                            date = str(r.get("date")) if pd.notna(r.get("date")) else ""
                            if not date:
                                continue
                            strain = pick_column(r, ["strain", "Day Strain"])
                            rec = pick_column(r, ["recovery_score", "recovery", "score", "Recovery score %"])  # percent/score
                            # rhr = pick_column(r, ["resting_heart_rate", "rhr", "Resting heart rate (bpm)"])  # bpm
                            hrv = pick_column(r, ["hrv", "hrv_rmssd_milli", "rmssd", "Heart rate variability (ms)"])  # ms
                            parts = [
                                f"WHOOP recovery {date}:",
                                f"strain {strain}" if strain not in (None, "") else "",
                                f"recovery {rec}" if rec not in (None, "") else "",
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

                    if "workouts" in nlow:
                        logger.info(f"Processing workouts from {fp}")
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
                            activity = pick_column(r, ["sport", "activity_type", "activity_type_name", "Activity type"])
                            duration = pick_column(r, ["duration", "workout_duration", "Duration (minutes)"])
                            strain = pick_column(r, ["strain", "Activity Strain"])
                            parts = [
                                f"WHOOP workout {date}:",
                                f"activity {activity}" if activity not in (None, "") else "",
                                f"duration {duration}" if duration not in (None, "") else "",
                                f"strain {strain}" if strain not in (None, "") else "",
                            ]
                            line = " ".join([p for p in parts if p]).strip()
                            if line and line != f"WHOOP workout {date}:":
                                corpus_rows.append({
                                    "text": line,
                                    "source": os.path.relpath(fp),
                                    "source_type": "whoop",
                                    "whoop_type": "workout",
                                    "date": date,
                                    "ingested_at": datetime.now(UTC).isoformat(timespec='seconds') + "Z",
                                })

                except Exception as _:
                    # keep ingestion resilient; skip enriched corpus on any error
                    pass

    if not found_any:
        logger.warning("No WHOOP CSVs found.")

    if corpus_rows:
        cdf = pd.DataFrame(corpus_rows)
        cdf.to_parquet(os.path.join(out, "corpus", "whoop_corpus.parquet"))
        logger.info(f"Wrote WHOOP corpus: {len(cdf)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_src = os.path.join(os.getenv("DATA_DIR", "./data"), "whoop")
    parser.add_argument("--src", default=default_src)
    parser.add_argument("--out", default=os.getenv("PROCESSED_DIR", "./data/processed"))
    args = parser.parse_args()
    main(args.src, args.out)
