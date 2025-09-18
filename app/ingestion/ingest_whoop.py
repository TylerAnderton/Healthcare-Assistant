import argparse
import os
import pandas as pd
from datetime import datetime, UTC
import logging

from app.constants import (
    WHOOP_SLEEPS_COL_MAP,
    WHOOP_SLEEPS_PROCESSED_COLS,
    WHOOP_WORKOUTS_COL_MAP,
    WHOOP_WORKOUTS_PROCESSED_COLS,
    WHOOP_RECOVERY_COL_MAP,
    WHOOP_RECOVERY_PROCESSED_COLS,
    WHOOP_TABLE_FILES,
    WHOOP_CORPUS_FILE,
)

from dotenv import load_dotenv
load_dotenv()

RAW_DIR = os.getenv("RAW_DIR", "./data/raw")
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "./data/processed")

# logger = logging.getLogger(__name__)

def ensure_dirs(base_out: str):
    os.makedirs(os.path.join(base_out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(base_out, "corpus"), exist_ok=True)


def main(src: str, out: str):
    ensure_dirs(out)
    corpus_rows: List[Dict] = []

    # Load known CSVs if present
    # Build mapping from expected CSV filename -> standardized parquet filename
    csv_to_parquet = {f"{key}.csv": parquet for key, parquet in WHOOP_TABLE_FILES.items()}

    found_any = False

    for root, _, files in os.walk(src):
        for name in files:
            if name in csv_to_parquet:
                found_any = True
                fp = os.path.join(root, name)
                try:
                    df = pd.read_csv(fp)
                    print(f"Loaded {fp} with {len(df)} rows")
                except Exception as e:
                    print(f"Failed to read {fp}: {e}")
                    continue

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
                        print(f"Processing sleeps from {fp}")
                        df = df.rename(columns=WHOOP_SLEEPS_COL_MAP)[WHOOP_SLEEPS_PROCESSED_COLS]
                        try:
                            df[WHOOP_SLEEPS_PROCESSED_COLS[0]] = pd.to_datetime(df[WHOOP_SLEEPS_PROCESSED_COLS[0]], errors="coerce")
                        except Exception:
                            print(f"Failed to parse date in {WHOOP_SLEEPS_PROCESSED_COLS[0]}")
                            pass
                        
                        # unify date
                        for _, r in df.iterrows():
                            date = str(r.get(WHOOP_SLEEPS_PROCESSED_COLS[0]))
                            if not date:
                                continue
                            score = r.get(WHOOP_SLEEPS_PROCESSED_COLS[1])  # percent/score
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

                    elif "physiological_cycles" in nlow:
                        print(f"Processing physiological cycles from {fp}")
                        df = df.rename(columns=WHOOP_RECOVERY_COL_MAP)[WHOOP_RECOVERY_PROCESSED_COLS]
                        try:
                            df[WHOOP_RECOVERY_PROCESSED_COLS[0]] = pd.to_datetime(df[WHOOP_RECOVERY_PROCESSED_COLS[0]], errors="coerce")
                        except Exception:
                            print(f"Failed to parse date in {WHOOP_RECOVERY_PROCESSED_COLS[0]}")
                            pass

                        # unify date
                        for _, r in df.iterrows():
                            date = str(r.get(WHOOP_RECOVERY_PROCESSED_COLS[0]))
                            if not date:
                                continue
                            strain = r.get(WHOOP_RECOVERY_PROCESSED_COLS[1])
                            rec = r.get(WHOOP_RECOVERY_PROCESSED_COLS[2])
                            # rhr = r.get(WHOOP_RECOVERY_PROCESSED_COLS[3])
                            hrv = r.get(WHOOP_RECOVERY_PROCESSED_COLS[4])
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

                    elif "workouts" in nlow:
                        print(f"Processing workouts from {fp}")
                        df = df.rename(columns=WHOOP_WORKOUTS_COL_MAP)[WHOOP_WORKOUTS_PROCESSED_COLS]
                        try:
                            df[WHOOP_WORKOUTS_PROCESSED_COLS[0]] = pd.to_datetime(df[WHOOP_WORKOUTS_PROCESSED_COLS[0]], errors="coerce")
                        except Exception:
                            print(f"Failed to parse date in {WHOOP_WORKOUTS_PROCESSED_COLS[0]}")
                            pass

                        # unify date
                        for _, r in df.iterrows():
                            date = str(r.get(WHOOP_WORKOUTS_PROCESSED_COLS[0]))
                            if not date:
                                continue
                            activity = r.get(WHOOP_WORKOUTS_PROCESSED_COLS[1])
                            duration = r.get(WHOOP_WORKOUTS_PROCESSED_COLS[2])
                            strain = r.get(WHOOP_WORKOUTS_PROCESSED_COLS[3])
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
                                
                    out_fp = os.path.join(out, "tables", csv_to_parquet[nlow])
                    df.to_parquet(out_fp) 
                    print(f"Wrote WHOOP table {csv_to_parquet[nlow]}: {len(df)} rows")

                except Exception as _:
                    print(f"Failed to process {fp}")
                    pass

    if not found_any:
        print("No WHOOP CSVs found.")

    if corpus_rows:
        cdf = pd.DataFrame(corpus_rows)
        cdf.to_parquet(os.path.join(out, "corpus", WHOOP_CORPUS_FILE))
        print(f"Wrote WHOOP corpus: {len(cdf)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_src = os.path.join(RAW_DIR, "whoop")
    parser.add_argument("--src", default=default_src)
    parser.add_argument("--out", default=PROCESSED_DIR)
    args = parser.parse_args()
    main(args.src, args.out)
