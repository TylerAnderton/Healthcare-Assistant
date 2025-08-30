import os
import pandas as pd
from typing import Dict

BASE = os.path.join(os.getenv("PROCESSED_DIR", "./data/processed"), "tables")


def quick_summary() -> Dict[str, int]:
    out = {}
    for name in ["sleeps", "workouts", "physiological_cycles", "journal_entries"]:
        fp = os.path.join(BASE, f"whoop_{name}.parquet")
        if os.path.exists(fp):
            try:
                df = pd.read_parquet(fp)
                out[name] = len(df)
            except Exception:
                pass
    return out
