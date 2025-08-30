from typing import List
import os
import pandas as pd

TABLE_PATH = os.path.join(os.getenv("PROCESSED_DIR", "./data/processed"), "tables", "meds.parquet")


def list_current(date: str = None) -> List[str]:
    if not os.path.exists(TABLE_PATH):
        return []
    try:
        df = pd.read_parquet(TABLE_PATH)
    except Exception:
        return []
    cols = [c.lower() for c in df.columns]
    # naive result
    names = []
    for _, r in df.fillna("").iterrows():
        n = r.get("name") or r.get("medication") or r.get("drug")
        if n:
            names.append(str(n))
    return sorted(set(names))
