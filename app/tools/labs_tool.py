from typing import Optional, List
import os
import pandas as pd

# Placeholder: later implement structured tool with LangChain

TABLE_PATH = os.path.join(os.getenv("PROCESSED_DIR", "./data/processed"), "tables", "labs.parquet")


def latest_value(analyte: str) -> Optional[dict]:
    if not os.path.exists(TABLE_PATH):
        return None
    try:
        df = pd.read_parquet(TABLE_PATH)
    except Exception:
        return None
    if "analyte" not in df.columns or "value" not in df.columns:
        return None
    dff = df[df["analyte"].str.lower() == analyte.lower()].sort_values("date", ascending=False)
    if dff.empty:
        return None
    r = dff.iloc[0]
    return {"analyte": analyte, "value": r.get("value"), "unit": r.get("unit"), "date": r.get("date")}
