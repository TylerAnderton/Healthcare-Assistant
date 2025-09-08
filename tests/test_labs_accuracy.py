import json
import os
from pathlib import Path

import pytest
import pandas as pd

from app.tools import labs_tool as LT
from app.tools.structured_context import load_labs_panel

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture():
    with open(FIXTURES_DIR / "labs.json", "r") as f:
        return json.load(f)


@pytest.mark.parametrize("case", load_fixture()["cases"])
def test_labs_history_contains_all_fixture_points(case):
    analyte = case["analyte"]
    expected = case["values"]
    hist = LT.history(analyte, ascending=True)
    assert hist, f"No history returned for {analyte}"
    # index by date for quick lookup
    by_date = {str(r.get("date")): r for r in hist}
    for item in expected:
        d = str(item["date"])  # iso string
        assert d in by_date, f"Missing date {d} for {analyte} in history"
        got = by_date[d]
        # Compare value and unit
        assert float(got.get("value")) == pytest.approx(float(item["value"]), rel=0, abs=1e-6)
        if item.get("unit") is not None:
            assert (got.get("unit") or "") == item["unit"]


def test_labs_latest_value_matches_fixture_last_points():
    data = load_fixture()
    for case in data["cases"]:
        analyte = case["analyte"]
        last = case["values"][-1]
        lv = LT.latest_value(analyte)
        assert lv is not None, f"latest_value returned None for {analyte}"
        assert str(lv.get("date")) == str(last["date"])  # exact date match
        assert float(lv.get("value")) == pytest.approx(float(last["value"]), rel=0, abs=1e-6)
        if last.get("unit") is not None:
            assert (lv.get("unit") or "") == last["unit"]


def test_labs_summary_last_date_and_value():
    data = load_fixture()
    for case in data["cases"]:
        analyte = case["analyte"]
        last = case["values"][-1]
        summ = LT.summary(analyte)
        assert summ is not None
        assert str(summ.get("last_date")) == str(last["date"])  # last date
        assert float(summ.get("last_value")) == pytest.approx(float(last["value"]), rel=0, abs=1e-6)
        if last.get("unit") is not None:
            assert (summ.get("unit") or "") == last["unit"]


def test_labs_panel_includes_latest_date_row_for_known_analyte():
    """Validate that the labs panel block contains the latest date header
    and at least one analyte/value/unit line from that same date.
    This avoids hard-coding a specific date/value which may change as data updates.
    """
    # Determine latest lab date from the gold table
    processed_dir = os.getenv("PROCESSED_DIR", "./data/processed")
    labs_path = os.path.join(processed_dir, "tables", "labs.parquet")
    assert os.path.exists(labs_path), f"Missing labs table at {labs_path}"
    df = pd.read_parquet(labs_path)
    assert not df.empty
    dff = df[df["date"].notna()].copy()
    assert not dff.empty
    latest_date = sorted(dff["date"].astype(str).unique())[-1]
    # Build expected snippets (analyte: value [unit]) for rows from that date
    from_date = dff[dff["date"].astype(str) == latest_date].copy()
    snippets = []
    for _, r in from_date.iterrows():
        analyte = str(r.get("analyte"))
        val = str(r.get("value"))
        unit = str(r.get("unit")) if r.get("unit") is not None else ""
        snippets.append(f"{analyte}: {val} {unit}".strip())

    # Request a large panel to avoid truncating out expected analytes
    block = load_labs_panel(max_rows=2000)
    assert block, "Empty labs panel block"
    assert "[structured_labs_panel]" in block
    assert f"Date: {latest_date}" in block
    # Verify the block contains at least one analyte/value[/unit] line from that date
    assert any(sn in block for sn in snippets), (
        "None of the expected analyte/value lines for the latest date were found in the panel.\n"
        f"Latest date: {latest_date}\n"
        f"Tried snippets like: {snippets[:5]} ...\n"
        f"Panel was:\n{block}"
    )


def main(argv=None):
    import sys
    import pytest as _pytest
    from pathlib import Path as _Path
    test_path = str(_Path(__file__).resolve())
    opts = [test_path]
    rc = _pytest.main(opts if argv is None else argv + [test_path])
    sys.exit(rc)


if __name__ == "__main__":
    main()
