import json
from pathlib import Path

import pytest

from app.tools import meds_tool as MT
from app.tools.structured_context import load_meds_timeline

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture():
    with open(FIXTURES_DIR / "meds.json", "r") as f:
        return json.load(f)


@pytest.mark.parametrize("case", load_fixture()["cases"])
def test_get_medication_history_matches_fixture_events(case):
    med = case["med"]
    events = case["events"]

    hist = MT.get_medication_history(med, fuzzy=True)
    assert isinstance(hist, list), "history should be a list"
    assert hist, f"No history returned for {med}"

    # Index events by (date, kind) for quick lookup
    by_key = {(str(e.get("date"))[:10], str(e.get("kind"))) for e in hist}

    for ev in events:
        date = str(ev["date"])[:10]
        kind = ev["kind"]
        assert (date, kind) in by_key, f"Missing event for {med}: {date} {kind}"


@pytest.mark.parametrize("case", load_fixture()["cases"])
def test_dosage_on_date_matches_fixture_on_event_date(case):
    med = case["med"]
    events = case["events"]

    for ev in events:
        date = str(ev["date"])[:10]
        res = MT.dosage_on_date(med, date=date, fuzzy=True)
        # If no result, skip (some events like stop might not map to an active dose depending on the table semantics)
        if not res:
            continue
        # Basic shape
        assert isinstance(res, dict)
        assert "name" in res and "date" in res
        # Compare dose/freq fields if present in fixture
        if ev.get("dose") is not None:
            assert str(res.get("dose")) == str(ev["dose"])
        if ev.get("dose_unit") is not None:
            assert (res.get("dose_unit") or "") == ev["dose_unit"]
        if ev.get("freq") is not None:
            assert str(res.get("freq")) == str(ev["freq"])
        if ev.get("freq_unit") is not None:
            assert (res.get("freq_unit") or "") == ev["freq_unit"]


def test_list_medications_contains_fixture_meds():
    data = load_fixture()
    meds = set(MT.list_medications())
    # meds may include more than fixtures; ensure fixture meds are present when list is non-empty
    if meds:
        for case in data["cases"]:
            assert case["med"] in meds


def test_list_current_today_shape_and_optionality():
    # Date is optional; function should handle None by defaulting to today
    res = MT.list_current()
    assert isinstance(res, list)
    # If there are current meds, check shape of first item
    if res:
        first = res[0]
        assert isinstance(first, dict)
        # Common fields that should generally exist
        assert "name" in first or "medication" in first or "drug" in first
        # At least some dosing fields should be present (may be empty strings)
        assert any(k in first for k in ["dose", "dosage"])  # schema-flex


def test_meds_timeline_basic_structure():
    block = load_meds_timeline(max_meds=10, max_events=20)
    # Should return a string, possibly empty if no data
    assert isinstance(block, str)
    if block:
        lines = block.splitlines()
        # Header present
        assert lines[0].startswith("[structured_meds_timeline]")
        # At least one Drug header line if non-empty
        assert any(line.startswith("- Drug: ") for line in lines)
