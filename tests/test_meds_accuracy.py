import json
from pathlib import Path

from app.tools.structured_context import load_meds_timeline

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture():
    with open(FIXTURES_DIR / "meds.json", "r") as f:
        return json.load(f)


def _parse_timeline(block: str):
    by_med = {}
    current_med = None
    for raw in block.splitlines():
        line = raw.rstrip("\n")
        if line.startswith("- Drug: "):
            current_med = line.split(": ", 1)[1].strip()
            by_med.setdefault(current_med, [])
        elif current_med and line.startswith("  - "):
            by_med[current_med].append(line.strip())
    return by_med


def test_meds_timeline_contains_expected_events():
    data = load_fixture()
    block = load_meds_timeline(max_meds=50, max_events=120)
    assert block, "Empty meds timeline block"
    parsed = _parse_timeline(block)

    for case in data["cases"]:
        med = case["med"]
        assert med in parsed, f"Drug header missing for {med}"
        lines = parsed[med]
        assert lines, f"No events listed for {med}"
        for ev in case["events"]:
            date = str(ev["date"])[:10]
            kind = ev["kind"]
            dose_s = f"{ev['dose']} {ev['dose_unit']}".strip()
            found = any((date in ln and f" {kind}" in ln and dose_s in ln) for ln in lines)
            assert found, f"Missing event for {med}: {date} {kind} with {dose_s}"


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
