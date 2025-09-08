import json
from pathlib import Path

import pytest

from app.tools.whoop_tool import recovery as whoop_recovery

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture():
    with open(FIXTURES_DIR / "whoop.json", "r") as f:
        return json.load(f)


def _find_by_date(rows, date):
    date = str(date)
    return [r for r in rows if str(r.get("date")) == date]


@pytest.mark.parametrize("case", load_fixture()["recovery_cases"])
def test_whoop_recovery_and_strain_on_dates(case):
    rows = whoop_recovery(ascending=True)
    assert rows, "Empty WHOOP recovery data"
    matches = _find_by_date(rows, case["date"])
    assert matches, f"No WHOOP recovery row found for date {case['date']}"
    # Allow any matching row for the date (CSV exports may duplicate same day rows)
    assert any(
        (
            (m.get("recovery_score") is None or pytest.approx(float(case["recovery_score"])) == float(m.get("recovery_score")))
            and (m.get("strain") is None or pytest.approx(float(case["strain"])) == float(m.get("strain")))
            and (m.get("rhr_bpm") is None or pytest.approx(float(case["rhr_bpm"])) == float(m.get("rhr_bpm")))
            and (m.get("hrv_ms") is None or pytest.approx(float(case["hrv_ms"])) == float(m.get("hrv_ms")))
        )
        for m in matches
    ), f"WHOOP metrics mismatch for date {case['date']}"


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
