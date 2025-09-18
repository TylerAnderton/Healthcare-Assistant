# Tests for Healthcare Assistant

This suite validates retrieval accuracy against the gold-standard parquet tables in `data/processed/tables/`.

What we validate:

- Labs: correct values and units on the correct dates, plus the latest panel snapshot formatting.
- Meds: correct dose and frequency information on the correct dates in the structured meds timeline.
- WHOOP: recovery score and strain on the correct dates.

## Setup

1. Activate the project environment:

   ```bash
   source .venv/bin/activate
   ```

2. Install dependencies (ensure `pytest` is present):

   ```bash
   pip install -r requirements.txt
   ```

## Running the tests

```bash
# All tests (chat ON by default; pass --no-chat to skip chat tests)
pytest -vv tests

# Skip chat tests
pytest -vv tests --no-chat

# Or run specific tests
pytest -vv tests/test_labs_accuracy.py tests/test_meds_accuracy.py tests/test_whoop_accuracy.py

```

## Fixtures

Static fixtures live under `tests/fixtures/`:

- `labs.json`: Selected analytes and date-stamped values with units and ranges.
- `meds.json`: Selected medications and their start/change/stop events.
- `whoop.json`: Selected dates with recovery, strain, RHR, and HRV.
- `prompts.json`: Chat prompts with expected substrings for manual/chat validation.

These were sampled from the gold parquet files. If you update `data/processed/tables/`, you may need to regenerate fixtures.

## Generate/rebuild fixtures from parquet (CLI)

Use the fixture generator to rebuild JSON fixtures from the parquet tables:

```bash
python tests/generate_fixtures.py --whoop-days 7
```

This script writes updated files under `tests/fixtures/`.
