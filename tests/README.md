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

### One-shot runner (recommended)

```bash
python tests/run_all_tests.py
```

- Chat tests are enabled by default. If you do not have a local LLM and vector index ready, disable chat tests:

```bash
python tests/run_all_tests.py --no-chat
```

### Running individual test files via Python

```bash
# Labs
python tests/test_labs_accuracy.py

# Meds
python tests/test_meds_accuracy.py

# WHOOP
python tests/test_whoop_accuracy.py

# Chat prompts (chat is ON by default; disable with --no-chat)
python tests/test_chat_prompts.py            # runs chat tests
python tests/test_chat_prompts.py --no-chat  # skips chat tests
```

### Running with pytest directly

```bash
# All tests (chat ON by default; pass --no-chat to skip chat tests)
pytest -q tests

# Or only non-chat tests
pytest -q tests/test_labs_accuracy.py tests/test_meds_accuracy.py tests/test_whoop_accuracy.py

# Skip chat tests
pytest -q tests --no-chat
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

## Regenerating fixtures (optional)

Use a short Python snippet to sample from parquet and (re)write JSON fixtures. Run inside the venv:

```bash
python - << 'PY'
import os, json, pandas as pd
base = 'data/processed/tables'

# Example: sample WHOOP recovery for recent dates
wc = pd.read_parquet(os.path.join(base, 'whoop_physiological_cycles.parquet'))
# Add date col if missing
if 'date' not in wc.columns:
    date_candidates = [c for c in wc.columns if str(c).lower() in ['date','cycle_date','start','day','cycle start time']]
    def norm_date(row):
        for c in date_candidates:
            v = row.get(c)
            if pd.notna(v):
                s = str(v)
                if 'T' in s: s = s.split('T',1)[0]
                return s[:10]
        return ''
    wc = wc.copy(); wc['date'] = wc.apply(norm_date, axis=1)

dff = wc[wc['date'].astype(str)!=''].sort_values('date').tail(7)
cols_map = {
    'strain': ['strain','Day Strain'],
    'recovery_score': ['recovery_score','recovery','score','Recovery score %'],
    'rhr_bpm': ['resting_heart_rate','rhr','Resting heart rate (bpm)'],
    'hrv_ms': ['hrv','hrv_rmssd_milli','rmssd','Heart rate variability (ms)'],
}

def pick(row, cands):
    for c in cands:
        if c in wc.columns and pd.notna(row.get(c)):
            return row.get(c)
    return None

rows = []
for _, r in dff.iterrows():
    rows.append({
        'date': str(r.get('date')),
        'strain': pick(r, cols_map['strain']),
        'recovery_score': pick(r, cols_map['recovery_score']),
        'rhr_bpm': pick(r, cols_map['rhr_bpm']),
        'hrv_ms': pick(r, cols_map['hrv_ms']),
    })

dst = 'tests/fixtures/whoop.json'
with open(dst, 'w') as f:
    json.dump({'recovery_cases': rows}, f, indent=2)
print('Wrote', dst)
PY
```
