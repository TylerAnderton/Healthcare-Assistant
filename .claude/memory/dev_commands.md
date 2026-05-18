---
name: Development Commands Reference
description: Full command reference for setup, ingestion, indexing, running, testing
type: reference
---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Data Ingestion

Run in order after populating `data/raw/labs/`, `data/raw/medications/`, `data/raw/whoop/`:

```bash
python -m app.ingestion.ingest_labs      # Parses PDFs; vendor-detects (LabCorp, LetsGetChecked, Ways2Well)
python -m app.ingestion.ingest_meds      # Parses CSV/XLSX; normalizes to canonical schema
python -m app.ingestion.ingest_whoop     # Parses CSVs; builds sleeps, workouts, recovery tables
```

Output: parquet tables in `data/processed/tables/` (labs.parquet, meds.parquet, whoop_*.parquet).

## Index Building

```bash
python -m app.indexing.build_index
```

Reads corpus from `data/processed/corpus/`, splits into chunks (1000 char, 150 overlap), embeds with HuggingFace model, stores in Chroma at `VECTORSTORE_DIR`.

Optional args: `--corpus DIR --store DIR --embedding_model NAME`

## Running UI

```bash
streamlit run app/app.py                          # Normal
streamlit run app/app.py --logger.level=info      # Verbose logging
streamlit run app/app.py --logger.level=debug     # Debug logging
```

Access at http://localhost:8501

## Testing

```bash
pytest -vv tests                         # All tests (chat ON by default)
pytest -vv tests --no-chat               # Skip chat tests (faster)
pytest -vv tests/test_labs_accuracy.py   # Single test file
pytest -vv tests/test_meds_accuracy.py
pytest -vv tests/test_whoop_accuracy.py
```

Chat tests slow; disabled by default. Use `--no-chat` flag to skip.

## Test Fixtures

Static fixtures in `tests/fixtures/` (labs.json, meds.json, whoop.json, prompts.json). Regenerate from parquet after data updates:

```bash
python tests/generate_fixtures.py --whoop-days 7
```

Script samples from gold parquet files, outputs JSON fixtures.
