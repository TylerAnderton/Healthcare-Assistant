# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Local-first healthcare RAG: ingests labs PDFs, meds CSV/XLSX, WHOOP CSVs → parquet tables → Chroma vectorstore → Streamlit UI + Ollama LLM.

## Setup & Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Ingest: `python -m app.ingestion.ingest_{labs,meds,whoop}`
Index: `python -m app.indexing.build_index`
Run: `streamlit run app/app.py [--logger.level=info]`
Test: `pytest -vv tests [--no-chat]`

## Architecture

**Data flow:** raw PDFs/CSVs → ingestion (vendor-detected, normalized) → parquet tables (`data/processed/tables/`) → text corpus (`data/processed/corpus/`) → Chroma vectorstore.

**Chat flow:** query → multi-source retrieval (MMR, dedup) → structured blocks (meds timeline, labs panel, WHOOP snapshot) + vector results → LLM context + few-shot + history → response w/ citations.

**Key modules:**
- `constants.py`: canonical schemas (LABS_PROCESSED_COLS, MEDS_PROCESSED_COLS), tuning knobs
- `app/chains/chat.py`: retrieval + prompting entry point (`answer_question()`)
- `app/ingestion/`: vendor detection + parsing
- `app/tools/`: structured data loaders (meds_tool, labs_tool, whoop_tool)
- `app/app.py`: Streamlit UI

## Config

Copy `.env.example` → `.env`. Key vars: `OLLAMA_MODEL`, `EMBEDDING_MODEL`, `MAX_CONTEXT_DOCS`, `ENABLE_TOOL_CALLING`, `PREFER_STRUCTURED`.

## Data Schema

**Labs:** analyte, value, unit, ref_low, ref_high, date, source, page, source_type, vendor, flag
**Meds:** name, dose, dose_unit, frequency, frequency_unit, date_start, date_updated, date_stop, current
**WHOOP:** date, sleep_score, recovery_score, strain, rhr_bpm, hrv_ms, activity, calories, etc.

## Testing Conventions

### TDD Workflow
- Always write failing tests BEFORE implementation
- Use AAA pattern: Arrange-Act-Assert
- One assertion per test when possible
- Test names describe behavior: "should_return_empty_when_no_items"

### Test-First Rules
- When I ask for a feature, write tests first
- Tests should FAIL initially (no implementation exists)
- Only after tests are written, implement minimal code to pass
