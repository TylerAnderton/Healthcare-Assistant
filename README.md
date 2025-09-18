# Healthcare Assistant (Local, Ollama)

A local-first chatbot to understand your healthcare data (labs PDFs, medications CSV/XLSX, WHOOP CSVs) using LangChain + Ollama + Chroma.

## Features
- Ingests:
  - PDFs in `data/labs/`
  - Meds CSV/XLSX in `data/medications/`
  - WHOOP CSVs in `data/whoop/`
- Builds a vector index (Chroma) from a lightweight text corpus.
- Streamlit chat UI with basic RAG and citations.
- No external API costs (uses Ollama + sentence-transformers).

## Quickstart
1. Install Ollama and a model:
   - https://ollama.com/download
   - Recommended chat model: `ollama pull qwen3:14b`
2. Python env
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`
3. Configure env
   - `cp .env.example .env` and adjust if needed
4. Ingest data
   - `python -m app.ingestion.ingest_labs`
   - `python -m app.ingestion.ingest_meds`
   - `python -m app.ingestion.ingest_whoop`
5. Build index
   - `python -m app.indexing.build_index`
6. Run UI
   - `streamlit run app/app.py`
   - Use the `--logger.level=info` flag to see more detailed logging

## Notes
- This app does not provide medical advice. Use for education and discussion with clinicians.
- If no corpus/index exists, the chat will prompt you to run ingestion+indexing.
