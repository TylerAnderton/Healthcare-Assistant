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
   - Recommended chat model: `ollama pull llama3.1:8b-instruct`
   - Alternatives: `ollama pull qwen2.5:7b-instruct` (faster) or `ollama pull llama3.1:70b-instruct` (heavier)
2. Python env
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`
3. Configure env
   - `cp .env.example .env` and adjust if needed
4. Ingest data
   - `python app/ingestion/ingest_labs.py --src ./data/labs --out ./data/processed`
   - `python app/ingestion/ingest_meds.py --src ./data/medications --out ./data/processed`
   - `python app/ingestion/ingest_whoop.py --src ./data/whoop --out ./data/processed`
5. Build index
   - `python app/indexing/build_index.py --corpus ./data/processed/corpus --store ./data/processed/vectorstore`
6. Run UI
   - `streamlit run app/app.py`

## Notes
- This app does not provide medical advice. Use for education and discussion with clinicians.
- If no corpus/index exists, the chat will prompt you to run ingestion+indexing.
