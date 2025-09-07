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

## OCR Environment (Linux/macOS)

OCR behavior can differ across machines due to native Tesseract, tessdata, and Python package versions. To avoid breaking parsers (e.g., `app/ingestion/vendors/labcorp.py`) when switching OSes, capture and pin these details.

### Capture your OCR environment

Option A: one-shot diagnostic script

```bash
python scripts/print_ocr_env.py
```

This prints:

- Platform and Python version
- Tesseract path, version (with Leptonica), installed languages
- TESSDATA_PREFIX (if set)
- PyMuPDF, Pillow, pytesseract versions (and resolved `tesseract_cmd`)

Option B: minimal shell commands

```bash
which tesseract
tesseract --version
tesseract --list-langs
echo "$TESSDATA_PREFIX"
python -c "import fitz; print('PyMuPDF:', getattr(fitz, '__version__', getattr(fitz, 'VersionBind', None)))"
python -c "from PIL import __version__ as v; print('Pillow:', v)"
python -c "import pytesseract; print('pytesseract:', getattr(pytesseract, '__version__', 'unknown')); print('tesseract_cmd:', pytesseract.pytesseract.tesseract_cmd)"
```

### Recommended versions to pin in README

Include a small table or list similar to:

- PyMuPDF: 1.24.x
- Pillow: 10.x
- pytesseract: 0.3.x
- Tesseract (native): 5.4.0 (Leptonica 1.84.1)
- Tessdata languages: eng
- TESSDATA_PREFIX: unset (system default) or explicit path
- which tesseract: `/usr/bin/tesseract` (Linux) or `/opt/homebrew/bin/tesseract` (macOS ARM)

### Install tips

- macOS (Homebrew): `brew install tesseract`
  - tessdata at `/opt/homebrew/Cellar/tesseract/<version>/share/tessdata`
- Ubuntu/Debian: `sudo apt-get install tesseract-ocr tesseract-ocr-eng`

The ingestion script (`app/ingestion/ingest_labs.py`) logs the OCR environment at startup to aid debugging.

## Quickstart
1. Install Ollama and a model:
   - https://ollama.com/download
   - Recommended chat model: `ollama pull llama3.1:8b`
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
