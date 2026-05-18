---
name: Architecture Overview
description: High-level data flow, chat pipeline, module responsibilities
type: project
---

## Data Pipeline

**Raw → Ingested → Tables → Corpus → Vectorstore**

1. **Ingestion** (`app/ingestion/ingest_*.py`):
   - Labs PDFs: vendor detection heuristic (filename + first page text) → parse via vendor module → extract analytes/dates/values/units
   - Meds CSV/XLSX: column normalization → canonical schema (name, dose, dose_unit, frequency, etc.)
   - WHOOP CSVs: map raw column names to processed names (via WHOOP_*_COL_MAP in constants.py)

2. **Processing**:
   - All normalized to canonical schemas defined in `app/constants.py`
   - Output: parquet tables in `data/processed/tables/` (labs.parquet, meds.parquet, whoop_sleeps.parquet, whoop_workouts.parquet, whoop_physiological_cycles.parquet, whoop_journal_entries.parquet)

3. **Corpus Generation**:
   - Each table is summarized into digestible text blocks (e.g., "TSH: 2.1 mIU/L on 2025-05-01, ref 0.4-4.0, normal")
   - Output: `{labs,meds,whoop}_corpus.parquet` in `data/processed/corpus/`
   - Each corpus row: {text, source, source_type, date, ...} metadata

4. **Vectorstore**:
   - `app/indexing/build_index.py` loads corpus files
   - Chunks docs (RecursiveCharacterTextSplitter: 1000 char, 150 overlap, separators: \n\n, \n, . , space)
   - Embeds with HuggingFace model (default: BAAI/bge-large-en-v1.5)
   - Stores in Chroma at `VECTORSTORE_DIR`

## Chat Pipeline

Entry: `answer_question(question, history)` in `app/chains/chat.py`

1. **Retrieval** (`_retrieve(query)`):
   - 4 parallel retrievers (general + labs-specific + meds-specific + WHOOP-specific)
   - Each uses MMR (maximal marginal relevance): balances relevance vs. diversity
   - Config: k=MAX_DOCS_PER_RETRIEVER, fetch_k=k*3, lambda_mult=0.5
   - Dedup by source+page+content_hash; cap at MAX_CONTEXT_DOCS

2. **Context Building** (`_build_context(question, docs)`):
   - Keyword matching: question contains "lab"/"med"/"whoop"/etc. → load structured blocks
   - Structured blocks (from `app/tools/structured_context.py`):
     - Meds timeline: date-ordered medication changes
     - Labs panel: current values + ref ranges + flags + delta from previous
     - WHOOP snapshot: recent sleep/recovery/workout metrics
   - Analyte-specific summaries (if labs mentioned): call `labs_tool.summary(analyte_name)` and `labs_tool.history(analyte_name, limit=5)`
   - Background docs: vector retrieval results (marked as secondary to structured blocks)

3. **Prompting** (`_convert_history_to_messages`, `answer_question`):
   - Dynamic ChatPromptTemplate:
     - System: SYSTEM_BASE + context (structured + background)
     - Few-shot examples: 2-3 sample Q&A pairs from `prompts.py`
     - Conversation history: last 12 messages (Streamlit format → LangChain format)
     - User question: current query
   - Environment var `PREFER_STRUCTURED`: if True, suppress background docs for date-specific/structured queries

4. **LLM Invocation**:
   - If `ENABLE_TOOL_CALLING=1`: call `answer_with_tools(llm, prompt)` from `app/tools/tool_calls.py`
     - Lightweight tool-calling loop: LLM can invoke meds_tool, labs_tool, whoop_tool for specific facts
     - Tools return structured data; LLM integrates into response
   - Else: simple chain (prompt | llm | parser)
   - LLM: ChatOllama (configurable; default: qwen3:14b)

5. **Response**:
   - Return (answer_text, sources_list) where sources = [src + " (page X)" for each doc]
   - Streamlit UI displays answer + expandable sources

## Key Modules

| File | Role |
|---|---|
| `app/constants.py` | Canonical schemas (column names), retrieval tuning (MAX_CONTEXT_DOCS, MAX_DOCS_PER_RETRIEVER), tool flags |
| `app/chains/chat.py` | Retrieval, context building, prompting, entry point |
| `app/chains/prompts.py` | System prompt (SYSTEM_BASE), few-shot examples |
| `app/ingestion/ingest_*.py` | Data loading pipelines (detect vendor, parse, normalize) |
| `app/ingestion/vendors/*.py` | Vendor-specific PDF parsers (LabCorp, LetsGetChecked, Ways2Well) |
| `app/indexing/build_index.py` | Vectorstore construction (chunk → embed → store) |
| `app/tools/structured_context.py` | Load meds_timeline, labs_panel, whoop_recent from parquet |
| `app/tools/*_tool.py` | Structured data queries (meds_tool.list_meds(), labs_tool.summary(analyte), whoop_tool.recent()) |
| `app/tools/tool_calls.py` | Tool-calling orchestration (LLM → tool invocation → result integration) |
| `app/app.py` | Streamlit UI (chat history, message formatting, source display) |

## Configuration

Environment vars in `.env` (copy `.env.example`):

| Variable | Default | Impact |
|---|---|---|
| OLLAMA_BASE_URL | http://localhost:11434 | Ollama server location |
| OLLAMA_MODEL | qwen3:14b | Chat LLM (pull via `ollama pull <model>`) |
| EMBEDDING_MODEL | BAAI/bge-large-en-v1.5 | HuggingFace embedding model for vectorstore |
| RAW_DIR | ./data/raw | Input directory (labs/, medications/, whoop/) |
| PROCESSED_DIR | ./data/processed | Output directory (tables/, corpus/) |
| VECTORSTORE_DIR | ./data/processed/vectorstore | Chroma index location |
| MAX_CONTEXT_DOCS | 20 | Cap retrieved docs in context |
| MAX_DOCS_PER_RETRIEVER | 20 | Docs per individual retriever (before dedup) |
| ENABLE_TOOL_CALLING | 1 | Use LLM tool-calling loop for structured queries |
| PREFER_STRUCTURED | 0 | Suppress background docs if query mentions dates/structured keywords |
