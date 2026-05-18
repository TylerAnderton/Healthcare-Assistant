---
name: Common Development Tasks
description: How-tos for typical dev work: adding vendors, tuning retrieval, extending prompts, adding tools
type: reference
---

## Add New Lab Vendor

1. Create `app/ingestion/vendors/new_vendor.py` with `parse(doc)` function:
   - Input: `Document` object (PyMuPDF fitz.Document)
   - Output: list of dicts matching LABS_PROCESSED_COLS (see constants.py)
   - Missing fields → None

2. Update vendor detection in `app/ingestion/ingest_labs.py:detect_vendor()`:
   - Add filename pattern or first-page text hint
   - Return vendor name string (matches module name)

3. Import in ingest_labs:
   ```python
   from app.ingestion.vendors import new_vendor as v_new
   ```
   
4. Add to vendor switch in ingest_labs main loop:
   ```python
   if vendor == "new_vendor":
       rows = v_new.parse(doc)
   ```

## Tune Retrieval Behavior

**Retrieved docs count/quality:**
- Adjust `MAX_CONTEXT_DOCS` (total cap) or `MAX_DOCS_PER_RETRIEVER` (per retriever) in `.env` or `constants.py`
- Change MMR diversity: modify `lambda_mult` in `chat.py:_retrieve()` (0=max diversity, 1=max relevance)

**Add/remove retriever source:**
- Edit `retrievers` list in `chat.py:_retrieve()`
- Each tuple: (retriever, query_augmentation)
- Example: add meds-specific retriever with enhanced query

**Filter by source_type:**
- Modify filter dict in `make_retriever(filter_)` calls
- Example: `{"source_type": "labs"}` to retrieve only labs

## Extend Prompting

**Modify system behavior:**
- Edit `SYSTEM_BASE` in `app/chains/prompts.py`

**Add few-shot examples:**
- Add dict to `few_shot_examples` list in `prompts.py`:
  ```python
  {
      "human": "Example question?",
      "assistant": "Example answer with citations."
  }
  ```

**Adjust context priority:**
- In `chat.py:_build_context()`, structured blocks are inserted before background docs
- Modify order or content of structured block insertion to change emphasis

## Add New Structured Tool

1. Create data loader in `app/tools/new_tool.py`:
   ```python
   def load_new_data():
       # Load from parquet, format as string
       return "Formatted text block for LLM context"
   ```

2. Call from `chat.py:_build_context()` when keywords match:
   ```python
   if "keyword" in question.lower():
       new_block = load_new_data()
       if new_block:
           structured_blocks.append(new_block)
   ```

3. For LLM tool-calling (ENABLE_TOOL_CALLING=1):
   - Define tool schema in `app/tools/tool_calls.py` (tool_definitions list)
   - Add handler function that returns JSON result
   - LLM invokes via tool name; result integrated into response

## Regenerate Test Fixtures

After updating `data/processed/tables/` parquet files:

```bash
python tests/generate_fixtures.py --whoop-days 7
```

Script samples from parquet files → `tests/fixtures/{labs,meds,whoop,prompts}.json`

Test suite validates retrieval accuracy against fixtures.

## Debug Chat Context

Enable verbose logging:

```bash
streamlit run app/app.py --logger.level=info
```

Logs show:
- Retrieved doc count + sources
- Structured blocks loaded
- Matched analytes
- Context length

Debug retrieval specifics in `chat.py:_retrieve()` and `_build_context()`.

## Run Single Test

```bash
pytest -vv tests/test_labs_accuracy.py::test_analyte_accuracy
```

Test fixtures validate:
- Labs: correct values + units on dates
- Meds: correct dose/frequency in timeline
- WHOOP: recovery/strain on dates
