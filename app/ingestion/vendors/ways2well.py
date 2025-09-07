import re
from typing import List, Dict, Optional, Tuple
import fitz  # PyMuPDF

from .utils import *

# --- Ways2Well-specific helpers ---

# Actual box character:  
BOXLIKE_CHARS = "■□☐☑☒●○•▪▫◻◼▢❑▣▮▯" + "\uf0a0\uf0a1\uf0a2"  # include a few common private-use glyphs
BOXLIKE_RE = re.compile("[" + re.escape(BOXLIKE_CHARS) + "]+")


def _filter_analyte(text: str) -> Optional[str]:
    if re.search(BOXLIKE_RE, text):
        text = re.sub(BOXLIKE_RE, "", text)
        return text.strip()
    return None


def _clean_analyte(text: str) -> str:
    s = text
    for ch in BOXLIKE_CHARS:
        s = s.replace(ch, " ")
    # Remove lingering 'Text box' artifacts if any slipped in
    s = re.sub(r"(?i)text\s+box.*", " ", s)
    # Collapse spaces and strip punctuation at ends
    s = re.sub(r"\s+", " ", s).strip(" :.-\t")
    return s


# --- Main entry point ---

def extract_rows(doc: fitz.Document, filepath: str) -> List[Dict]:
    date = extract_report_date(doc, filepath)
    out: List[Dict] = []

    # Locate the start page of the detailed results section
    start_page: Optional[int] = None
    for i, page in enumerate(doc):
        # print(f"Searching page {i}")
        lines = page_lines_text(page)
        # if looks_like_accessibility_boxes(lines) and HAVE_OCR:
        #     olines = page_lines_ocr(page)
        #     if olines:
        #         lines = olines
        joined = "\n".join(lines)
        if re.search(r"Blood\s+Test\s+Results\s+Comparative", joined, re.IGNORECASE):
            # "Blood Test Results Comparative" is being found in another section in the 20230603 report on page 4
            # print(f"Found Blood Test Results Comparative on page {i}")
            # print(f"Page lines: {lines}")
            start_page = i
            break

    if start_page is None:
        # No section found; return empty
        return out

    # in_section = True
    for pi in range(start_page, doc.page_count):
        page = doc[pi]
        lines = page_lines_text(page)
        # if looks_like_accessibility_boxes(lines) and HAVE_OCR:
        #     olines = page_lines_ocr(page)
        #     if olines:
        #         lines = olines

        # Stop when the next section header is encountered. Join lines because tokens may be split.
        joined = "\n".join(lines)
        blood_test_score_report_sent = r"This\s+report\s+shows\s+the\s+biomarkers\s+on\s+the\s+blood\s+test\s+that\s+are\s+farthest\s+from\s+the\s+median\s+expressed\s+as\s+a\s+\%"
        blood_test_history_sent = r"The\s+Blood\s+Test\s+History\s+Report\s+lists\s+the\s+results\s+of\s+your\s+patient's\s+Chemistry"
        header_hit = bool(re.search(blood_test_score_report_sent, joined, re.IGNORECASE)) or bool(re.search(blood_test_history_sent, joined, re.IGNORECASE))
        if header_hit:
            # print(f"Found next section header on page {pi}")
            break

        # print(f"Processing page {pi}")
        # print(f"Page lines: {lines}")

        # Token-by-token parsing across the page; each "line" item is effectively a word/token.
        # Strategy:
        # - Detect start of a row by an analyte token ending with the box-like glyph (e.g., '\xa0\uf03d').
        # - Accumulate subsequent tokens until a Units token is found; ignore flag/arrow glyphs.
        # - Within the accumulated tokens, extract [previous?], current value, optimal range, [standard range].

        parsing_line = False
        analyte: Optional[str] = None
        row_tokens: List[str] = []
        # pre_analyte_buf: List[str] = []  # keep last few tokens to assemble multi-token analyte names

        for tok in lines:
            if tok in ign_exact:
                continue

            if not parsing_line:
                a_tail = _filter_analyte(tok)
                if a_tail:
                    # Build analyte from preceding context tokens + this tail
                    a_full = _clean_analyte(a_tail)
                    if a_full and len(a_full) >= 2:
                        parsing_line = True
                        analyte = a_full
                        row_tokens = [analyte]
                        # print(f"Found analyte: {analyte}")
                    continue
                continue

            # Inside a row: accumulate until we hit a units token
            ctk = clean_tok(tok)
            if not ctk:
                continue
            # Units token marks end-of-row
            if ctk.lower() in units_lc:
                unit = ctk.strip()
                prev_v, curr_v, opt_l, opt_h, std_l, std_h = parse_values(row_tokens)
                # Prefer optimal range; fall back to standard if optimal missing
                rl, rh = (opt_l, opt_h) if (opt_l is not None or opt_h is not None) else (std_l, std_h)
                if curr_v is not None:
                    out.append({
                        "analyte": analyte,
                        "value": curr_v,
                        "unit": unit,
                        "ref_low": rl,
                        "ref_high": rh,
                        "flag": None,
                        "date": date,
                        "page": pi + 1,
                        "vendor": "Ways2Well",
                    })

                # Reset for potential next row on the same page
                parsing_line = False
                analyte = None
                row_tokens = []
                continue

            # Otherwise accumulate token for later parsing
            row_tokens.append(ctk)


    return out
