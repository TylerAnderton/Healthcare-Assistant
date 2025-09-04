import re
from typing import List, Dict, Optional, Tuple
import fitz  # PyMuPDF

# Optional OCR fallback
try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
    HAVE_OCR = True
except Exception:
    HAVE_OCR = False

# Common date patterns
DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b"),  # MM/DD/YYYY
    re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b"),  # YYYY-MM-DD
    re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),\s*(20\d{2})\b", re.IGNORECASE),
]

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

UNIT_TOKENS = [
    "%",
    # Mass/volume with case variants commonly seen in PDFs
    "ng/mL", "pg/mL", "mg/dL", "g/dL", "ng/dL", "ng/ml", "mg/dl", "g/dl",
    # Immuno and enzyme units
    "µIU/mL", "uIU/mL", "µIU/ml", "uIU/ml", "µU/mL", "µU/ml", "mIU/L", "IU/L", "U/L", "IU/l",
    # Electrolytes and misc.
    "mEq/L", "mL/min/1.73m2",
    # Moles
    "mmol/L", "umol/L", "μmol/L", "nmol/L",
    # Cell sizes/counts
    "fL", "pg", "10^3/uL", "10^3/µL", "10^9/L", "10^6/uL", "10^6/µL", "10^12/L", "cells/uL", "k/uL", "m/cumm",
    # Alternate scientific notation variants seen in some reports
    "10E3/µL", "10E3/uL", "10e3/µL", "10e3/uL",
    # Generic per-volume fallbacks
    "/uL", "/µL", "/L", "x10^3/uL", "x10^3/µL",
    # Ratio/index style
    "Ratio", "ratio", "Index", "index",
]

units_lc = {u.lower() for u in UNIT_TOKENS}
arrow_re = re.compile(r"[\uf05d\uf045]")  # up/down arrow-like glyphs to ignore inside tokens
ign_exact = {"\uf513", "\uf511", "\uf05d", "\uf045"}  # flag & arrow emoji tokens to ignore entirely
header_stop = {t.lower() for t in [
    "biomarker", "quest", "current", "previous", "optimal", "range", "standard", "units",
    "blood", "test", "results", "comparative", "history", "score", "report", "out", "of"
]}

num_re = re.compile(r"-?\d+(?:\.\d+)?")
range_re = re.compile(r"(-?\d+(?:\.\d+)?)\s*[-–—]\s*(-?\d+(?:\.\d+)?)")

FLAG_TOKENS = {"H", "High", "L", "Low"}


def _to_iso(y: int, m: int, d: int) -> str:
    return f"{y:04d}-{m:02d}-{d:02d}"


def extract_report_date(doc: fitz.Document, filepath: str) -> Optional[str]:
    if doc.page_count == 0:
        return None
    text = doc[0].get_text("text") or ""

    # Try common labels
    preferred_labels = [
        "Collection Date", "Collected", "Date Collected",
        "Order Date", "Report Date", "Completed Date", "Received Date",
    ]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if any(lbl in ln for lbl in preferred_labels):
            for pat in DATE_PATTERNS:
                m = pat.search(ln)
                if not m:
                    continue
                if pat.pattern.startswith("\\b(\\d{1,2})/"):
                    mm, dd, yyyy = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    return _to_iso(yyyy, mm, dd)
                if pat.pattern.startswith("\\b(20"):
                    yyyy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    return _to_iso(yyyy, mm, dd)
                mon = MONTHS[m.group(1).lower()]
                dd, yyyy = int(m.group(2)), int(m.group(3))
                return _to_iso(yyyy, mon, dd)

    # Fallback: first date on first page
    for pat in DATE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        if pat.pattern.startswith("\\b(\\d{1,2})/"):
            mm, dd, yyyy = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return _to_iso(yyyy, mm, dd)
        if pat.pattern.startswith("\\b(20"):
            yyyy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return _to_iso(yyyy, mm, dd)
        mon = MONTHS[m.group(1).lower()]
        dd, yyyy = int(m.group(2)), int(m.group(3))
        return _to_iso(yyyy, mon, dd)
    return None


# --- Helpers for text and OCR ---

def _page_lines_text(page: fitz.Page) -> List[str]:
    t = page.get_text("text") or ""
    return [ln.strip() for ln in t.splitlines() if ln.strip()]


def _looks_like_accessibility_boxes(lines: List[str]) -> bool:
    if not lines:
        return False
    hits = sum(1 for ln in lines if ln.lower().startswith("text box"))
    return hits / max(1, len(lines)) > 0.3 or any("Ref Range:" in ln for ln in lines)


def _page_lines_ocr(page: fitz.Page) -> List[str]:
    if not HAVE_OCR:
        return []
    pm = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    mode = "RGB" if pm.n >= 3 else "L"
    img = Image.frombytes(mode, (pm.width, pm.height), pm.samples)
    text = pytesseract.image_to_string(img)
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


# --- Parsing helpers ---

def _parse_ref_range(text: str) -> Tuple[Optional[float], Optional[float]]:
    s = text.replace("–", "-").replace("—", "-")
    # Common formats: 0.45 - 4.50 | (0.45 - 4.50) | Ref Range: 0.45-4.50
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*[-/]\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r"Ref(?:erence)?\s*Range\s*[:]?\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", s, re.IGNORECASE)
    if m:
        return float(m.group(1)), float(m.group(2))
    # Inequalities
    m = re.search(r">\s*=?\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1)), None
    m = re.search(r"<\s*=?\s*(-?\d+(?:\.\d+)?)", s)
    if m:
        return None, float(m.group(1))
    return None, None


def _parse_value_unit(text: str) -> Tuple[Optional[float], Optional[str]]:
    # pick the first numeric token that isn't part of a range (heuristic)
    # We'll take the first number, then find the nearest unit token following it.
    mval = re.search(r"-?\d+(?:\.\d+)?", text)
    if not mval:
        return None, None
    val = float(mval.group(0))
    unit = None
    lower = text.lower()
    # find first unit token after the value index
    start_idx = mval.end()
    for u in UNIT_TOKENS:
        pos = lower.find(u.lower(), start_idx)
        if pos != -1:
            if unit is None or pos < lower.find(unit, start_idx):
                unit = u
    return val, unit


def _candidate_line(line: str) -> bool:
    if not any(u in line for u in UNIT_TOKENS):
        return False
    if not re.search(r"-?\d+(?:\.\d+)?", line):
        return False
    # avoid obvious headings
    if any(h in line for h in ["Summary", "Score", "Report", "Section", "Overview"]):
        return False
    return True


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


def _extract_analyte_via_boxmark(text: str) -> Optional[str]:
    """Return analyte trimmed to end before the last box-like glyph.
    If no glyph is found, return None to indicate this line likely isn't an analyte row.
    """
    last = None
    for m in BOXLIKE_RE.finditer(text):
        last = m
    if not last:
        return None
    # Text before the first box glyph in the last group
    analyte = text[: last.start()].rstrip()
    analyte = _clean_analyte(analyte)
    return analyte if analyte else None


def _clean_tok(t: str) -> str:
    # Remove arrow glyphs and collapse whitespace
    t = arrow_re.sub("", t).strip()
    return re.sub(r"\s+", " ", t)

# def _try_parse_range(tokens: List[str], idx: int) -> Optional[Tuple[float, float, int]]:
#     # Attempt 3-token, then 2-token, then 1-token range parsing.
#     parts = [_clean_tok(tokens[idx])]
#     if idx + 1 < len(tokens):
#         parts.append(parts[0] + " " + _clean_tok(tokens[idx + 1]))
#     if idx + 2 < len(tokens):
#         parts.append(parts[1] + " " + _clean_tok(tokens[idx + 2]))
#     # Check longest first
#     for consumed, s in ((3, parts[-1]) if len(parts) == 3 else (0, None), (2, parts[1] if len(parts) >= 2 else None), (1, parts[0])):
#         if not s:
#             continue
#         m = range_re.search(s)
#         if m:
#             low, high = float(m.group(1)), float(m.group(2))
#             return low, high, idx + consumed
#     return None

def _try_parse_range(token: str) -> Optional[Tuple[float, float]]:
    # Ranges are always a single token
    m = range_re.search(token)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None
    

def _parse_values(value_tokens: List[str]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    prev_v: Optional[float] = None
    curr_v: Optional[float] = None
    opt_l: Optional[float] = None
    opt_h: Optional[float] = None
    std_l: Optional[float] = None
    std_h: Optional[float] = None
    i = 0
    while i < len(value_tokens):
        # Skip empty or pure glyph tokens
        tk = value_tokens[i]
        if not tk or tk in ign_exact:
            i += 1
            continue

        # First, try to parse a range spanning up to 3 tokens -- no, ranges are always a single token
        # rng = _try_parse_range(value_tokens, i)
        # if rng:
        #     low, high, new_i = rng
        #     if opt_l is None and opt_h is None:
        #         opt_l, opt_h = low, high
        #     elif std_l is None and std_h is None:
        #         std_l, std_h = low, high
        #     i = new_i
        #     continue
        rng = _try_parse_range(tk)
        if rng:
            if opt_l is None and opt_h is None:
                opt_l, opt_h = rng
            elif std_l is None and std_h is None:
                std_l, std_h = rng
            i += 1
            continue

        # Else, try single numeric value
        m = num_re.search(tk)
        if m:
            val = float(m.group(0))
            if prev_v is None:
                prev_v = val
            elif curr_v is None:
                curr_v = val
            # Any further numerics are ignored for robustness
        i += 1

    # If only one numeric was found, it's the current value (no previous reported)
    if curr_v is None and prev_v is not None:
        curr_v, prev_v = prev_v, None

    return prev_v, curr_v, opt_l, opt_h, std_l, std_h


# --- Main entry point ---

def extract_rows(doc: fitz.Document, filepath: str) -> List[Dict]:
    date = extract_report_date(doc, filepath)
    out: List[Dict] = []

    # Locate the start page of the detailed results section
    start_page: Optional[int] = None
    for i, page in enumerate(doc):
        print(f"Searching page {i}")
        lines = _page_lines_text(page)
        if _looks_like_accessibility_boxes(lines) and HAVE_OCR:
            olines = _page_lines_ocr(page)
            if olines:
                lines = olines
        joined = "\n".join(lines)
        if re.search(r"Blood\s+Test\s+Results\s+Comparative", joined, re.IGNORECASE):
            # "Blood Test Results Comparative" is being found in another section in the 20230603 report on page 4
            print(f"Found Blood Test Results Comparative on page {i}")
            # print(f"Page lines: {lines}")
            start_page = i
            break

    if start_page is None:
        # No section found; return empty
        return out

    # in_section = True
    for pi in range(start_page, doc.page_count):
        page = doc[pi]
        lines = _page_lines_text(page)
        if _looks_like_accessibility_boxes(lines) and HAVE_OCR:
            olines = _page_lines_ocr(page)
            if olines:
                lines = olines

        # Stop when the next section header is encountered. Join lines because tokens may be split.
        joined = "\n".join(lines)
        # header_hit = bool(re.search(r"Blood\s+Test\s+Score\s+Report|Blood\s+Test\s+History", joined, re.IGNORECASE))
        blood_test_score_report_sent = r"This\s+report\s+shows\s+the\s+biomarkers\s+on\s+the\s+blood\s+test\s+that\s+are\s+farthest\s+from\s+the\s+median\s+expressed\s+as\s+a\s+\%"
        blood_test_history_sent = r"The\s+Blood\s+Test\s+History\s+Report\s+lists\s+the\s+results\s+of\s+your\s+patient's\s+Chemistry"
        header_hit = bool(re.search(blood_test_score_report_sent, joined, re.IGNORECASE)) or bool(re.search(blood_test_history_sent, joined, re.IGNORECASE))
        if header_hit:
            print(f"Found next section header on page {pi}")
            break

        print(f"Processing page {pi}")
        print(f"Page lines: {lines}")

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
                        print(f"Found analyte: {analyte}")
                    continue
                continue

            # Inside a row: accumulate until we hit a units token
            ctk = _clean_tok(tok)
            if not ctk:
                continue
            # Units token marks end-of-row
            if ctk.lower() in units_lc:
                unit = ctk.strip()
                prev_v, curr_v, opt_l, opt_h, std_l, std_h = _parse_values(row_tokens)
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
