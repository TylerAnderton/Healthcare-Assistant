import re
from typing import List, Dict, Optional, Tuple
import fitz  # PyMuPDF

HAVE_OCR = False # TODO: remove all OCR references

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
    # Mass/volume
    "ng/ml", "pg/ml", "mg/dl", "g/dl", "ng/dl",
    "µg/dl", "ug/dl", "μg/dl",
    # Immuno and enzyme units
    "µiu/ml", "uiu/ml", "µu/ml", "μiu/ml", "μu/ml", "miu/l", "iu/l", "u/l",
    # Electrolytes and misc.
    "meq/l", "ml/min/1.73m2", "ml/min/1.73",
    # Moles
    "mmol/l", "umol/l", "μmol/l", "nmol/l",
    # Cell sizes/counts
    "fl", "pg", "10^3/ul", "10^3/µl", "10^9/l", "10^6/ul", "10^6/µl", "10^12/l", "cells/ul", "k/ul", "m/cumm", "k/cumm",
    # Alternate scientific notation variants
    "10e3/µl", "10e3/ul", "x10e3/ul", "x10e3/µl",
    # Generic per-volume fallbacks and split tokens
    "/ul", "/µl", "/l", "x10^3/ul", "x10^3/µl", "mg/d", "mg/l",
    # Ratio/index style
    "ratio", "index",
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


def to_iso(y: int, m: int, d: int) -> str:
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
                    return to_iso(yyyy, mm, dd)
                if pat.pattern.startswith("\\b(20"):
                    yyyy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    return to_iso(yyyy, mm, dd)
                mon = MONTHS[m.group(1).lower()]
                dd, yyyy = int(m.group(2)), int(m.group(3))
                return to_iso(yyyy, mon, dd)

    # Fallback: first date on first page
    for pat in DATE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        if pat.pattern.startswith("\\b(\\d{1,2})/"):
            mm, dd, yyyy = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return to_iso(yyyy, mm, dd)
        if pat.pattern.startswith("\\b(20"):
            yyyy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return to_iso(yyyy, mm, dd)
        mon = MONTHS[m.group(1).lower()]
        dd, yyyy = int(m.group(2)), int(m.group(3))
        return to_iso(yyyy, mon, dd)
    return None


# --- Helpers for text and OCR ---

def page_lines_text(page: fitz.Page) -> List[str]:
    t = page.get_text("text") or ""
    return [ln.strip() for ln in t.splitlines() if ln.strip()]


def looks_like_accessibility_boxes(lines: List[str]) -> bool:
    if not lines:
        return False
    hits = sum(1 for ln in lines if ln.lower().startswith("text box"))
    return hits / max(1, len(lines)) > 0.3 or any("Ref Range:" in ln for ln in lines)


def page_lines_ocr(page: fitz.Page) -> List[str]:
    if not HAVE_OCR:
        return []
    pm = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    mode = "RGB" if pm.n >= 3 else "L"
    img = Image.frombytes(mode, (pm.width, pm.height), pm.samples)
    text = pytesseract.image_to_string(img)
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


# --- Parsing helpers ---

# def parse_ref_range(text: str) -> Tuple[Optional[float], Optional[float]]:
#     s = text.replace("–", "-").replace("—", "-")
#     # Common formats: 0.45 - 4.50 | (0.45 - 4.50) | Ref Range: 0.45-4.50
#     m = re.search(r"(-?\d+(?:\.\d+)?)\s*[-/]\s*(-?\d+(?:\.\d+)?)", s)
#     if m:
#         return float(m.group(1)), float(m.group(2))
#     m = re.search(r"Ref(?:erence)?\s*Range\s*[:]?\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", s, re.IGNORECASE)
#     if m:
#         return float(m.group(1)), float(m.group(2))
#     # Inequalities
#     m = re.search(r">\s*=?\s*(-?\d+(?:\.\d+)?)", s)
#     if m:
#         return float(m.group(1)), None
#     m = re.search(r"<\s*=?\s*(-?\d+(?:\.\d+)?)", s)
#     if m:
#         return None, float(m.group(1))
#     return None, None


# def parse_value_unit(text: str) -> Tuple[Optional[float], Optional[str]]:
#     # pick the first numeric token that isn't part of a range (heuristic)
#     # We'll take the first number, then find the nearest unit token following it.
#     mval = re.search(r"-?\d+(?:\.\d+)?", text)
#     if not mval:
#         return None, None
#     val = float(mval.group(0))
#     unit = None
#     lower = text.lower()
#     # find first unit token after the value index
#     start_idx = mval.end()
#     for u in UNIT_TOKENS:
#         pos = lower.find(u.lower(), start_idx)
#         if pos != -1:
#             if unit is None or pos < lower.find(unit, start_idx):
#                 unit = u
#     return val, unit


# def candidate_line(line: str) -> bool:
#     if not any(u in line for u in UNIT_TOKENS):
#         return False
#     if not re.search(r"-?\d+(?:\.\d+)?", line):
#         return False
#     # avoid obvious headings
#     if any(h in line for h in ["Summary", "Score", "Report", "Section", "Overview"]):
#         return False
#     return True


# def _extract_analyte_via_boxmark(text: str) -> Optional[str]:
#     """Return analyte trimmed to end before the last box-like glyph.
#     If no glyph is found, return None to indicate this line likely isn't an analyte row.
#     """
#     last = None
#     for m in BOXLIKE_RE.finditer(text):
#         last = m
#     if not last:
#         return None
#     # Text before the first box glyph in the last group
#     analyte = text[: last.start()].rstrip()
#     analyte = _clean_analyte(analyte)
#     return analyte if analyte else None


def clean_tok(t: str) -> str:
    # Remove arrow glyphs and collapse whitespace
    t = arrow_re.sub("", t).strip()
    return re.sub(r"\s+", " ", t)

def try_parse_range(token: str) -> Optional[Tuple[float, float]]:
    # Ranges are always a single token
    m = range_re.search(token)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None
    

def parse_values(value_tokens: List[str]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
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

        rng = try_parse_range(tk)
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