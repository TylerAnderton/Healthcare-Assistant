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
    "%", "ng/mL", "pg/mL", "mg/dL", "g/dL", "µIU/mL", "uIU/mL", "mIU/L", "IU/L", "U/L",
    "mmol/L", "umol/L", "μmol/L", "nmol/L", "fL", "pg", "ng/dL", "mcg/dL", "ug/dL", "µg/dL",
    "10^3/uL", "10^3/µL", "10^9/L", "10^6/uL", "10^6/µL", "10^12/L", "cells/uL", "k/uL",
]

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
    lower = text
    # find first unit token after the value index
    start_idx = mval.end()
    for u in UNIT_TOKENS:
        pos = lower.find(u, start_idx)
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


def extract_rows(doc: fitz.Document, filepath: str) -> List[Dict]:
    date = extract_report_date(doc, filepath)
    out: List[Dict] = []

    # Locate the start page of the detailed results section
    start_page: Optional[int] = None
    for i, page in enumerate(doc):
        lines = _page_lines_text(page)
        if _looks_like_accessibility_boxes(lines) and HAVE_OCR:
            olines = _page_lines_ocr(page)
            if olines:
                lines = olines
        joined = "\n".join(lines)
        if re.search(r"Blood\s+Test\s+Results\s+Comparative", joined, re.IGNORECASE):
            start_page = i
            break

    if start_page is None:
        # No section found; return empty
        return out

    in_section = True
    for pi in range(start_page, doc.page_count):
        page = doc[pi]
        lines = _page_lines_text(page)
        if _looks_like_accessibility_boxes(lines) and HAVE_OCR:
            olines = _page_lines_ocr(page)
            if olines:
                lines = olines

        # Stop when the next section header is encountered
        header_hit = any(
            re.search(r"Blood\s+Test\s+Score\s+Report", ln, re.IGNORECASE) for ln in lines
        )
        if header_hit:
            break

        # Parse candidate lines
        prev_line: Optional[str] = None
        for ln in lines:
            if not _candidate_line(ln):
                prev_line = ln
                continue

            rl, rh = _parse_ref_range(ln)
            val, unit = _parse_value_unit(ln)
            if unit is None or val is None:
                prev_line = ln
                continue

            # Analyte is the text before the value
            m = re.search(r"(.*?)(-?\d+(?:\.\d+)?)", ln)
            analyte = (m.group(1).strip(" :") if m else "")
            if len(analyte) < 2 and prev_line:
                analyte = prev_line.strip(" :")
            if len(analyte) < 2:
                prev_line = ln
                continue

            out.append({
                "analyte": analyte,
                "value": val,
                "unit": unit,
                "ref_low": rl,
                "ref_high": rh,
                "flag": None,
                "date": date,
                "page": pi + 1,
                "vendor": "Ways2Well",
            })
            prev_line = ln

    return out
