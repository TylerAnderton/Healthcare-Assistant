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
    "/uL", "/µL", "/L", "x10^3/uL", "x10^3/µL",
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

def _words_with_bbox(page: fitz.Page) -> List[Tuple[str, fitz.Rect]]:
    words: List[Tuple[str, fitz.Rect]] = []
    d = page.get_text("dict")
    for b in d.get("blocks", []):
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                txt = s.get("text", "").strip()
                if txt:
                    words.append((txt, fitz.Rect(s["bbox"])) )
    return words


def _group_lines(words: List[Tuple[str, fitz.Rect]], ytol: float = 3.0) -> List[List[Tuple[str, fitz.Rect]]]:
    lines: List[List[Tuple[str, fitz.Rect]]] = []
    for w in sorted(words, key=lambda wr: wr[1].y0):
        if not lines:
            lines.append([w])
            continue
        last_line = lines[-1]
        if abs(last_line[-1][1].y0 - w[1].y0) <= ytol:
            last_line.append(w)
        else:
            lines.append([w])
    for ln in lines:
        ln.sort(key=lambda wr: wr[1].x0)
    return lines


def _join_in_range(items: List[Tuple[str, fitz.Rect]], x0: float, x1: float) -> str:
    toks = [w for w, r in items if r.x0 >= x0 - 1 and r.x1 <= x1 + 1]
    return " ".join(toks).strip()


def _parse_value_and_flag(text: str) -> Tuple[Optional[float], Optional[str]]:
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    val = float(m.group(0)) if m else None
    flag = None
    for tok in FLAG_TOKENS:
        if re.search(rf"\b{re.escape(tok)}\b", text):
            flag = tok
            break
    return val, flag


# TODO: Maybe use BOXLIKE_CHARS to identify analyte lines
BOXLIKE_CHARS = "■□☐☑☒●○•▪▫◻◼▢❑▣▮▯" + "\uf0a0\uf0a1\uf0a2"  # include a few common private-use glyphs
BOXLIKE_RE = re.compile("[" + re.escape(BOXLIKE_CHARS) + "]+")


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


def _find_w2w_header_bounds(page: fitz.Page) -> Optional[Tuple[Dict[str, Tuple[float, float]], float]]:
    """Find column x-ranges for the Ways2Well Comparative table on a page.
    Returns (bounds, header_y0). bounds maps keys: analyte, current, previous, units, optimal, standard (any subset present).
    """
    words = _words_with_bbox(page)
    if not words:
        return None
    lines = _group_lines(words)
    header_idx = None
    header_words: List[Tuple[str, fitz.Rect]] = []
    # Heuristic: header line contains at least two of these tokens
    header_tokens = ["Units", "Current", "Previous", "Optimal", "Standard", "Reference", "range", "Range", "Test", "Biomarker", "Analyte"]
    for i, ln in enumerate(lines):
        txt = " ".join(w for w, _ in ln)
        hits = sum(1 for tok in header_tokens if tok in txt)
        if hits >= 2 and ("Units" in txt or "Current" in txt or "Optimal" in txt or "Reference" in txt or "Standard" in txt):
            header_idx = i
            header_words = ln
            break
    if header_idx is None:
        return None
    # Determine candidate x anchors per column by token presence in header line
    anchors: List[Tuple[str, float]] = []
    label_map = {
        "analyte": ["Biomarker", "Test", "Analyte", "Marker"],
        "current": ["Current", "Result"],
        "previous": ["Previous", "Prior", "Last"],
        "units": ["Units", "Unit"],
        "optimal": ["Optimal"],
        "standard": ["Standard", "Reference"],
    }
    for label, toks in label_map.items():
        for w, r in header_words:
            if any(tok in w for tok in toks):
                anchors.append((label, r.x0))
                break
    if not anchors:
        return None
    # Ensure analyte anchor exists; if not, take first word's x0 as analyte
    if not any(lbl == "analyte" for lbl, _ in anchors) and header_words:
        anchors.append(("analyte", header_words[0][1].x0))
    # Sort by x and create x ranges
    anchors.sort(key=lambda kv: kv[1])
    order = anchors
    bounds: Dict[str, Tuple[float, float]] = {}
    for i, (label, x0) in enumerate(order):
        x1 = order[i+1][1] if i+1 < len(order) else page.rect.x1
        bounds[label] = (x0, x1)
    header_y0 = max(r.y0 for _, r in header_words)
    return bounds, header_y0


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
    prev_bounds: Optional[Tuple[Dict[str, Tuple[float, float]], float]] = None
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

        # Try bbox-based header detection for structured columns
        hb = _find_w2w_header_bounds(page)
        if hb:
            prev_bounds = hb
        if prev_bounds:
            bounds, header_y0 = prev_bounds
            words = _words_with_bbox(page)
            if words:
                glines = _group_lines(words)
                for ln in glines:
                    # Skip header line and anything clearly above it
                    if not ln or ln[0][1].y0 <= header_y0 + 1.0:
                        continue
                    # Extract by column ranges if present
                    a_raw = _join_in_range(ln, *bounds.get("analyte", (page.rect.x0, page.rect.x1))) if bounds else ""
                    cur = _join_in_range(ln, *bounds["current"]) if "current" in bounds else ""
                    # prev = _join_in_range(ln, *bounds["previous"]) if "previous" in bounds else ""
                    u = _join_in_range(ln, *bounds["units"]) if "units" in bounds else ""
                    opt = _join_in_range(ln, *bounds["optimal"]) if "optimal" in bounds else ""
                    std = _join_in_range(ln, *bounds["standard"]) if "standard" in bounds else ""
                    # Use box-like glyphs to positively identify analyte label boundaries
                    a = _extract_analyte_via_boxmark(a_raw)
                    if not a:
                        continue
                    # Skip empty or non-row lines
                    has_num = bool(re.search(r"-?\d+(?:\.\d+)?", cur or opt or std))
                    if len(a) < 2 or not has_num:
                        continue

                    cur_val, cur_flag = _parse_value_and_flag(cur)
                    # prev_val, _ = _parse_value_and_flag(prev)
                    rl, rh = (None, None)
                    if opt:
                        rl, rh = _parse_ref_range(opt)
                    if (rl is None and rh is None) and std:
                        rl, rh = _parse_ref_range(std)

                    unit = u or None
                    # # Fallback: try to find any known unit token in cur/prev strings
                    # if not unit:
                    #     unit = next((tok for tok in UNIT_TOKENS if tok in (cur + " " + prev)), None)

                    # Only add if at least one value is present
                    if cur_val is None and rl is None and rh is None:
                        continue

                    out.append({
                        "analyte": a,
                        "value": cur_val,
                        "unit": unit,
                        "ref_low": rl,
                        "ref_high": rh,
                        "flag": cur_flag,
                        "date": date,
                        "page": pi + 1,
                        "vendor": "Ways2Well",
                        # "current_value": cur_val,
                        # "previous_value": prev_val,
                    })
                continue  # done with bbox-based parsing for this page

        # Fallback: line-based parsing (less accurate)
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

            # Identify analyte strictly via trailing box-like glyphs
            analyte = _extract_analyte_via_boxmark(ln) or ""
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
