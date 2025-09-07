import re
from typing import List, Dict, Optional, Tuple
import fitz  # PyMuPDF

HAVE_OCR = False # TODO: remove all OCR references

DATE_PATTERNS = [
    re.compile(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b"),  # MM/DD/YYYY
    re.compile(r"\b(20\d{2})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b"),  # YYYY-MM-DD
    re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),\s*(20\d{2})\b", re.IGNORECASE),
]

UNIT_TOKENS = [
    "%", "ng/mL", "pg/mL", "mg/dL", "g/dL", "µIU/mL", "uIU/mL", "mIU/L", "IU/L", "U/L",
    "mmol/L", "umol/L", "μmol/L", "nmol/L", "fL", "pg", "ng/dL"
]

FLAG_TOKENS = {"H", "High", "L", "Low"}

MONTHS = {
    "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
    "jul":7,"aug":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12
}

def _to_iso(y: int, m: int, d: int) -> str:
    return f"{y:04d}-{m:02d}-{d:02d}"


def extract_report_date(doc: fitz.Document, filepath: str) -> Optional[str]:
    if doc.page_count == 0:
        return None
    text = doc[0].get_text("text") or ""

    # Prefer 'Collection Date' if present
    for ln in (ln.strip() for ln in text.splitlines() if ln.strip()):
        # if any(tag in ln for tag in ["Collection Date", "Completed Date", "Received Date"]):
        if "Collection Date" in ln:
            for pat in DATE_PATTERNS:
                m = pat.search(ln)
                if m:
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
        if m:
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


def _words_with_bbox(page: fitz.Page) -> List[Tuple[str, fitz.Rect]]:
    words = []
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


def _parse_ref_range(text: str) -> Tuple[Optional[float], Optional[float]]:
    text = text.replace("–", "-").replace("—", "-")
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r">\s*=?\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1)), None
    m = re.search(r"<\s*=?\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return None, float(m.group(1))
    return None, None


def _parse_value_and_flag(text: str) -> Tuple[Optional[float], Optional[str]]:
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    val = float(m.group(0)) if m else None
    flag = None
    for tok in FLAG_TOKENS:
        if re.search(rf"\b{re.escape(tok)}\b", text):
            flag = tok
            break
    return val, flag


def extract_rows(doc: fitz.Document, filepath: str) -> List[Dict]:
    date = extract_report_date(doc, filepath)
    print(f"Extracting rows from {filepath}")
    print(f"Date: {date}")
    out: List[Dict] = []

    def _page_lines_text(p: fitz.Page) -> List[str]:
        t = p.get_text("text") or ""
        return [ln.strip() for ln in t.splitlines() if ln.strip()]

    def _looks_like_accessibility_boxes(lines: List[str]) -> bool:
        if not lines:
            return False
        hits = sum(1 for ln in lines if ln.lower().startswith("text box"))
        return hits / max(1, len(lines)) > 0.3 or any("Ref Range:" in ln for ln in lines)

    def _page_lines_ocr(p: fitz.Page) -> List[str]:
        if not HAVE_OCR:
            return []
        # Render at 2x for better OCR
        pm = p.get_pixmap(matrix=fitz.Matrix(2, 2))
        mode = "RGB" if pm.n >= 3 else "L"
        img = Image.frombytes(mode, (pm.width, pm.height), pm.samples)
        text = pytesseract.image_to_string(img)
        return [ln.strip() for ln in text.splitlines() if ln.strip()]

    prev_line: Optional[str] = None
    for pi, page in enumerate(doc, start=1):
        # First try text; if it looks like accessibility placeholders, OCR fallback
        lines = _page_lines_text(page)
        if _looks_like_accessibility_boxes(lines) and HAVE_OCR:
            ocr_lines = _page_lines_ocr(page)
            if ocr_lines:
                lines = ocr_lines

        # Primary: parse by lines directly
        for ln in lines:
            line_text = ln
            if not any(u in line_text for u in UNIT_TOKENS):
                prev_line = line_text
                continue
            rl, rh = _parse_ref_range(line_text)
            val, flag = _parse_value_and_flag(line_text)
            unit = next((u for u in UNIT_TOKENS if u in line_text), None)
            if unit is None or val is None:
                prev_line = line_text
                continue
            # analyte before first number; if too short, look to prev line
            m = re.search(r"(.*?)(-?\d+(?:\.\d+)?)", line_text)
            analyte = (m.group(1).strip(" :") if m else "")
            if len(analyte) < 2 and prev_line:
                analyte = prev_line.strip(" :")
            if len(analyte) < 2:
                prev_line = line_text
                continue
            out.append({
                "analyte": analyte,
                "value": val,
                "unit": unit,
                "ref_low": rl,
                "ref_high": rh,
                "flag": flag,
                "date": date,
                "page": pi,
                "vendor": "LetsGetChecked",
            })
            prev_line = line_text

        # Secondary: if nothing found on page, try bbox heuristic as a fallback
        if not any(r.get("page") == pi for r in out):
            words = _words_with_bbox(page)
            lines_b = _group_lines(words)
            for ln in lines_b:
                line_text = " ".join(w for w, _ in ln)
                if not any(u in line_text for u in UNIT_TOKENS):
                    continue
                rl, rh = _parse_ref_range(line_text)
                val, flag = _parse_value_and_flag(line_text)
                unit = next((u for u in UNIT_TOKENS if u in line_text), None)
                if unit is None or val is None:
                    continue
                m = re.search(r"(.*?)(-?\d+(?:\.\d+)?)", line_text)
                analyte = (m.group(1).strip(" :") if m else "")
                if len(analyte) < 2:
                    continue
                out.append({
                    "analyte": analyte,
                    "value": val,
                    "unit": unit,
                    "ref_low": rl,
                    "ref_high": rh,
                    "flag": flag,
                    "date": date,
                    "page": pi,
                    "vendor": "LetsGetChecked",
                })

    return out
