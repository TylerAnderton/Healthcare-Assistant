import re
from typing import List, Dict, Optional, Tuple
import fitz  # PyMuPDF

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

def _to_iso(y: int, m: int, d: int) -> str:
    return f"{y:04d}-{m:02d}-{d:02d}"


def extract_report_date(doc: fitz.Document, filepath: str) -> Optional[str]:
    if doc.page_count == 0:
        return None
    text = doc[0].get_text("text") or ""

    # Prefer anchors near LabCorp
    for ln in (ln.strip() for ln in text.splitlines() if ln.strip()):
        # if any(tag in ln for tag in ["Date Collected", "Date Reported", "Date Received"]):
        if "Date Collected" in ln:
            for pat in DATE_PATTERNS:
                m = pat.search(ln)
                if m:
                    if pat.pattern.startswith("\\b(\\d{1,2})/"):
                        mm, dd, yyyy = int(m.group(1)), int(m.group(2)), int(m.group(3))
                        return _to_iso(yyyy, mm, dd)
                    if pat.pattern.startswith("\\b(20"):
                        yyyy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                        return _to_iso(yyyy, mm, dd)
                    # Month name
                    months = {
                        "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                        "jul":7,"aug":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12
                    }
                    mon = months[m.group(1).lower()]
                    dd, yyyy = int(m.group(2)), int(m.group(3))
                    return _to_iso(yyyy, mon, dd)

    # Fallback: first date on page
    for pat in DATE_PATTERNS:
        m = pat.search(text)
        if m:
            if pat.pattern.startswith("\\b(\\d{1,2})/"):
                mm, dd, yyyy = int(m.group(1)), int(m.group(2)), int(m.group(3))
                return _to_iso(yyyy, mm, dd)
            if pat.pattern.startswith("\\b(20"):
                yyyy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                return _to_iso(yyyy, mm, dd)
            months = {
                "jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,
                "jul":7,"aug":8,"sep":9,"sept":9,"oct":10,"nov":11,"dec":12
            }
            mon = months[m.group(1).lower()]
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


def _find_header_bounds(page: fitz.Page) -> Optional[Dict[str, float]]:
    # Identify header columns by proximity of header tokens
    text = page.get_text("text") or ""
    if not ("Test" in text and "Units" in text and "Reference" in text):
        return None
    # Approximate columns by scanning words and grouping x ranges for header tokens
    header_labels = {
        "analyte": ["Test"],
        "result": ["Current", "Result"],
        "units": ["Units"],
        "ref": ["Reference"],
    }
    words = _words_with_bbox(page)
    xs = {}
    for label, toks in header_labels.items():
        for w, rect in words:
            if any(tok in w for tok in toks):
                xs[label] = rect.x0
                break
    if not all(k in xs for k in header_labels):
        return None
    # Sort by x and assign ranges
    order = sorted(xs.items(), key=lambda kv: kv[1])
    bounds = {}
    for i, (label, x0) in enumerate(order):
        x1 = order[i+1][1] if i+1 < len(order) else page.rect.x1
        bounds[label] = (x0, x1)
    return bounds


def _group_lines(words: List[Tuple[str, fitz.Rect]], ytol: float = 3.0) -> List[List[Tuple[str, fitz.Rect]]]:
    # Group by similar y0
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
    # Sort each line by x
    for ln in lines:
        ln.sort(key=lambda wr: wr[1].x0)
    return lines


def _join_in_range(items: List[Tuple[str, fitz.Rect]], x0: float, x1: float) -> str:
    toks = [w for w, r in items if r.x0 >= x0 - 1 and r.x1 <= x1 + 1]
    return " ".join(toks).strip()


def _parse_ref_range(text: str) -> Tuple[Optional[float], Optional[float]]:
    text = text.replace("–", "-").replace("—", "-")
    # Range like 232-1245
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1)), float(m.group(2))
    # >= 3.0 or >3.0
    m = re.search(r">\s*=?\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1)), None
    m = re.search(r"<\s*=?\s*(-?\d+(?:\.\d+)?)", text)
    if m:
        return None, float(m.group(1))
    return None, None


def _parse_value_and_flag(text: str) -> Tuple[Optional[float], Optional[str]]:
    # capture first numeric
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
    out: List[Dict] = []
    for pi, page in enumerate(doc, start=1):
        bounds = _find_header_bounds(page)
        words = _words_with_bbox(page)
        lines = _group_lines(words)
        # Skip header line; parse subsequent lines
        for ln in lines:
            line_text = " ".join(w for w, _ in ln)
            if "Test" in line_text and "Units" in line_text and "Reference" in line_text:
                continue
            if not bounds:
                # fallback: require a number, a unit token, and a ref range in the same line
                if not any(u in line_text for u in UNIT_TOKENS):
                    continue
                rl, rh = _parse_ref_range(line_text)
                if rl is None and rh is None:
                    continue
                # analyte is portion before first value
                m = re.search(r"(.*?)(-?\d+(?:\.\d+)?)", line_text)
                if not m:
                    continue
                analyte = m.group(1).strip(" :")
                val, flag = _parse_value_and_flag(line_text)
                unit = next((u for u in UNIT_TOKENS if u in line_text), None)
                out.append({
                    "analyte": analyte,
                    "value": val,
                    "unit": unit,
                    "ref_low": rl,
                    "ref_high": rh,
                    "flag": flag,
                    "date": date,
                    "page": pi,
                    "vendor": "LabCorp",
                })
                continue
            # header-based parsing
            a = _join_in_range(ln, *bounds["analyte"]) if "analyte" in bounds else ""
            r = _join_in_range(ln, *bounds["result"]) if "result" in bounds else ""
            u = _join_in_range(ln, *bounds["units"]) if "units" in bounds else ""
            rf = _join_in_range(ln, *bounds["ref"]) if "ref" in bounds else ""
            if not a or (not r and not rf):
                continue
            val, flag = _parse_value_and_flag(r)
            rl, rh = _parse_ref_range(rf)
            # basic filter: skip if no numeric value found at all
            if val is None and rl is None and rh is None:
                continue
            out.append({
                "analyte": a,
                "value": val,
                "unit": u or None,
                "ref_low": rl,
                "ref_high": rh,
                "flag": flag,
                "date": date,
                "page": pi,
                "vendor": "LabCorp",
            })
    return out
