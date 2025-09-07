import re
from typing import List, Dict, Optional, Tuple
import fitz  # PyMuPDF

from .utils import *

# --- LabCorp-specific helpers ---

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


# --- Main entry point ---

def extract_rows(doc: fitz.Document, filepath: str) -> List[Dict]:
    date = extract_report_date(doc, filepath)
    out: List[Dict] = []

    for pi, page in enumerate(doc):
        lines = page_lines_text(page)

        # Stop when the next section header is encountered. Join lines because tokens may be split.
        joined = "\n".join(lines)
        historical_results_header = r"Historical\s+Results\s+&\s+Insights"
        header_hit = bool(re.search(historical_results_header, joined, re.IGNORECASE))
        if header_hit:
            # print(f"Found next section header on page {pi}")
            break

        # print(f"Processing page {pi+1}") # page numbers are 0-indexed
        # print(f"Page lines: {lines}")

        # Row scanner according to LabCorp token structure
        def is_numeric(tok: str) -> bool:
            return bool(re.fullmatch(r"-?\d+(?:\.\d+)?", tok))

        def is_date(tok: str) -> bool:
            for pat in DATE_PATTERNS:
                if pat.search(tok):
                    return True
            return False

        def strip_superscript(name: str) -> str:
            # Remove trailing footnote-like numbers regardless of separators.
            # Examples to strip: "Ferritin 01", "Hemoglobin A1c\xa001", "Estradiol, Sensitive A, 02"
            s = name
            pat = re.compile(r"(?:[\s,;:\-]*|\u00a0)\d{1,3}\s*$")
            prev = None
            while prev != s:
                prev = s
                s = pat.sub("", s)
            # Trim any lingering trailing punctuation/spaces
            s = s.rstrip(" ,;:.-\t")
            return s

        def is_unit(tok: str) -> bool:
            return tok.lower() in units_lc

        def parse_ref_token(tok: str) -> Tuple[Optional[float], Optional[float], bool]:
            t = tok.strip()
            # Accept "Not Estab." or similar
            if re.search(r"(?i)not\s+estab", t):
                return None, None, True
            # Use existing parser for ranges and inequalities
            rl, rh = _parse_ref_range(t)
            ok = (rl is not None or rh is not None)
            return rl, rh, ok

        i = 0
        SKIP_TOKENS = {
            "Test", "Current Result and Flag", "Previous Result and Date",
            "Units", "Reference Interval", "Please Note:", "Comments",
            "Disclaimer", "Icon Legend", "Performing Labs", "Patient Details",
            "Physician Details", "Specimen Details"
        }

        while i < len(lines):
            tok = clean_tok(lines[i])
            if not tok or tok in SKIP_TOKENS:
                i += 1
                continue

            # Candidate analyte must contain a letter and must NOT be a number
            if not re.search(r"[A-Za-z]", tok) or is_numeric(tok):
                i += 1
                continue

            analyte_raw = tok
            # Next must be numeric current value
            if i + 1 >= len(lines):
                break
            curr_tok = clean_tok(lines[i + 1])
            if not curr_tok or not is_numeric(curr_tok):
                i += 1
                continue

            curr_val = float(curr_tok)
            j = i + 2
            flag: Optional[str] = None
            prev_val: Optional[float] = None
            prev_date: Optional[str] = None

            # Optional flag
            if j < len(lines):
                t = clean_tok(lines[j])
                if t in FLAG_TOKENS:
                    flag = t
                    j += 1

            # Optional previous value
            if j < len(lines):
                t = clean_tok(lines[j])
                if t and is_numeric(t):
                    prev_val = float(t)
                    j += 1

            # Optional previous date
            if j < len(lines):
                t = clean_tok(lines[j])
                if t and is_date(t):
                    prev_date = t
                    j += 1

            # Units (required in spec)
            if j >= len(lines):
                break
            unit_tok = clean_tok(lines[j])
            if not unit_tok or not is_unit(unit_tok):
                # Not a unit; invalid row
                i += 1
                continue
            unit = unit_tok
            j += 1

            # Reference range (required)
            if j >= len(lines):
                break
            ref_tok = clean_tok(lines[j])
            rl, rh, ok = parse_ref_token(ref_tok)
            if not ok:
                i += 1
                continue

            analyte = strip_superscript(analyte_raw)
            out.append({
                "analyte": analyte,
                "value": curr_val,
                # "current_value": curr_val,
                # "previous_value": prev_val,
                # "previous_date": prev_date,
                "unit": unit,
                "ref_low": rl,
                "ref_high": rh,
                "flag": flag,
                "date": date,
                "page": pi + 1,
                "vendor": "LabCorp",
            })

            # Advance past this row
            i = j + 1
            continue

    return out
