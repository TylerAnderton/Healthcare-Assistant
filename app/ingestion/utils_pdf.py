import fitz  # PyMuPDF
from typing import List, Dict


def extract_pdf_pages(pdf_path: str) -> List[Dict]:
    """Extract text by page with basic metadata."""
    out = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            out.append({"page": i, "text": text})
    return out
