# Vendor-specific lab parsers
# Each module should expose:
# - extract_report_date(doc, filepath) -> Optional[str]  # ISO date string
# - extract_rows(doc, filepath) -> List[Dict]            # rows with analyte/value/unit/ref_low/ref_high/flag/date/page
