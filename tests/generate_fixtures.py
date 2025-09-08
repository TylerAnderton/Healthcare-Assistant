"""
CLI script to regenerate test fixtures from gold parquet tables under data/processed/tables/.

Usage:
  source .venv/bin/activate
  python tests/generate_fixtures.py [--limit WHOOP_DAYS]
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = Path(os.getenv("PROCESSED_DIR", ROOT / "data/processed")) / "tables"
FIXTURES = Path(__file__).parent / "fixtures"


def _safe_str(x):
    try:
        import math
        if x is None:
            return ""
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return ""
        return str(x)
    except Exception:
        return ""


def regenerate_labs():
    fp = PROCESSED / "labs.parquet"
    if not fp.exists():
        print(f"labs parquet not found: {fp}")
        return
    df = pd.read_parquet(fp)
    # Choose 3 analytes with >= 2 entries
    cases = []
    for name, g in df.groupby("analyte"):
        if len(g) >= 2:
            g = g.sort_values("date")
            cols = [c for c in ["date","analyte","value","unit","ref_low","ref_high","flag","vendor","source"] if c in g.columns]
            recs = g.tail(3)[cols].to_dict(orient="records")
            cases.append({"analyte": str(name), "values": recs})
        if len(cases) >= 3:
            break
    dst = FIXTURES / "labs.json"
    dst.write_text(json.dumps({"cases": cases}, indent=2, default=str))
    print("Wrote", dst)


def regenerate_meds():
    fp = PROCESSED / "meds.parquet"
    if not fp.exists():
        print(f"meds parquet not found: {fp}")
        return
    md = pd.read_parquet(fp)
    # Heuristic to build event list mirroring load_meds_timeline
    name_col = None
    for c in md.columns:
        if str(c).lower() in ["name","medication","drug"]:
            name_col = c; break
    dose_cols = [c for c in md.columns if str(c).lower() in ["dose","dosage"]]
    dose_unit_cols = [c for c in md.columns if str(c).lower() in ["dose_unit"]]
    freq_cols = [c for c in md.columns if str(c).lower() in ["frequency","freq","dose_frequency"]]
    freq_unit_cols = [c for c in md.columns if str(c).lower() in ["frequency_unit","dose_frequency_unit"]]
    start_cols = [c for c in md.columns if str(c).lower() in ["start_date","date_start","start"]]
    updated_cols = [c for c in md.columns if str(c).lower() in ["dose_updated","date_updated","updated","date_changed","change_date"]]
    end_cols = [c for c in md.columns if str(c).lower() in ["end_date","date_stop","end"]]

    from collections import defaultdict
    grouped = defaultdict(list)

    for _, r in md.iterrows():
        nm = _safe_str(r.get(name_col)).strip()
        if not nm:
            continue
        dose = _safe_str(r.get(dose_cols[0])) if dose_cols else ""
        dunit = _safe_str(r.get(dose_unit_cols[0])) if dose_unit_cols else ""
        freq = _safe_str(r.get(freq_cols[0])) if freq_cols else ""
        funit = _safe_str(r.get(freq_unit_cols[0])) if freq_unit_cols else ""
        def add(dt, kind):
            dt = _safe_str(dt)
            if not dt or dt == 'NaT':
                return
            grouped[nm].append({"date": dt[:10], "kind": kind, "dose": dose, "dose_unit": dunit, "freq": freq, "freq_unit": funit})
        for c in start_cols: add(r.get(c), "start")
        for c in updated_cols: add(r.get(c), "dose_change")
        for c in end_cols: add(r.get(c), "stop")

    cases = []
    for med, evs in grouped.items():
        if len(evs) >= 2:
            evs = sorted(evs, key=lambda x: x["date"])[:5]
            cases.append({"med": med, "events": evs})
        if len(cases) >= 3:
            break

    dst = FIXTURES / "meds.json"
    dst.write_text(json.dumps({"cases": cases}, indent=2, default=str))
    print("Wrote", dst)


def regenerate_whoop(days: int = 7):
    fp = PROCESSED / "whoop_physiological_cycles.parquet"
    if not fp.exists():
        print(f"whoop cycles parquet not found: {fp}")
        return
    wc = pd.read_parquet(fp)
    # Build date
    if "date" not in wc.columns:
        date_candidates = [c for c in wc.columns if str(c).lower() in ["date","cycle_date","start","day","cycle start time"]]
        def norm_date(row):
            for c in date_candidates:
                v = row.get(c)
                if pd.notna(v):
                    s = str(v)
                    if 'T' in s: s = s.split('T',1)[0]
                    return s[:10]
            return ''
        wc = wc.copy(); wc['date'] = wc.apply(norm_date, axis=1)
    dff = wc[wc['date'].astype(str)!=''].sort_values('date').tail(days)

    cols_map = {
        'strain': ['strain','Day Strain'],
        'recovery_score': ['recovery_score','recovery','score','Recovery score %'],
        'rhr_bpm': ['resting_heart_rate','rhr','Resting heart rate (bpm)'],
        'hrv_ms': ['hrv','hrv_rmssd_milli','rmssd','Heart rate variability (ms)'],
    }
    def pick(row, cands):
        for c in cands:
            if c in wc.columns and pd.notna(row.get(c)):
                return row.get(c)
        return None

    rows = []
    for _, r in dff.iterrows():
        rows.append({
            'date': str(r.get('date')),
            'strain': pick(r, cols_map['strain']),
            'recovery_score': pick(r, cols_map['recovery_score']),
            'rhr_bpm': pick(r, cols_map['rhr_bpm']),
            'hrv_ms': pick(r, cols_map['hrv_ms']),
        })

    dst = FIXTURES / "whoop.json"
    dst.write_text(json.dumps({'recovery_cases': rows}, indent=2, default=str))
    print("Wrote", dst)


def regenerate_prompts():
    # Construct prompts anchored to the other fixtures to keep consistency
    labs = json.loads((FIXTURES / "labs.json").read_text()) if (FIXTURES / "labs.json").exists() else {"cases": []}
    whoop = json.loads((FIXTURES / "whoop.json").read_text()) if (FIXTURES / "whoop.json").exists() else {"recovery_cases": []}

    prompts = []
    # If we have any labs cases, build some prompts
    for c in labs.get("cases", [])[:2]:
        name = c["analyte"]
        dates = [v["date"] for v in c["values"]]
        dates_s = ", ".join(map(str, dates))
        prompts.append({
            "category": "labs",
            "prompt": f"Compare my {name} values across {dates_s}.",
            "expect_contains": [name] + [str(d) for d in dates],
            "notes": "Auto-generated from labs fixture"
        })
    # WHOOP: pick last item
    if whoop.get("recovery_cases"):
        last = whoop["recovery_cases"][min(2, len(whoop["recovery_cases"]) - 1)]
        prompts.append({
            "category": "whoop",
            "prompt": f"What was my WHOOP recovery and strain on {last['date']}?",
            "expect_contains": [str(last['date']), "recovery", str(last.get('recovery_score')), "strain", str(last.get('strain'))],
            "notes": "Auto-generated from whoop fixture"
        })

    dst = FIXTURES / "prompts.json"
    dst.write_text(json.dumps({"cases": prompts}, indent=2))
    print("Wrote", dst)


def cli(argv=None):
    p = argparse.ArgumentParser(description="Regenerate test fixtures from parquet tables")
    p.add_argument("--whoop-days", type=int, default=7, help="Number of recent days to include in WHOOP fixture")
    args = p.parse_args(argv)

    FIXTURES.mkdir(parents=True, exist_ok=True)

    regenerate_labs()
    regenerate_meds()
    regenerate_whoop(days=args.whoop_days)
    regenerate_prompts()


if __name__ == "__main__":
    cli()
