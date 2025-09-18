"""
Centralized constants for processed data schemas and shared configuration.
These constants are imported by ingestion scripts and tools across the app
so that schemas are consistent and not guessed in multiple places.
"""
from __future__ import annotations

from typing import List, Dict

# Labs processed table schema
# Note: Some vendors may not provide all fields; missing columns should be filled with None.
# Includes optional current/previous values per Ways2Well comparative sections.
LABS_PROCESSED_COLS: List[str] = [
    "analyte",
    "value",
    "unit",
    "ref_low",
    "ref_high",
    "date",
    "source",
    "page",
    "source_type",
    "vendor",
    "flag",
    # Optional enhanced fields used by some parsers (e.g., Ways2Well)
    "current_value",
    "previous_value",
]

LABS_TABLE_FILE: str = "labs.parquet"
LABS_CORPUS_FILE: str = "labs_corpus.parquet"

# Medications raw table schema
MEDS_RAW_COLS: List[str] = [
    "medication",
    "dose",
    "dose_unit",
    "dose_frequency",
    "dose_frequency_unit",
    "date_start",
    "date_updated",
    "date_stop",
    "current",
]

# Medications processed table schema (canonical column names)
# Ingestion normalizes incoming column variants to this schema.
MEDS_PROCESSED_COLS: List[str] = [
    "name",
    "dose",
    "dose_unit",
    "frequency",
    "frequency_unit",
    "date_start",
    "date_updated",
    "date_stop",
    "current",
    "__source_file",
]

MEDS_COL_MAP: Dict[str, str] = dict(zip(MEDS_RAW_COLS, MEDS_PROCESSED_COLS[:-1]))

# Medications date columns
MEDS_DATE_COLS: List[str] = ["date_start", "date_updated", "date_stop"]

MEDS_TABLE_FILE: str = "meds.parquet"
MEDS_CORPUS_FILE: str = "meds_corpus.parquet"


# WHOOP processed table filenames (within processed tables directory)
WHOOP_TABLE_FILES: Dict[str, str] = {
    "sleeps": "whoop_sleeps.parquet",
    "workouts": "whoop_workouts.parquet",
    "physiological_cycles": "whoop_physiological_cycles.parquet",
    "journal_entries": "whoop_journal_entries.parquet",
}

# WHOOP corpus table filenames (within processed corpus directory)
WHOOP_CORPUS_FILE: str = "whoop_corpus.parquet"

# WHOOP raw and processed column lists (one-to-one mapping by index)
WHOOP_SLEEPS_RAW_COLS: List[str] = [
    "Cycle start time",
    "Sleep performance %",
    "Sleep onset",
    "In bed duration (min)",
    "Asleep duration (min)",
    "Light sleep duration (min)",
    "Deep (SWS) duration (min)",
    "REM duration (min)",
    "Awake duration (min)",
    "Sleep efficiency %",
    "Sleep consistency %",
    "Nap",
]

WHOOP_SLEEPS_PROCESSED_COLS: List[str] = [
    "date",
    "sleep_score",
    "sleep_start_time",
    "inbed_duration_min",
    "asleep_duration_min",
    "light_sleep_duration_min",
    "deep_sleep_duration_min",
    "rem_sleep_duration_min",
    "awake_duration_min",
    "efficiency_pct",
    "consistency_pct",
    "nap",
]

WHOOP_SLEEPS_COL_MAP: Dict[str, str] = dict(zip(WHOOP_SLEEPS_RAW_COLS, WHOOP_SLEEPS_PROCESSED_COLS))

WHOOP_RECOVERY_RAW_COLS: List[str] = [
    "Cycle start time",
    "Day Strain",
    "Recovery score %",
    "Resting heart rate (bpm)",
    "Heart rate variability (ms)",
]

WHOOP_RECOVERY_PROCESSED_COLS: List[str] = [
    "date",
    "strain",
    "recovery_score",
    "rhr_bpm",
    "hrv_ms",
]

WHOOP_RECOVERY_COL_MAP: Dict[str, str] = dict(zip(WHOOP_RECOVERY_RAW_COLS, WHOOP_RECOVERY_PROCESSED_COLS))

WHOOP_WORKOUTS_RAW_COLS: List[str] = [
    "Cycle start time",
    "Activity",
    "Duration (min)",
    "Day Strain",
    "Average heart rate (bpm)",
    "Max heart rate (bpm)",
    "Calories",
    "Heart rate zone 1",
    "Heart rate zone 2",
    "Heart rate zone 3",
    "Heart rate zone 4",
    "Heart rate zone 5",
]

WHOOP_WORKOUTS_PROCESSED_COLS: List[str] = [
    "date",
    "activity",
    "duration_min",
    "strain",
    "avg_hr_bpm",
    "max_hr_bpm",
    "calories",
    "hr_zone_1",
    "hr_zone_2",
    "hr_zone_3",
    "hr_zone_4",
    "hr_zone_5",
]

WHOOP_WORKOUTS_COL_MAP: Dict[str, str] = dict(zip(WHOOP_WORKOUTS_RAW_COLS, WHOOP_WORKOUTS_PROCESSED_COLS))

# Retrieval & Tooling Settings
MAX_CONTEXT_DOCS=20
MAX_DOCS_PER_RETRIEVER=20
ENABLE_TOOL_CALLING=1
PREFER_STRUCTURED=0