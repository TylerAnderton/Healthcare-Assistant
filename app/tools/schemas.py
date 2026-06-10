"""
Pydantic input and output schemas for all 15 structured tools.

TOOL_SCHEMAS maps tool name → (InputModel, OutputModel | None).
OutputModel=None means output validation is skipped (str-returning tools, List[str] tools).
"""

import re
from typing import Optional, Any
from pydantic import BaseModel, field_validator


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _check_date_required(v: str) -> str:
    if not _DATE_RE.match(v):
        raise ValueError(f"date must be YYYY-MM-DD, got: {v!r}")
    return v


def _check_date_optional(v: Optional[str]) -> Optional[str]:
    if v is not None and not _DATE_RE.match(v):
        raise ValueError(f"date must be YYYY-MM-DD, got: {v!r}")
    return v


def _check_nonempty(v: str) -> str:
    if not v.strip():
        raise ValueError("must be a non-empty string")
    return v.strip()


# ============================================================================
# Labs Input Models
# ============================================================================

class LabsListAnalytesInput(BaseModel):
    prefix: Optional[str] = None
    table_path: Optional[str] = None


class LabsLatestValueInput(BaseModel):
    analyte: str
    table_path: Optional[str] = None

    @field_validator("analyte")
    @classmethod
    def analyte_nonempty(cls, v: str) -> str:
        return _check_nonempty(v)


class LabsHistoryInput(BaseModel):
    analyte: str
    limit: int = 10
    ascending: bool = True
    table_path: Optional[str] = None

    @field_validator("analyte")
    @classmethod
    def analyte_nonempty(cls, v: str) -> str:
        return _check_nonempty(v)

    @field_validator("limit")
    @classmethod
    def limit_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("limit must be >= 1")
        return v


class LabsSummaryInput(BaseModel):
    analyte: str
    table_path: Optional[str] = None

    @field_validator("analyte")
    @classmethod
    def analyte_nonempty(cls, v: str) -> str:
        return _check_nonempty(v)


class LabsValueOnDateInput(BaseModel):
    analyte: str
    date: str

    @field_validator("analyte")
    @classmethod
    def analyte_nonempty(cls, v: str) -> str:
        return _check_nonempty(v)

    @field_validator("date")
    @classmethod
    def date_format(cls, v: str) -> str:
        return _check_date_required(v)


class LabsPanelInput(BaseModel):
    pass


# ============================================================================
# Labs Output Models
# ============================================================================

class LabsLatestValueOutput(BaseModel):
    model_config = {"extra": "ignore"}

    analyte: str
    value: Optional[Any] = None
    unit: Optional[str] = None
    date: Optional[str] = None
    vendor: Optional[str] = None
    ref_low: Optional[float] = None
    ref_high: Optional[float] = None
    flag: Optional[str] = None
    source: Optional[str] = None
    page: Optional[Any] = None


class LabsHistoryItemOutput(BaseModel):
    model_config = {"extra": "ignore"}

    date: Optional[str] = None
    value: Optional[Any] = None
    unit: Optional[str] = None
    ref_low: Optional[float] = None
    ref_high: Optional[float] = None
    flag: Optional[str] = None
    vendor: Optional[str] = None
    source: Optional[str] = None
    page: Optional[Any] = None


class LabsSummaryOutput(BaseModel):
    model_config = {"extra": "ignore"}

    analyte: Optional[str] = None
    count: Optional[int] = None
    last_value: Optional[Any] = None
    unit: Optional[str] = None
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    ref_low: Optional[float] = None
    ref_high: Optional[float] = None
    flag: Optional[str] = None


class LabsValueOnDateOutput(BaseModel):
    model_config = {"extra": "ignore"}

    analyte: Optional[str] = None
    date: Optional[str] = None
    value: Optional[Any] = None
    unit: Optional[str] = None
    ref_low: Optional[float] = None
    ref_high: Optional[float] = None
    flag: Optional[str] = None


# ============================================================================
# Meds Input Models
# ============================================================================

class MedsListMedicationsInput(BaseModel):
    pass


class MedsListCurrentInput(BaseModel):
    date: Optional[str] = None

    @field_validator("date")
    @classmethod
    def date_format(cls, v: Optional[str]) -> Optional[str]:
        return _check_date_optional(v)


class MedsHistoryInput(BaseModel):
    medication: str
    fuzzy: bool = True

    @field_validator("medication")
    @classmethod
    def medication_nonempty(cls, v: str) -> str:
        return _check_nonempty(v)


class MedsDosageOnDateInput(BaseModel):
    medication: str
    date: Optional[str] = None
    fuzzy: bool = True

    @field_validator("medication")
    @classmethod
    def medication_nonempty(cls, v: str) -> str:
        return _check_nonempty(v)

    @field_validator("date")
    @classmethod
    def date_format(cls, v: Optional[str]) -> Optional[str]:
        return _check_date_optional(v)


class MedsTimelineInput(BaseModel):
    pass


# ============================================================================
# Meds Output Models
# ============================================================================

class MedsListCurrentItemOutput(BaseModel):
    model_config = {"extra": "ignore"}

    name: Optional[str] = None
    dose: Optional[Any] = None
    dose_unit: Optional[str] = None
    frequency: Optional[Any] = None
    frequency_unit: Optional[str] = None
    date_start: Optional[str] = None
    date_stop: Optional[str] = None
    current: Optional[Any] = None


class MedsHistoryItemOutput(BaseModel):
    model_config = {"extra": "ignore"}

    name: Optional[str] = None
    dose: Optional[Any] = None
    dose_unit: Optional[str] = None
    date_start: Optional[str] = None
    date_stop: Optional[str] = None
    current: Optional[Any] = None


class MedsDosageOnDateOutput(BaseModel):
    model_config = {"extra": "ignore"}

    name: Optional[str] = None
    dose: Optional[Any] = None
    dose_unit: Optional[str] = None
    frequency: Optional[Any] = None
    frequency_unit: Optional[str] = None
    date_start: Optional[str] = None
    date_stop: Optional[str] = None


# ============================================================================
# WHOOP Input Models
# ============================================================================

class WhoopSleepsOnDateInput(BaseModel):
    date: str

    @field_validator("date")
    @classmethod
    def date_format(cls, v: str) -> str:
        return _check_date_required(v)


class WhoopRecoveryOnDateInput(BaseModel):
    date: str

    @field_validator("date")
    @classmethod
    def date_format(cls, v: str) -> str:
        return _check_date_required(v)


class WhoopWorkoutsOnDateInput(BaseModel):
    date: str

    @field_validator("date")
    @classmethod
    def date_format(cls, v: str) -> str:
        return _check_date_required(v)


class WhoopRecentInput(BaseModel):
    days: int = 7

    @field_validator("days")
    @classmethod
    def days_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("days must be >= 1")
        return v


# ============================================================================
# WHOOP Output Models
# ============================================================================

class WhoopRecoveryOnDateOutput(BaseModel):
    model_config = {"extra": "ignore"}

    date: Optional[str] = None
    strain: Optional[float] = None
    recovery_score: Optional[float] = None
    rhr_bpm: Optional[float] = None
    hrv_ms: Optional[float] = None


# ============================================================================
# Schema Registry
# ============================================================================

TOOL_SCHEMAS: dict = {
    "labs_list_analytes":             (LabsListAnalytesInput, None),
    "labs_latest_value":              (LabsLatestValueInput, LabsLatestValueOutput),
    "labs_history":                   (LabsHistoryInput, LabsHistoryItemOutput),
    "labs_summary":                   (LabsSummaryInput, LabsSummaryOutput),
    "labs_value_on_date":             (LabsValueOnDateInput, LabsValueOnDateOutput),
    "labs_panel":                     (LabsPanelInput, None),
    "meds_list_medications":          (MedsListMedicationsInput, None),
    "meds_list_current":              (MedsListCurrentInput, MedsListCurrentItemOutput),
    "meds_history":                   (MedsHistoryInput, MedsHistoryItemOutput),
    "meds_dosage_on_date":            (MedsDosageOnDateInput, MedsDosageOnDateOutput),
    "meds_timeline":                  (MedsTimelineInput, None),
    "whoop_sleeps_on_date":           (WhoopSleepsOnDateInput, None),
    "whoop_recovery_metrics_on_date": (WhoopRecoveryOnDateInput, WhoopRecoveryOnDateOutput),
    "whoop_workouts_on_date":         (WhoopWorkoutsOnDateInput, None),
    "whoop_recent":                   (WhoopRecentInput, None),
}
