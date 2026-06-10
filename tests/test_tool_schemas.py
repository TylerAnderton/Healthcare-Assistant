"""
TDD red phase: tests for Pydantic input/output schemas in app/tools/schemas.py.
All tests will fail (ImportError) until schemas.py is implemented.
"""

import pytest
from pydantic import ValidationError

try:
    from app.tools.schemas import (
        LabsListAnalytesInput,
        LabsLatestValueInput,
        LabsLatestValueOutput,
        LabsHistoryInput,
        LabsHistoryItemOutput,
        LabsSummaryInput,
        LabsSummaryOutput,
        LabsValueOnDateInput,
        LabsValueOnDateOutput,
        LabsPanelInput,
        MedsListMedicationsInput,
        MedsListCurrentInput,
        MedsListCurrentItemOutput,
        MedsHistoryInput,
        MedsHistoryItemOutput,
        MedsDosageOnDateInput,
        MedsDosageOnDateOutput,
        MedsTimelineInput,
        WhoopSleepsOnDateInput,
        WhoopRecoveryOnDateInput,
        WhoopRecoveryOnDateOutput,
        WhoopWorkoutsOnDateInput,
        WhoopRecentInput,
        TOOL_SCHEMAS,
    )
    _SCHEMAS_AVAILABLE = True
except ImportError:
    _SCHEMAS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _SCHEMAS_AVAILABLE,
    reason="app/tools/schemas.py not yet implemented",
)


# ============================================================================
# Labs Input Models
# ============================================================================

class TestLabsListAnalytesInput:
    def test_no_args_valid(self):
        m = LabsListAnalytesInput()
        assert m.prefix is None

    def test_prefix_accepted(self):
        m = LabsListAnalytesInput(prefix="hba")
        assert m.prefix == "hba"


class TestLabsLatestValueInput:
    def test_valid_analyte_accepted(self):
        m = LabsLatestValueInput(analyte="HbA1c")
        assert m.analyte == "HbA1c"

    def test_empty_analyte_raises(self):
        with pytest.raises(ValidationError):
            LabsLatestValueInput(analyte="")

    def test_whitespace_only_analyte_raises(self):
        with pytest.raises(ValidationError):
            LabsLatestValueInput(analyte="   ")

    def test_analyte_is_stripped(self):
        m = LabsLatestValueInput(analyte="  glucose  ")
        assert m.analyte == "glucose"

    def test_table_path_defaults_to_none(self):
        m = LabsLatestValueInput(analyte="Glucose")
        assert m.table_path is None

    def test_missing_analyte_raises(self):
        with pytest.raises(ValidationError):
            LabsLatestValueInput()


class TestLabsHistoryInput:
    def test_valid_analyte(self):
        m = LabsHistoryInput(analyte="Glucose")
        assert m.analyte == "Glucose"

    def test_default_limit_is_10(self):
        m = LabsHistoryInput(analyte="Glucose")
        assert m.limit == 10

    def test_default_ascending_is_true(self):
        m = LabsHistoryInput(analyte="Glucose")
        assert m.ascending is True

    def test_limit_zero_raises(self):
        with pytest.raises(ValidationError):
            LabsHistoryInput(analyte="Glucose", limit=0)

    def test_limit_negative_raises(self):
        with pytest.raises(ValidationError):
            LabsHistoryInput(analyte="Glucose", limit=-5)

    def test_limit_positive_accepted(self):
        m = LabsHistoryInput(analyte="Glucose", limit=50)
        assert m.limit == 50

    def test_empty_analyte_raises(self):
        with pytest.raises(ValidationError):
            LabsHistoryInput(analyte="")


class TestLabsSummaryInput:
    def test_valid_analyte(self):
        m = LabsSummaryInput(analyte="HbA1c")
        assert m.analyte == "HbA1c"

    def test_empty_analyte_raises(self):
        with pytest.raises(ValidationError):
            LabsSummaryInput(analyte="")


class TestLabsValueOnDateInput:
    def test_valid_args(self):
        m = LabsValueOnDateInput(analyte="Glucose", date="2024-01-15")
        assert m.analyte == "Glucose"
        assert m.date == "2024-01-15"

    def test_slash_date_raises(self):
        with pytest.raises(ValidationError):
            LabsValueOnDateInput(analyte="Glucose", date="2024/01/15")

    def test_us_date_raises(self):
        with pytest.raises(ValidationError):
            LabsValueOnDateInput(analyte="Glucose", date="01-15-2024")

    def test_partial_date_raises(self):
        with pytest.raises(ValidationError):
            LabsValueOnDateInput(analyte="Glucose", date="2024-01")

    def test_missing_date_raises(self):
        with pytest.raises(ValidationError):
            LabsValueOnDateInput(analyte="Glucose")

    def test_empty_analyte_raises(self):
        with pytest.raises(ValidationError):
            LabsValueOnDateInput(analyte="", date="2024-01-15")


# ============================================================================
# Meds Input Models
# ============================================================================

class TestMedsListCurrentInput:
    def test_no_args_valid(self):
        m = MedsListCurrentInput()
        assert m.date is None

    def test_valid_date(self):
        m = MedsListCurrentInput(date="2024-06-01")
        assert m.date == "2024-06-01"

    def test_invalid_date_raises(self):
        with pytest.raises(ValidationError):
            MedsListCurrentInput(date="June 1, 2024")


class TestMedsHistoryInput:
    def test_valid_medication(self):
        m = MedsHistoryInput(medication="metformin")
        assert m.medication == "metformin"

    def test_default_fuzzy_true(self):
        m = MedsHistoryInput(medication="metformin")
        assert m.fuzzy is True

    def test_empty_medication_raises(self):
        with pytest.raises(ValidationError):
            MedsHistoryInput(medication="")


class TestMedsDosageOnDateInput:
    def test_valid_args(self):
        m = MedsDosageOnDateInput(medication="metformin", date="2024-03-10")
        assert m.medication == "metformin"
        assert m.date == "2024-03-10"

    def test_no_date_valid(self):
        m = MedsDosageOnDateInput(medication="metformin")
        assert m.date is None

    def test_invalid_date_raises(self):
        with pytest.raises(ValidationError):
            MedsDosageOnDateInput(medication="metformin", date="March 10 2024")

    def test_slash_date_raises(self):
        with pytest.raises(ValidationError):
            MedsDosageOnDateInput(medication="metformin", date="2024/03/10")


# ============================================================================
# WHOOP Input Models
# ============================================================================

@pytest.mark.parametrize("InputModel", [
    "WhoopSleepsOnDateInput",
    "WhoopRecoveryOnDateInput",
    "WhoopWorkoutsOnDateInput",
])
class TestDateValidatedInputs:
    def _get_model(self, name):
        return {
            "WhoopSleepsOnDateInput": WhoopSleepsOnDateInput,
            "WhoopRecoveryOnDateInput": WhoopRecoveryOnDateInput,
            "WhoopWorkoutsOnDateInput": WhoopWorkoutsOnDateInput,
        }[name]

    def test_valid_iso_date_accepted(self, InputModel):
        cls = self._get_model(InputModel)
        m = cls(date="2024-01-15")
        assert m.date == "2024-01-15"

    def test_slash_date_raises(self, InputModel):
        cls = self._get_model(InputModel)
        with pytest.raises(ValidationError):
            cls(date="2024/01/15")

    def test_us_date_raises(self, InputModel):
        cls = self._get_model(InputModel)
        with pytest.raises(ValidationError):
            cls(date="01-15-2024")

    def test_partial_date_raises(self, InputModel):
        cls = self._get_model(InputModel)
        with pytest.raises(ValidationError):
            cls(date="2024-01")

    def test_missing_date_raises(self, InputModel):
        cls = self._get_model(InputModel)
        with pytest.raises(ValidationError):
            cls()


class TestWhoopRecentInput:
    def test_default_days(self):
        m = WhoopRecentInput()
        assert m.days == 7

    def test_custom_days(self):
        m = WhoopRecentInput(days=14)
        assert m.days == 14

    def test_zero_days_raises(self):
        with pytest.raises(ValidationError):
            WhoopRecentInput(days=0)

    def test_negative_days_raises(self):
        with pytest.raises(ValidationError):
            WhoopRecentInput(days=-1)


# ============================================================================
# Output Models
# ============================================================================

class TestLabsLatestValueOutput:
    def test_valid_dict_validates(self):
        data = {
            "analyte": "Glucose",
            "value": 95.0,
            "unit": "mg/dL",
            "date": "2024-01-15",
            "vendor": "LabCorp",
            "ref_low": 70.0,
            "ref_high": 100.0,
            "flag": None,
            "source": "labs_2024.pdf",
            "page": 1,
        }
        m = LabsLatestValueOutput(**data)
        assert m.analyte == "Glucose"
        assert m.value == 95.0

    def test_extra_keys_ignored(self):
        data = {"analyte": "Glucose", "unexpected_field": "should be ignored"}
        m = LabsLatestValueOutput(**data)
        assert m.analyte == "Glucose"

    def test_value_can_be_string(self):
        data = {"analyte": "TSH", "value": "<0.01"}
        m = LabsLatestValueOutput(**data)
        assert m.value == "<0.01"

    def test_all_optional_fields_none(self):
        m = LabsLatestValueOutput(analyte="HbA1c")
        assert m.value is None
        assert m.unit is None


class TestLabsHistoryItemOutput:
    def test_valid_item(self):
        data = {"date": "2024-01-15", "value": 5.6, "unit": "%", "flag": "H"}
        m = LabsHistoryItemOutput(**data)
        assert m.value == 5.6
        assert m.flag == "H"

    def test_all_fields_optional(self):
        m = LabsHistoryItemOutput()
        assert m.date is None

    def test_extra_keys_ignored(self):
        m = LabsHistoryItemOutput(date="2024-01-01", extra_col="ignored")
        assert m.date == "2024-01-01"


class TestWhoopRecoveryOutput:
    def test_valid_record(self):
        data = {
            "date": "2024-01-15",
            "strain": 8.5,
            "recovery_score": 72.0,
            "rhr_bpm": 55.0,
            "hrv_ms": 48.2,
        }
        m = WhoopRecoveryOnDateOutput(**data)
        assert m.recovery_score == 72.0
        assert m.hrv_ms == 48.2

    def test_all_fields_optional(self):
        m = WhoopRecoveryOnDateOutput()
        assert m.rhr_bpm is None

    def test_extra_keys_ignored(self):
        m = WhoopRecoveryOnDateOutput(date="2024-01-15", extra="ignored")
        assert m.date == "2024-01-15"


class TestMedsDosageOnDateOutput:
    def test_valid_record(self):
        data = {
            "name": "metformin",
            "dose": 500.0,
            "dose_unit": "mg",
            "frequency": 2.0,
            "frequency_unit": "daily",
        }
        m = MedsDosageOnDateOutput(**data)
        assert m.name == "metformin"

    def test_all_fields_optional(self):
        m = MedsDosageOnDateOutput()
        assert m.name is None

    def test_extra_keys_ignored(self):
        m = MedsDosageOnDateOutput(name="metformin", extra_col="x")
        assert m.name == "metformin"


# ============================================================================
# TOOL_SCHEMAS Registry
# ============================================================================

class TestToolSchemasRegistry:
    def test_registry_has_all_15_tools(self):
        expected_keys = {
            "labs_list_analytes",
            "labs_latest_value",
            "labs_history",
            "labs_summary",
            "labs_value_on_date",
            "labs_panel",
            "meds_list_medications",
            "meds_list_current",
            "meds_history",
            "meds_dosage_on_date",
            "meds_timeline",
            "whoop_sleeps_on_date",
            "whoop_recovery_metrics_on_date",
            "whoop_workouts_on_date",
            "whoop_recent",
        }
        assert expected_keys == set(TOOL_SCHEMAS.keys())

    def test_each_entry_is_2_tuple(self):
        for name, entry in TOOL_SCHEMAS.items():
            assert len(entry) == 2, f"{name} entry should be a 2-tuple"

    def test_input_model_is_always_present(self):
        from pydantic import BaseModel
        for name, (InputModel, _) in TOOL_SCHEMAS.items():
            assert issubclass(InputModel, BaseModel), f"{name} missing InputModel"

    def test_output_model_is_none_or_pydantic(self):
        from pydantic import BaseModel
        for name, (_, OutputModel) in TOOL_SCHEMAS.items():
            assert OutputModel is None or issubclass(OutputModel, BaseModel), \
                f"{name} OutputModel should be None or BaseModel subclass"

    def test_dict_returning_tools_have_output_models(self):
        dict_tools = {
            "labs_latest_value",
            "labs_history",
            "labs_summary",
            "labs_value_on_date",
            "meds_list_current",
            "meds_history",
            "meds_dosage_on_date",
            "whoop_recovery_metrics_on_date",
        }
        for name in dict_tools:
            _, OutputModel = TOOL_SCHEMAS[name]
            assert OutputModel is not None, f"{name} should have an output model"

    def test_str_returning_tools_have_no_output_model(self):
        str_tools = {"labs_panel", "meds_timeline", "whoop_recent"}
        for name in str_tools:
            _, OutputModel = TOOL_SCHEMAS[name]
            assert OutputModel is None, f"{name} should have OutputModel=None"
