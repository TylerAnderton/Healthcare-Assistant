"""Unit tests for labs_tool NaN sanitization.

Missing values in the labs parquet (e.g. flag, which is NaN for most rows)
must surface as None in tool outputs, not float NaN — Pydantic output
validation (LabsLatestValueOutput etc.) rejects NaN for Optional[str] fields.
"""
import math

import pandas as pd
import pytest

from app.tools import labs_tool as LT


@pytest.fixture()
def labs_table(tmp_path):
    df = pd.DataFrame(
        [
            {
                "analyte": "ALT (SGPT)",
                "value": 38.0,
                "unit": "IU/L",
                "date": "2025-01-15",
                "ref_low": 9.0,
                "ref_high": 46.0,
                "flag": float("nan"),
                "vendor": "labcorp",
                "source": "labs.pdf",
                "page": 2,
            },
            {
                "analyte": "ALT (SGPT)",
                "value": 41.0,
                "unit": "IU/L",
                "date": "2024-06-01",
                "ref_low": 9.0,
                "ref_high": 46.0,
                "flag": float("nan"),
                "vendor": "labcorp",
                "source": "labs.pdf",
                "page": 2,
            },
            {
                "analyte": "HbA1c",
                "value": 5.4,
                "unit": "%",
                "date": "2025-01-15",
                "ref_low": 4.8,
                "ref_high": 5.6,
                "flag": "HIGH",
                "vendor": "labcorp",
                "source": "labs.pdf",
                "page": 3,
            },
        ]
    )
    path = tmp_path / "labs.parquet"
    df.to_parquet(path)
    return str(path)


def _is_nan(x):
    return isinstance(x, float) and math.isnan(x)


def test_latest_value_should_return_none_flag_when_flag_missing(labs_table):
    # Act
    result = LT.latest_value("ALT (SGPT)", table_path=labs_table)

    # Assert
    assert result["flag"] is None


def test_latest_value_should_preserve_present_flag(labs_table):
    # Act
    result = LT.latest_value("HbA1c", table_path=labs_table)

    # Assert
    assert result["flag"] == "HIGH"


def test_history_should_not_contain_nan_values(labs_table):
    # Act
    rows = LT.history("ALT (SGPT)", table_path=labs_table)

    # Assert
    assert not any(_is_nan(v) for row in rows for v in row.values())


def test_summary_should_return_none_flag_when_flag_missing(labs_table):
    # Act
    result = LT.summary("ALT (SGPT)", table_path=labs_table)

    # Assert
    assert result["flag"] is None


def test_latest_panel_should_not_contain_nan_values(labs_table):
    # Act
    rows = LT.latest_panel(table_path=labs_table)

    # Assert
    assert not any(_is_nan(v) for row in rows for v in row.values())
