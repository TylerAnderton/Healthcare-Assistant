"""Unit tests for structured_context.load_whoop_recent.

whoop_tool.recent() returns a dict of DataFrames, but load_whoop_recent
iterated the values as lists of dicts — `DataFrame or []` raised
"truth value of a DataFrame is ambiguous", silently swallowed by the
except, so the WHOOP snapshot block was always empty.
"""
from unittest.mock import patch

import pandas as pd

from app.tools import structured_context


def test_load_whoop_recent_renders_recovery_from_dataframes():
    # Arrange — recent() returns DataFrames, not lists of dicts.
    fake = {
        "sleeps": pd.DataFrame(),
        "recovery": pd.DataFrame(
            [{"date": "2025-08-30", "recovery_score": 44, "rhr_bpm": 54, "hrv_ms": 62}]
        ),
        "workouts": pd.DataFrame(),
    }
    with patch.object(structured_context, "whoop_recent", return_value=fake):
        # Act
        block = structured_context.load_whoop_recent(days=7)

    # Assert
    assert "2025-08-30" in block
    assert "recovery 44" in block
    assert "RHR 54 bpm" in block


def test_load_whoop_recent_returns_empty_when_all_frames_empty():
    # Arrange
    fake = {
        "sleeps": pd.DataFrame(),
        "recovery": pd.DataFrame(),
        "workouts": pd.DataFrame(),
    }
    with patch.object(structured_context, "whoop_recent", return_value=fake):
        # Act
        block = structured_context.load_whoop_recent(days=7)

    # Assert
    assert block == ""
