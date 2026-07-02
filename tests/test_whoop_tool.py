"""Unit tests for whoop_tool.recent() date-window behavior.

`recent()` previously anchored its window strictly to `datetime.now()`, so
stale ingestions (e.g. data ending months before the wall clock) returned
nothing and forced the agent to claim "no recent data" — fixed by adding a
fallback that re-windows to the maximum date present in the WHOOP tables.
"""
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

from app.tools import whoop_tool


def _df_with_dates(*dates: str) -> pd.DataFrame:
    return pd.DataFrame({"date": list(dates), "value": list(range(len(dates)))})


def test_recent_returns_wall_clock_window_when_data_fresh():
    # Arrange — sub-tools return data on the first (wall-clock) call.
    fresh = _df_with_dates("2026-06-10", "2026-06-11", "2026-06-12")
    with (
        patch.object(whoop_tool, "sleeps", return_value=fresh.copy()) as mock_sleeps,
        patch.object(whoop_tool, "recovery", return_value=fresh.copy()),
        patch.object(whoop_tool, "workouts", return_value=fresh.copy()),
        patch.object(whoop_tool, "_max_data_date") as mock_max,
    ):
        # Act
        result = whoop_tool.recent(days=7)

    # Assert
    assert not result["sleeps"].empty
    assert mock_sleeps.call_count == 1, "no fallback should run when data is fresh"
    mock_max.assert_not_called()


def test_recent_falls_back_to_max_data_date_when_wall_clock_window_empty():
    # Arrange — first call empty, second (fallback) call returns data.
    empty = pd.DataFrame()
    fallback = _df_with_dates("2025-08-28", "2025-08-29", "2025-08-30")
    sleeps_mock = MagicMock(side_effect=[empty.copy(), fallback.copy()])
    recovery_mock = MagicMock(side_effect=[empty.copy(), fallback.copy()])
    workouts_mock = MagicMock(side_effect=[empty.copy(), fallback.copy()])
    with (
        patch.object(whoop_tool, "sleeps", sleeps_mock),
        patch.object(whoop_tool, "recovery", recovery_mock),
        patch.object(whoop_tool, "workouts", workouts_mock),
        patch.object(whoop_tool, "_max_data_date", return_value=date(2025, 8, 30)),
    ):
        # Act
        result = whoop_tool.recent(days=7)

    # Assert
    assert not result["sleeps"].empty
    assert sleeps_mock.call_count == 2
    fallback_start = sleeps_mock.call_args_list[1].kwargs["start"]
    assert fallback_start == "2025-08-24"  # 2025-08-30 minus 6 days


def test_recent_returns_empty_when_no_data_anywhere():
    # Arrange — no fallback date available; all sub-tools stay empty.
    empty = pd.DataFrame()
    with (
        patch.object(whoop_tool, "sleeps", return_value=empty.copy()),
        patch.object(whoop_tool, "recovery", return_value=empty.copy()),
        patch.object(whoop_tool, "workouts", return_value=empty.copy()),
        patch.object(whoop_tool, "_max_data_date", return_value=None),
    ):
        # Act
        result = whoop_tool.recent(days=7)

    # Assert
    assert result["sleeps"].empty
    assert result["recovery"].empty
    assert result["workouts"].empty
