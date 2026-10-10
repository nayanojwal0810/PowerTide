"""Unit tests for PowerTide data coverage preflight utilities."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from power_tide.preflight import (
    EXPECTED_HOLDOUT_HOURS,
    HOLDOUT_TARGET_END,
    HOLDOUT_TARGET_START,
    TEXAS_PROXY_LOCATIONS,
    WEATHER_VARIABLES,
    audit_local_eia_json_files,
    generate_evaluation_date_range,
    query_ecmwf_ifs_run_audit,
)


def test_generate_evaluation_date_range() -> None:
    dates = generate_evaluation_date_range()
    assert len(dates) == 92
    assert dates[0] == date(2026, 6, 30)
    assert dates[-1] == date(2026, 9, 29)


def test_proxy_locations_and_variables_definitions() -> None:
    assert len(TEXAS_PROXY_LOCATIONS) == 5
    names = {loc["name"] for loc in TEXAS_PROXY_LOCATIONS}
    assert names == {"Dallas", "Houston", "Austin", "San Antonio", "Midland"}
    assert len(WEATHER_VARIABLES) == 4
    assert set(WEATHER_VARIABLES) == {
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "shortwave_radiation",
    }
    assert EXPECTED_HOLDOUT_HOURS == 2_208
    assert HOLDOUT_TARGET_START == "2026-07-01T01"
    assert HOLDOUT_TARGET_END == "2026-10-01T00"


def test_audit_local_eia_synthetic_files(tmp_path: Path) -> None:
    # Create two synthetic JSON batch files
    f1 = tmp_path / "batch_1.json"
    f2 = tmp_path / "batch_2.json"

    data1 = {
        "response": {
            "data": [
                {"period": "2024-01-01T00", "respondent": "ERCO", "type": "D", "value": 35000},
                {"period": "2024-01-01T01", "respondent": "ERCO", "type": "D", "value": 34000},
            ]
        }
    }
    data2 = {
        "response": {
            "data": [
                {"period": "2024-01-01T02", "respondent": "ERCO", "type": "D", "value": 33000},
                # Duplicate period with null value to verify anomaly counters
                {"period": "2024-01-01T02", "respondent": "ERCO", "type": "D", "value": None},
                # An observation inside holdout window
                {"period": "2026-07-01T05", "respondent": "ERCO", "type": "D", "value": 55000},
            ]
        }
    }

    f1.write_text(json.dumps(data1), encoding="utf-8")
    f2.write_text(json.dumps(data2), encoding="utf-8")

    summary = audit_local_eia_json_files([f1, f2])

    assert summary.dataset_type == "D"
    assert summary.total_records == 5
    assert summary.unique_periods == 4
    assert summary.duplicate_periods == 1
    assert summary.null_value_count == 1
    assert summary.min_period == "2024-01-01T00"
    assert summary.max_period == "2026-07-01T05"
    assert summary.holdout_hours_present == 1


def test_audit_local_eia_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        audit_local_eia_json_files([])


def test_query_ecmwf_ifs_run_audit_success() -> None:
    # Synthetic mocked response for 5 locations with 168 hours each
    mock_locations_data = []
    for _ in range(5):
        mock_locations_data.append(
            {
                "hourly": {
                    "time": [f"t_{i}" for i in range(168)],
                    "temperature_2m": [30.0] * 168,
                    "relative_humidity_2m": [50.0] * 168,
                    "wind_speed_10m": [15.0] * 168,
                    "shortwave_radiation": [200.0] * 168,
                }
            }
        )

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_locations_data).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = query_ecmwf_ifs_run_audit(date(2026, 7, 1))

    assert res.available is True
    assert res.locations_count == 5
    assert res.lead_hours_count == 168
    assert res.target_window_nulls == 0


def test_query_ecmwf_ifs_run_audit_failure_handling() -> None:
    with patch("urllib.request.urlopen", side_effect=TimeoutError("Connection timed out")):
        res = query_ecmwf_ifs_run_audit(date(2026, 7, 1))

    assert res.available is False
    assert res.locations_count == 0
    assert res.target_window_nulls == -1
