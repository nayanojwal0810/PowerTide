"""Unit tests for PowerTide data coverage preflight utilities."""

from __future__ import annotations

import json
import urllib.error
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

from power_tide.preflight import (
    EXPECTED_HOLDOUT_HOURS,
    HOLDOUT_TARGET_END,
    HOLDOUT_TARGET_START,
    TEXAS_PROXY_LOCATIONS,
    WEATHER_MODEL_ID,
    WEATHER_VARIABLES,
    audit_local_eia_json_files,
    check_eia_api_coverage,
    compute_expected_weather_target_timestamps,
    generate_evaluation_date_range,
    query_ecmwf_ifs_run_audit,
)


def _make_mock_hourly_times(run_date: date, total_hours: int = 168) -> list[str]:
    """Helper to generate sequential ISO hourly timestamps starting from run initialization."""
    from datetime import datetime, timedelta

    start_dt = datetime(run_date.year, run_date.month, run_date.day, 0, 0)
    return [(start_dt + timedelta(hours=i)).strftime("%Y-%m-%dT%H:00") for i in range(total_hours)]


def _make_valid_locations_payload(run_date: date) -> list[dict]:
    """Helper to construct a valid 5-location mock response."""
    times = _make_mock_hourly_times(run_date, 168)
    locations = []
    for loc in TEXAS_PROXY_LOCATIONS:
        locations.append(
            {
                "latitude": loc["latitude"],
                "longitude": loc["longitude"],
                "hourly": {
                    "time": list(times),
                    "temperature_2m": [30.0] * 168,
                    "relative_humidity_2m": [50.0] * 168,
                    "wind_speed_10m": [15.0] * 168,
                    "shortwave_radiation": [250.0] * 168,
                },
            }
        )
    return locations


def test_generate_evaluation_date_range() -> None:
    dates = generate_evaluation_date_range()
    assert len(dates) == 92
    assert dates[0] == date(2026, 6, 30)
    assert dates[-1] == date(2026, 9, 29)


def test_proxy_locations_and_variables_definitions() -> None:
    assert len(TEXAS_PROXY_LOCATIONS) == 5
    names = {loc["name"] for loc in TEXAS_PROXY_LOCATIONS}
    assert names == {"Dallas", "Houston", "Austin", "San Antonio", "Midland"}
    assert WEATHER_MODEL_ID == "ecmwf_ifs"
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


def test_compute_expected_weather_target_timestamps() -> None:
    d = date(2026, 6, 30)
    targets = compute_expected_weather_target_timestamps(d)
    assert len(targets) == 24
    assert targets[0] == "2026-07-01T01:00"
    assert targets[-1] == "2026-07-02T00:00"


def test_audit_local_eia_valid_synthetic(tmp_path: Path) -> None:
    f1 = tmp_path / "demand_1.json"
    data = {
        "response": {
            "data": [
                {"period": "2024-01-01T00", "respondent": "ERCO", "type": "D", "value": 35000},
                {"period": "2024-01-01T01", "respondent": "ERCO", "type": "D", "value": 34000},
                {"period": "2024-01-01T02", "respondent": "ERCO", "type": "D", "value": 33000},
            ]
        }
    }
    f1.write_text(json.dumps(data), encoding="utf-8")
    summary = audit_local_eia_json_files([f1])
    assert summary.status == "PASS"
    assert summary.total_records == 3
    assert summary.unique_periods == 3
    assert summary.missing_periods_count == 0
    assert summary.duplicate_records_count == 0
    assert summary.null_value_count == 0
    assert summary.is_chronological is True


def test_audit_local_eia_gap_detection_synthetic(tmp_path: Path) -> None:
    f1 = tmp_path / "demand_gap.json"
    data = {
        "response": {
            "data": [
                {"period": "2024-01-01T00", "respondent": "ERCO", "type": "D", "value": 35000},
                # Missing 01:00 and 02:00
                {"period": "2024-01-01T03", "respondent": "ERCO", "type": "D", "value": 32000},
            ]
        }
    }
    f1.write_text(json.dumps(data), encoding="utf-8")
    summary = audit_local_eia_json_files([f1])
    assert summary.status == "FAIL"
    assert summary.missing_periods_count == 2
    assert "2024-01-01T01" in summary.missing_period_samples


def test_audit_local_eia_duplicate_and_type_mismatch(tmp_path: Path) -> None:
    f1 = tmp_path / "bad.json"
    data = {
        "response": {
            "data": [
                {"period": "2024-01-01T00", "respondent": "WRONG_RESP", "type": "DF", "value": 10},
                {"period": "2024-01-01T00", "respondent": "WRONG_RESP", "type": "DF", "value": 10},
            ]
        }
    }
    f1.write_text(json.dumps(data), encoding="utf-8")
    summary = audit_local_eia_json_files([f1], expected_type="D", expected_respondent="ERCO")
    assert summary.status == "FAIL"
    assert summary.duplicate_records_count == 1
    assert any("Unexpected types" in e for e in summary.errors)
    assert any("Unexpected respondents" in e for e in summary.errors)


def test_query_ecmwf_ifs_run_audit_success() -> None:
    d = date(2026, 7, 1)
    payload = _make_valid_locations_payload(d)
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = query_ecmwf_ifs_run_audit(d)

    assert res.available is True
    assert res.model_id == "ecmwf_ifs"
    assert res.locations_count == 5
    assert res.target_intervals_valid is True
    assert res.target_window_nulls == 0
    assert len(res.failure_reasons) == 0


def test_query_ecmwf_ifs_run_audit_missing_variable() -> None:
    d = date(2026, 7, 1)
    payload = _make_valid_locations_payload(d)
    # Delete wind_speed_10m from Dallas
    del payload[0]["hourly"]["wind_speed_10m"]

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = query_ecmwf_ifs_run_audit(d)

    assert res.available is False
    assert any("Missing required variable 'wind_speed_10m'" in r for r in res.failure_reasons)


def test_query_ecmwf_ifs_run_audit_short_array() -> None:
    d = date(2026, 7, 1)
    payload = _make_valid_locations_payload(d)
    # Truncate time array to 20 hours
    payload[1]["hourly"]["time"] = payload[1]["hourly"]["time"][:20]

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = query_ecmwf_ifs_run_audit(d)

    assert res.available is False
    assert any("Horizon too short" in r for r in res.failure_reasons)


def test_query_ecmwf_ifs_run_audit_incorrect_timestamps() -> None:
    d = date(2026, 7, 1)
    payload = _make_valid_locations_payload(d)
    # Tamper with index 25 timestamp in Houston
    payload[1]["hourly"]["time"][25] = "2026-07-02T05:00"

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = query_ecmwf_ifs_run_audit(d)

    assert res.available is False
    assert any("Target timestamps mismatch" in r for r in res.failure_reasons)


def test_query_ecmwf_ifs_run_audit_null_target_values() -> None:
    d = date(2026, 7, 1)
    payload = _make_valid_locations_payload(d)
    # Insert null in target slice of temperature_2m in Austin
    payload[2]["hourly"]["temperature_2m"][30] = None

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = query_ecmwf_ifs_run_audit(d)

    assert res.available is False
    assert res.target_window_nulls == 1
    assert any("temperature_2m' has 1 nulls in target window" in r for r in res.failure_reasons)


def test_query_ecmwf_ifs_run_audit_malformed_response() -> None:
    d = date(2026, 7, 1)
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"error": True, "reason": "server err"}).encode(
        "utf-8"
    )
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = query_ecmwf_ifs_run_audit(d)

    assert res.available is False
    assert any("not a list" in r for r in res.failure_reasons)


def test_query_ecmwf_ifs_run_audit_timeout() -> None:
    d = date(2026, 7, 1)
    with patch("urllib.request.urlopen", side_effect=TimeoutError("Request timed out")):
        res = query_ecmwf_ifs_run_audit(d)

    assert res.available is False
    assert any("timed out" in r for r in res.failure_reasons)


def test_check_eia_api_coverage_missing_key() -> None:
    with patch.dict("os.environ", {}, clear=True):
        res = check_eia_api_coverage()
        assert res.api_key_configured is False
        assert res.status == "BLOCKED"
        assert "EIA_API_KEY environment variable is not configured" in res.message


def test_check_eia_api_coverage_success_mocked() -> None:
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"response": {"total": 2208}}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch.dict("os.environ", {"EIA_API_KEY": "test_key_abc"}):
        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = check_eia_api_coverage()
            assert res.api_key_configured is True
            assert res.status == "GO"
            assert res.total_reported == 2208


def test_check_eia_api_coverage_http_error_mocked() -> None:
    with patch.dict("os.environ", {"EIA_API_KEY": "invalid_key"}):
        err = urllib.error.HTTPError("https://api.eia.gov", 403, "Forbidden", {}, None)
        with patch("urllib.request.urlopen", side_effect=err):
            res = check_eia_api_coverage()
            assert res.api_key_configured is True
            assert res.status == "BLOCKED"
            assert "HTTP 403" in res.message
