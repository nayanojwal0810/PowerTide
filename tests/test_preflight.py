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
    LOCAL_DEMAND_EXPECTED_END,
    LOCAL_DEMAND_EXPECTED_HOURS,
    LOCAL_DEMAND_EXPECTED_START,
    TEXAS_PROXY_LOCATIONS,
    WEATHER_MODEL_ID,
    WEATHER_VARIABLES,
    audit_local_eia_json_files,
    check_eia_api_coverage,
    compute_expected_eia_hourly_sequence,
    compute_expected_weather_target_timestamps,
    generate_evaluation_date_range,
    get_configured_eia_api_key,
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


def _make_mock_eia_payload(
    count: int = 2208,
    start_period: str = HOLDOUT_TARGET_START,
    end_period: str = HOLDOUT_TARGET_END,
    respondent: str = "ERCO",
    series_type: str = "D",
    value: float = 45000.0,
) -> dict:
    """Helper to construct a valid EIA API v2 holdout response payload."""
    periods = compute_expected_eia_hourly_sequence(start_period, end_period)[:count]
    rows = [
        {"period": p, "respondent": respondent, "type": series_type, "value": value}
        for p in periods
    ]
    return {
        "response": {
            "total": str(len(rows)),
            "data": rows,
        }
    }


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
    assert LOCAL_DEMAND_EXPECTED_START == "2024-01-01T00"
    assert LOCAL_DEMAND_EXPECTED_END == "2025-12-31T23"
    assert LOCAL_DEMAND_EXPECTED_HOURS == 17_544


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


def test_audit_local_eia_missing_period_field(tmp_path: Path) -> None:
    f = tmp_path / "missing_period.json"
    data = {
        "response": {
            "data": [
                {"respondent": "ERCO", "type": "D", "value": 35000},
                {"period": "2024-01-01T01", "respondent": "ERCO", "type": "D", "value": 34000},
            ]
        }
    }
    f.write_text(json.dumps(data), encoding="utf-8")
    summary = audit_local_eia_json_files([f])
    assert summary.status == "FAIL"
    assert any("missing a valid 'period' field" in err for err in summary.errors)


def test_audit_local_eia_null_value(tmp_path: Path) -> None:
    f = tmp_path / "null_value.json"
    data = {
        "response": {
            "data": [
                {"period": "2024-01-01T00", "respondent": "ERCO", "type": "D", "value": None},
                {"period": "2024-01-01T01", "respondent": "ERCO", "type": "D", "value": 34000},
            ]
        }
    }
    f.write_text(json.dumps(data), encoding="utf-8")
    summary = audit_local_eia_json_files([f])
    assert summary.status == "FAIL"
    assert summary.null_value_count == 1
    assert any("null demand values" in err for err in summary.errors)


def test_audit_local_eia_duplicate_periods_different_values(tmp_path: Path) -> None:
    f = tmp_path / "dup_diff_val.json"
    data = {
        "response": {
            "data": [
                {"period": "2024-01-01T00", "respondent": "ERCO", "type": "D", "value": 35000},
                {"period": "2024-01-01T00", "respondent": "ERCO", "type": "D", "value": 42000},
                {"period": "2024-01-01T01", "respondent": "ERCO", "type": "D", "value": 34000},
            ]
        }
    }
    f.write_text(json.dumps(data), encoding="utf-8")
    summary = audit_local_eia_json_files([f])
    assert summary.status == "FAIL"
    assert summary.duplicate_periods_count == 1
    assert any("duplicate hourly period timestamps" in err for err in summary.errors)


def test_audit_local_eia_invalid_numeric_values(tmp_path: Path) -> None:
    f = tmp_path / "invalid_vals.json"
    data = {
        "response": {
            "data": [
                {"period": "2024-01-01T00", "respondent": "ERCO", "type": "D", "value": -50.0},
                {
                    "period": "2024-01-01T01",
                    "respondent": "ERCO",
                    "type": "D",
                    "value": "not-a-number",
                },
                {"period": "2024-01-01T02", "respondent": "ERCO", "type": "D", "value": 34000},
            ]
        }
    }
    f.write_text(json.dumps(data), encoding="utf-8")
    summary = audit_local_eia_json_files([f])
    assert summary.status == "FAIL"
    assert summary.invalid_value_count == 2
    assert any("invalid or negative numeric values" in err for err in summary.errors)


def test_audit_local_eia_unsorted_records(tmp_path: Path) -> None:
    f = tmp_path / "unsorted.json"
    data = {
        "response": {
            "data": [
                {"period": "2024-01-01T02", "respondent": "ERCO", "type": "D", "value": 33000},
                {"period": "2024-01-01T01", "respondent": "ERCO", "type": "D", "value": 34000},
                {"period": "2024-01-01T00", "respondent": "ERCO", "type": "D", "value": 35000},
            ]
        }
    }
    f.write_text(json.dumps(data), encoding="utf-8")
    summary = audit_local_eia_json_files([f])
    assert summary.status == "FAIL"
    assert summary.is_chronological is False
    assert any("not in strictly ascending chronological order" in err for err in summary.errors)


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
    payload[2]["hourly"]["temperature_2m"][30] = None

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = query_ecmwf_ifs_run_audit(d)

    assert res.available is False
    assert res.target_window_nulls == 1
    assert any("temperature_2m' has 1 nulls in target window" in r for r in res.failure_reasons)


def test_query_ecmwf_ifs_run_audit_malformed_json_bytes() -> None:
    d = date(2026, 7, 1)
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"{not-valid-json-syntax"
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = query_ecmwf_ifs_run_audit(d)

    assert res.available is False
    assert any("Malformed JSON response" in r for r in res.failure_reasons)


def test_query_ecmwf_ifs_run_audit_incorrect_location_coordinates() -> None:
    d = date(2026, 7, 1)
    payload = _make_valid_locations_payload(d)
    # Move Dallas coordinates well outside 0.20 deg tolerance
    payload[0]["latitude"] = 35.0
    payload[0]["longitude"] = -90.0

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = query_ecmwf_ifs_run_audit(d)

    assert res.available is False
    assert any("exceed snapping tolerance" in r for r in res.failure_reasons)


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


def test_get_configured_eia_api_key_from_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("# Test Comment\nEIA_API_KEY=test_file_key_abc123\n", encoding="utf-8")
    with patch.dict("os.environ", {}, clear=True):
        key = get_configured_eia_api_key(workspace_dir=tmp_path)
        assert key == "test_file_key_abc123"


def test_check_eia_api_coverage_missing_key(tmp_path: Path) -> None:
    with patch.dict("os.environ", {}, clear=True):
        res = check_eia_api_coverage(workspace_dir=tmp_path)
        assert res.api_key_configured is False
        assert res.status == "BLOCKED"
        assert "EIA_API_KEY environment variable is not configured" in res.message


def test_check_eia_api_coverage_success_mocked(tmp_path: Path) -> None:
    mock_resp = MagicMock()
    payload = _make_mock_eia_payload(2208)
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch.dict("os.environ", {"EIA_API_KEY": "test_key_abc"}):
        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = check_eia_api_coverage(workspace_dir=tmp_path)
            assert res.api_key_configured is True
            assert res.status == "GO"
            assert res.total_reported == 2208
            assert res.rows_returned == 2208
            assert res.timestamps_valid is True
            assert res.data_integrity_valid is True
            assert len(res.validation_errors) == 0


def test_check_eia_api_coverage_row_count_mismatch(tmp_path: Path) -> None:
    mock_resp = MagicMock()
    payload = _make_mock_eia_payload(2208)
    payload["response"]["data"] = payload["response"]["data"][:100]  # only 100 rows returned
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch.dict("os.environ", {"EIA_API_KEY": "test_key_abc"}):
        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = check_eia_api_coverage(workspace_dir=tmp_path)
            assert res.status == "ERROR"
            assert res.rows_returned == 100
            assert any("does not match" in err for err in res.validation_errors)


def test_check_eia_api_coverage_duplicate_timestamps(tmp_path: Path) -> None:
    mock_resp = MagicMock()
    payload = _make_mock_eia_payload(2208)
    # Inject a duplicate timestamp
    payload["response"]["data"][1]["period"] = payload["response"]["data"][0]["period"]
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch.dict("os.environ", {"EIA_API_KEY": "test_key_abc"}):
        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = check_eia_api_coverage(workspace_dir=tmp_path)
            assert res.status == "ERROR"
            assert res.duplicate_timestamps_count > 0
            assert any("duplicate timestamps" in err for err in res.validation_errors)


def test_check_eia_api_coverage_null_demand_value(tmp_path: Path) -> None:
    mock_resp = MagicMock()
    payload = _make_mock_eia_payload(2208)
    payload["response"]["data"][5]["value"] = None
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch.dict("os.environ", {"EIA_API_KEY": "test_key_abc"}):
        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = check_eia_api_coverage(workspace_dir=tmp_path)
            assert res.status == "ERROR"
            assert res.null_values_count == 1
            assert any("null demand values" in err for err in res.validation_errors)


def test_check_eia_api_coverage_invalid_respondent_and_type(tmp_path: Path) -> None:
    mock_resp = MagicMock()
    payload = _make_mock_eia_payload(2208)
    payload["response"]["data"][0]["respondent"] = "MISO"
    payload["response"]["data"][0]["type"] = "NG"
    mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch.dict("os.environ", {"EIA_API_KEY": "test_key_abc"}):
        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = check_eia_api_coverage(workspace_dir=tmp_path)
            assert res.status == "ERROR"
            assert any("invalid respondent" in err for err in res.validation_errors)
            assert any("invalid type" in err for err in res.validation_errors)


def test_check_eia_api_coverage_http_error_mocked(tmp_path: Path) -> None:
    with patch.dict("os.environ", {"EIA_API_KEY": "invalid_key"}):
        err = urllib.error.HTTPError("https://api.eia.gov", 403, "Forbidden", {}, None)
        with patch("urllib.request.urlopen", side_effect=err):
            res = check_eia_api_coverage(workspace_dir=tmp_path)
            assert res.api_key_configured is True
            assert res.status == "BLOCKED"
            assert "HTTP 403" in res.message


def test_audit_local_eia_truncated_start_coverage(tmp_path: Path) -> None:
    f = tmp_path / "truncated_start.json"
    data = {
        "response": {
            "data": [
                {"period": "2024-01-02T00", "respondent": "ERCO", "type": "D", "value": 35000},
                {"period": "2024-01-02T01", "respondent": "ERCO", "type": "D", "value": 34000},
            ]
        }
    }
    f.write_text(json.dumps(data), encoding="utf-8")
    summary = audit_local_eia_json_files(
        [f],
        expected_start_period=LOCAL_DEMAND_EXPECTED_START,
        expected_end_period=LOCAL_DEMAND_EXPECTED_END,
    )
    assert summary.status == "FAIL"
    assert any("missing start periods" in err for err in summary.errors)


def test_audit_local_eia_truncated_end_coverage(tmp_path: Path) -> None:
    f = tmp_path / "truncated_end.json"
    data = {
        "response": {
            "data": [
                {"period": "2024-01-01T00", "respondent": "ERCO", "type": "D", "value": 35000},
                {"period": "2024-01-02T00", "respondent": "ERCO", "type": "D", "value": 34000},
            ]
        }
    }
    f.write_text(json.dumps(data), encoding="utf-8")
    summary = audit_local_eia_json_files(
        [f],
        expected_start_period=LOCAL_DEMAND_EXPECTED_START,
        expected_end_period=LOCAL_DEMAND_EXPECTED_END,
    )
    assert summary.status == "FAIL"
    assert any("missing end periods" in err for err in summary.errors)


def test_audit_local_eia_malformed_json_file(tmp_path: Path) -> None:
    f = tmp_path / "broken.json"
    f.write_bytes(b"{not-valid-json-syntax")
    summary = audit_local_eia_json_files([f])
    assert summary.status == "FAIL"
    assert any("malformed JSON" in err for err in summary.errors)


def test_audit_local_eia_malformed_schema_structures(tmp_path: Path) -> None:
    # 1. Top-level is a list
    f1 = tmp_path / "list_top.json"
    f1.write_text(json.dumps([{"period": "2024-01-01T00"}]), encoding="utf-8")
    s1 = audit_local_eia_json_files([f1])
    assert s1.status == "FAIL"
    assert any("top-level JSON is not an object" in err for err in s1.errors)

    # 2. Missing response object
    f2 = tmp_path / "missing_resp.json"
    f2.write_text(json.dumps({"wrong_key": {}}), encoding="utf-8")
    s2 = audit_local_eia_json_files([f2])
    assert s2.status == "FAIL"
    assert any("missing valid 'response' object" in err for err in s2.errors)

    # 3. Data is not a list
    f3 = tmp_path / "data_not_list.json"
    f3.write_text(json.dumps({"response": {"data": "not_a_list"}}), encoding="utf-8")
    s3 = audit_local_eia_json_files([f3])
    assert s3.status == "FAIL"
    assert any("'data' field is not a list" in err for err in s3.errors)

    # 4. Record is not a dictionary
    f4 = tmp_path / "bad_record.json"
    f4.write_text(json.dumps({"response": {"data": ["scalar_record"]}}), encoding="utf-8")
    s4 = audit_local_eia_json_files([f4])
    assert s4.status == "FAIL"
    assert any("is not an object" in err for err in s4.errors)


def test_check_eia_api_coverage_malformed_top_level(tmp_path: Path) -> None:
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(["unexpected", "list"]).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch.dict("os.environ", {"EIA_API_KEY": "test_key_abc"}):
        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = check_eia_api_coverage(workspace_dir=tmp_path)
            assert res.status == "ERROR"
            assert any(
                "top-level response is not a JSON object" in err for err in res.validation_errors
            )


def test_check_eia_api_coverage_malformed_nested_response(tmp_path: Path) -> None:
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"response": "not-a-dict"}).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch.dict("os.environ", {"EIA_API_KEY": "test_key_abc"}):
        with patch("urllib.request.urlopen", return_value=mock_resp):
            res = check_eia_api_coverage(workspace_dir=tmp_path)
            assert res.status == "ERROR"
            assert any(
                "'response' field is not a JSON object" in err for err in res.validation_errors
            )


def test_query_ecmwf_ifs_run_audit_non_finite_coordinates() -> None:
    d = date(2026, 7, 1)

    # NaN latitude
    payload_nan = _make_valid_locations_payload(d)
    payload_nan[0]["latitude"] = float("nan")
    mock_resp_nan = MagicMock()
    mock_resp_nan.read.return_value = json.dumps(payload_nan).encode("utf-8")
    mock_resp_nan.__enter__.return_value = mock_resp_nan

    with patch("urllib.request.urlopen", return_value=mock_resp_nan):
        res_nan = query_ecmwf_ifs_run_audit(d)
    assert res_nan.available is False
    assert any("non-finite or missing" in r for r in res_nan.failure_reasons)

    # Inf longitude
    payload_inf = _make_valid_locations_payload(d)
    payload_inf[0]["longitude"] = float("inf")
    mock_resp_inf = MagicMock()
    mock_resp_inf.read.return_value = json.dumps(payload_inf).encode("utf-8")
    mock_resp_inf.__enter__.return_value = mock_resp_inf

    with patch("urllib.request.urlopen", return_value=mock_resp_inf):
        res_inf = query_ecmwf_ifs_run_audit(d)
    assert res_inf.available is False
    assert any("non-finite or missing" in r for r in res_inf.failure_reasons)
