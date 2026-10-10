"""Data coverage preflight and verification utilities for PowerTide.

Provides functions to audit:
- Local raw EIA demand and forecast JSON archives
- Reusable EIA v2 API coverage checks with fail-closed key handling
- ECMWF IFS HRES (native 9 km, 'ecmwf_ifs') 00:00 UTC Single Runs via Open-Meteo
- Alignment with the fixed 2026 holdout evaluation window (2,208 hourly intervals)
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

# Operational evaluation window constants
HOLDOUT_TARGET_START: Final[str] = "2026-07-01T01"
HOLDOUT_TARGET_END: Final[str] = "2026-10-01T00"
EXPECTED_HOLDOUT_HOURS: Final[int] = 2_208

# Expected local raw demand archive constants (2024 through 2025)
LOCAL_DEMAND_EXPECTED_START: Final[str] = "2024-01-01T00"
LOCAL_DEMAND_EXPECTED_END: Final[str] = "2025-12-31T23"
LOCAL_DEMAND_EXPECTED_HOURS: Final[int] = 17_544

# Approved weather model identity on Open-Meteo Single Runs API
WEATHER_MODEL_ID: Final[str] = "ecmwf_ifs"

# Proxy coordinates for Texas load centers (with model-grid snapping tolerance)
TEXAS_PROXY_LOCATIONS: Final[tuple[dict[str, Any], ...]] = (
    {"name": "Dallas", "latitude": 32.7767, "longitude": -96.7970},
    {"name": "Houston", "latitude": 29.7604, "longitude": -95.3698},
    {"name": "Austin", "latitude": 30.2672, "longitude": -97.7431},
    {"name": "San Antonio", "latitude": 29.4241, "longitude": -98.4936},
    {"name": "Midland", "latitude": 31.9973, "longitude": -102.0779},
)
LOCATION_COORDINATE_TOLERANCE_DEG: Final[float] = 0.20

# Complete weather variables required by the project plan
WEATHER_VARIABLES: Final[tuple[str, ...]] = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
)


class EiaDatasetSummary(BaseModel):
    """Detailed summary of local raw EIA dataset coverage and integrity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_type: str = Field(..., description="EIA series type: 'D' (demand) or 'DF' (forecast)")
    respondent: str = Field(..., description="EIA reporting entity (e.g. 'ERCO')")
    total_records: int = Field(..., description="Total records combined across files")
    unique_records: int = Field(..., description="Count of distinct record signatures")
    duplicate_records_count: int = Field(..., description="Count of repeated record signatures")
    unique_periods: int = Field(..., description="Count of distinct period timestamps")
    duplicate_periods_count: int = Field(..., description="Count of repeated period timestamps")
    null_value_count: int = Field(..., description="Count of records with missing values")
    invalid_value_count: int = Field(
        ..., description="Count of records with non-finite/negative values"
    )
    min_period: str = Field(..., description="Earliest observed timestamp (e.g. 2024-01-01T00)")
    max_period: str = Field(..., description="Latest observed timestamp (e.g. 2025-12-31T23)")
    is_chronological: bool = Field(
        ..., description="True if periods appear in strictly ascending chronological order"
    )
    missing_periods_count: int = Field(
        ..., description="Gaps in the hourly sequence between min and max period"
    )
    missing_period_samples: list[str] = Field(
        default_factory=list, description="Sample of missing hourly timestamps"
    )
    holdout_hours_present: int = Field(
        ..., description="Count of records matching the 2026 holdout window"
    )
    holdout_coverage_ratio: float = Field(
        ..., description="Fraction of required holdout hours present (0.0 to 1.0)"
    )
    status: str = Field(..., description="'PASS' if complete and gap-free, else 'FAIL'")
    errors: list[str] = Field(default_factory=list, description="List of validation failure flags")


class WeatherRunAuditResult(BaseModel):
    """Fail-closed audit result for an ECMWF IFS HRES 00:00 UTC model run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_date: str = Field(..., description="Run initialization date (YYYY-MM-DD)")
    model_id: str = Field(default=WEATHER_MODEL_ID, description="Open-Meteo model parameter")
    available: bool = Field(..., description="True only if run passes all fail-closed criteria")
    locations_count: int = Field(..., description="Number of valid locations returned")
    lead_hours_count: int = Field(..., description="Number of forecast hours in horizon")
    target_intervals_valid: bool = Field(
        ..., description="True if target timestamps strictly match day D+1 EIA intervals"
    )
    target_window_nulls: int = Field(
        ..., description="Total null count in the D+1 target window across all stations/vars"
    )
    failure_reasons: list[str] = Field(
        default_factory=list, description="Reasons explaining any validation failure"
    )


class EiaApiAuditResult(BaseModel):
    """Audit result for live EIA v2 API coverage check verifying actual data rows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key_configured: bool = Field(
        ..., description="Whether EIA_API_KEY was found in environment or .env"
    )
    status: str = Field(..., description="'GO', 'BLOCKED', or 'ERROR'")
    total_reported: int = Field(..., description="Total matching records reported by the API")
    rows_returned: int = Field(..., description="Actual data rows parsed from response")
    start_period: str | None = Field(default=None, description="Queried start period")
    end_period: str | None = Field(default=None, description="Queried end period")
    timestamps_valid: bool = Field(
        ..., description="True if timestamps strictly match expected sequence"
    )
    data_integrity_valid: bool = Field(
        ..., description="True if respondent, type, and numeric values are valid"
    )
    null_values_count: int = Field(..., description="Count of rows with null values")
    invalid_values_count: int = Field(..., description="Count of rows with invalid numeric values")
    duplicate_timestamps_count: int = Field(..., description="Count of duplicate timestamps")
    missing_timestamps_count: int = Field(..., description="Count of missing hourly intervals")
    message: str = Field(..., description="Status summary or blocker explanation")
    validation_errors: list[str] = Field(
        default_factory=list, description="Detailed validation error list"
    )


def compute_expected_weather_target_timestamps(run_date: date) -> list[str]:
    """Generate the 24 consecutive target hourly timestamps for day D+1.

    Following EIA hour-ending convention:
    For initialization date D, day D+1 intervals span from (D+1)T01:00 to (D+2)T00:00.
    """
    day_d1 = run_date + timedelta(days=1)
    timestamps: list[str] = []
    for h in range(1, 24):
        timestamps.append(f"{day_d1.isoformat()}T{h:02d}:00")
    day_d2 = run_date + timedelta(days=2)
    timestamps.append(f"{day_d2.isoformat()}T00:00")
    return timestamps


def compute_expected_eia_hourly_sequence(start_period: str, end_period: str) -> list[str]:
    """Generate the consecutive hourly timestamps from start_period through end_period inclusive."""
    start_dt = datetime.strptime(start_period, "%Y-%m-%dT%H")
    end_dt = datetime.strptime(end_period, "%Y-%m-%dT%H")
    timestamps: list[str] = []
    curr = start_dt
    while curr <= end_dt:
        timestamps.append(curr.strftime("%Y-%m-%dT%H"))
        curr += timedelta(hours=1)
    return timestamps


def get_configured_eia_api_key(
    workspace_dir: Path | None = None,
    env_path: Path | None = None,
) -> str:
    """Retrieve EIA_API_KEY from os.environ or a local uncommitted .env file.

    Never logs, prints, or exposes the raw key.
    """
    key = os.environ.get("EIA_API_KEY", "").strip()
    if key:
        return key

    target_env = (
        env_path
        if env_path is not None
        else ((workspace_dir / ".env") if workspace_dir else Path(".env"))
    )
    if target_env.exists():
        try:
            with open(target_env, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == "EIA_API_KEY":
                            candidate = v.strip().strip("'\"")
                            if candidate:
                                return candidate
        except Exception:
            return ""

    return ""


def audit_local_eia_json_files(
    json_paths: list[Path],
    expected_type: str = "D",
    expected_respondent: str = "ERCO",
    expected_start_period: str | None = None,
    expected_end_period: str | None = None,
) -> EiaDatasetSummary:
    """Inspect and validate local raw EIA JSON files without mutating them."""
    if not json_paths:
        raise ValueError("json_paths list cannot be empty")

    def _path_sort_key(p: Path) -> int:
        parts = p.stem.split("_")
        if len(parts) > 1 and parts[-1].isdigit():
            return int(parts[-1])
        return 0

    sorted_paths = sorted(json_paths, key=_path_sort_key)
    records: list[dict[str, Any]] = []
    errors: list[str] = []

    for path in sorted_paths:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"File {path.name} contains malformed JSON: {e}")
            continue
        except (UnicodeDecodeError, OSError) as e:
            errors.append(f"File {path.name} could not be read: {e}")
            continue

        if not isinstance(data, dict):
            errors.append(
                f"File {path.name} top-level JSON is not an object (got {type(data).__name__})"
            )
            continue

        resp_obj = data.get("response")
        if not isinstance(resp_obj, dict):
            errors.append(
                f"File {path.name} missing valid 'response' object (got {type(resp_obj).__name__})"
            )
            continue

        batch = resp_obj.get("data")
        if not isinstance(batch, list):
            errors.append(
                f"File {path.name} 'data' field is not a list (got {type(batch).__name__})"
            )
            continue

        for rec_idx, r in enumerate(batch):
            if not isinstance(r, dict):
                errors.append(
                    f"File {path.name} record {rec_idx} is not an object (got {type(r).__name__})"
                )
            else:
                records.append(r)

    if not records:
        if not errors:
            errors.append("No records found in provided files")
        return EiaDatasetSummary(
            dataset_type="UNKNOWN",
            respondent="UNKNOWN",
            total_records=0,
            unique_records=0,
            duplicate_records_count=0,
            unique_periods=0,
            duplicate_periods_count=0,
            null_value_count=0,
            invalid_value_count=0,
            min_period="",
            max_period="",
            is_chronological=False,
            missing_periods_count=0,
            missing_period_samples=[],
            holdout_hours_present=0,
            holdout_coverage_ratio=0.0,
            status="FAIL",
            errors=errors,
        )

    # 1. Period presence and format validation on every record
    periods: list[str] = []
    missing_period_field_count = 0

    for idx, r in enumerate(records):
        p = r.get("period")
        if not p or not isinstance(p, str):
            missing_period_field_count += 1
            continue
        try:
            datetime.strptime(p, "%Y-%m-%dT%H")
            periods.append(p)
        except (ValueError, TypeError):
            errors.append(f"Invalid timestamp format in record {idx}: {p}")

    if missing_period_field_count > 0:
        errors.append(f"{missing_period_field_count} records are missing a valid 'period' field")

    # 2. Respondent and Type validation
    types_found = {r.get("type") for r in records}
    respondents_found = {r.get("respondent") for r in records}

    if types_found != {expected_type}:
        errors.append(f"Unexpected types found: {types_found} (expected: '{expected_type}')")
    if respondents_found != {expected_respondent}:
        errors.append(
            f"Unexpected respondents found: {respondents_found} (expected: '{expected_respondent}')"
        )

    # 3. Numeric value validation
    null_value_count = sum(1 for r in records if r.get("value") is None)
    invalid_value_count = 0

    for r in records:
        val = r.get("value")
        if val is None:
            continue
        try:
            num = float(val)
            if not math.isfinite(num) or num < 0:
                invalid_value_count += 1
        except (ValueError, TypeError):
            invalid_value_count += 1

    if null_value_count > 0:
        errors.append(f"Dataset contains {null_value_count} null demand values")
    if invalid_value_count > 0:
        errors.append(f"Dataset contains {invalid_value_count} invalid or negative numeric values")

    # 4. Record signatures and duplicate detection
    signatures = [
        (r.get("period"), r.get("respondent"), r.get("type"), r.get("value")) for r in records
    ]
    sig_counts = Counter(signatures)
    duplicate_records_count = sum(c - 1 for c in sig_counts.values() if c > 1)
    if duplicate_records_count > 0:
        errors.append(f"Dataset contains {duplicate_records_count} duplicate identical records")

    period_counts = Counter(periods)
    duplicate_periods_count = sum(c - 1 for c in period_counts.values() if c > 1)
    if duplicate_periods_count > 0:
        errors.append(
            f"Dataset contains {duplicate_periods_count} duplicate hourly period timestamps"
        )

    # 5. Strictly ascending chronological order (no equal timestamps allowed)
    is_chronological = False
    if periods and len(periods) == len(records):
        is_chronological = all(periods[i] < periods[i + 1] for i in range(len(periods) - 1))
        if not is_chronological:
            errors.append("Records are not in strictly ascending chronological order")

    # 6. Missing periods and span coverage check
    min_p = min(periods) if periods else ""
    max_p = max(periods) if periods else ""
    missing_period_samples: list[str] = []
    missing_periods_count = 0

    target_start = expected_start_period if expected_start_period else min_p
    target_end = expected_end_period if expected_end_period else max_p

    if expected_start_period and min_p and min_p > expected_start_period:
        errors.append(
            f"Archive is missing start periods: expected {expected_start_period}, "
            f"observed earliest {min_p}"
        )
    if expected_end_period and max_p and max_p < expected_end_period:
        errors.append(
            f"Archive is missing end periods: expected {expected_end_period}, "
            f"observed latest {max_p}"
        )

    if target_start and target_end and periods:
        try:
            expected_periods = set(compute_expected_eia_hourly_sequence(target_start, target_end))
            actual_periods_set = set(periods)
            missing_set = expected_periods - actual_periods_set
            missing_periods_count = len(missing_set)
            missing_period_samples = sorted(missing_set)[:5]
            if missing_periods_count > 0:
                errors.append(
                    f"Detected {missing_periods_count} missing hourly intervals "
                    f"in span {target_start} through {target_end}"
                )
            if expected_start_period and expected_end_period:
                expected_total = len(expected_periods)
                if len(actual_periods_set) != expected_total:
                    errors.append(
                        f"Expected {expected_total} unique hourly records for {target_start} "
                        f"through {target_end}, but found {len(actual_periods_set)}"
                    )
        except Exception as e:
            errors.append(f"Failed to generate expected period span: {e}")

    # 7. Check holdout overlap
    holdout_hours_present = sum(
        1 for p in set(periods) if HOLDOUT_TARGET_START <= p <= HOLDOUT_TARGET_END
    )
    holdout_coverage_ratio = holdout_hours_present / EXPECTED_HOLDOUT_HOURS

    status = "PASS" if not errors else "FAIL"

    return EiaDatasetSummary(
        dataset_type=list(types_found)[0] if len(types_found) == 1 else "MIXED",
        respondent=list(respondents_found)[0] if len(respondents_found) == 1 else "MIXED",
        total_records=len(records),
        unique_records=len(sig_counts),
        duplicate_records_count=duplicate_records_count,
        unique_periods=len(period_counts),
        duplicate_periods_count=duplicate_periods_count,
        null_value_count=null_value_count,
        invalid_value_count=invalid_value_count,
        min_period=min_p,
        max_period=max_p,
        is_chronological=is_chronological,
        missing_periods_count=missing_periods_count,
        missing_period_samples=missing_period_samples,
        holdout_hours_present=holdout_hours_present,
        holdout_coverage_ratio=holdout_coverage_ratio,
        status=status,
        errors=errors,
    )


def query_ecmwf_ifs_run_audit(
    run_date: date,
    timeout_seconds: int = 20,
) -> WeatherRunAuditResult:
    """Audit a specific ECMWF IFS HRES 00:00 UTC run, failing closed on any irregularity."""
    failure_reasons: list[str] = []
    lats = ",".join(str(loc["latitude"]) for loc in TEXAS_PROXY_LOCATIONS)
    lons = ",".join(str(loc["longitude"]) for loc in TEXAS_PROXY_LOCATIONS)
    vars_param = ",".join(WEATHER_VARIABLES)
    d_str = run_date.isoformat()

    params = {
        "latitude": lats,
        "longitude": lons,
        "models": WEATHER_MODEL_ID,
        "run": f"{d_str}T00:00",
        "hourly": vars_param,
    }
    url = f"https://single-runs-api.open-meteo.com/v1/forecast?{urllib.parse.urlencode(params)}"

    req = urllib.request.Request(url, headers={"User-Agent": "PowerTide-Preflight/0.1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw_bytes = resp.read()
    except urllib.error.HTTPError as e:
        failure_reasons.append(f"HTTPError {e.code}: {e.reason}")
        return WeatherRunAuditResult(
            run_date=d_str,
            available=False,
            locations_count=0,
            lead_hours_count=0,
            target_intervals_valid=False,
            target_window_nulls=-1,
            failure_reasons=failure_reasons,
        )
    except (urllib.error.URLError, TimeoutError) as e:
        failure_reasons.append(f"Network error: {e}")
        return WeatherRunAuditResult(
            run_date=d_str,
            available=False,
            locations_count=0,
            lead_hours_count=0,
            target_intervals_valid=False,
            target_window_nulls=-1,
            failure_reasons=failure_reasons,
        )

    try:
        data = json.loads(raw_bytes.decode())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        failure_reasons.append(f"Malformed JSON response: {e}")
        return WeatherRunAuditResult(
            run_date=d_str,
            available=False,
            locations_count=0,
            lead_hours_count=0,
            target_intervals_valid=False,
            target_window_nulls=-1,
            failure_reasons=failure_reasons,
        )

    # 1. Structure and location count check
    if not isinstance(data, list):
        failure_reasons.append("Response is not a list of location objects")
        return WeatherRunAuditResult(
            run_date=d_str,
            available=False,
            locations_count=0,
            lead_hours_count=0,
            target_intervals_valid=False,
            target_window_nulls=-1,
            failure_reasons=failure_reasons,
        )

    if len(data) != len(TEXAS_PROXY_LOCATIONS):
        failure_reasons.append(
            f"Expected {len(TEXAS_PROXY_LOCATIONS)} locations, received {len(data)}"
        )

    expected_targets = compute_expected_weather_target_timestamps(run_date)
    target_nulls_count = 0
    target_intervals_valid = True
    total_leads = 0

    # 2. Location-level checks with coordinate snapping verification
    for idx, loc in enumerate(data):
        if not isinstance(loc, dict):
            failure_reasons.append(f"Location {idx} payload is not a valid dictionary")
            target_intervals_valid = False
            continue

        proxy = TEXAS_PROXY_LOCATIONS[idx] if idx < len(TEXAS_PROXY_LOCATIONS) else None
        loc_name = proxy["name"] if proxy else f"Loc_{idx}"

        ret_lat = loc.get("latitude")
        ret_lon = loc.get("longitude")
        if (
            not isinstance(ret_lat, (int, float))
            or not isinstance(ret_lon, (int, float))
            or not math.isfinite(ret_lat)
            or not math.isfinite(ret_lon)
        ):
            failure_reasons.append(
                f"Coordinates for {loc_name} are non-finite or missing "
                f"(lat: {ret_lat}, lon: {ret_lon})"
            )
            target_intervals_valid = False
        elif proxy:
            # Check model-grid coordinate snapping tolerance
            lat_diff = abs(ret_lat - proxy["latitude"])
            lon_diff = abs(ret_lon - proxy["longitude"])
            if (
                lat_diff > LOCATION_COORDINATE_TOLERANCE_DEG
                or lon_diff > LOCATION_COORDINATE_TOLERANCE_DEG
            ):
                failure_reasons.append(
                    f"Coordinates for {loc_name} ({ret_lat}, {ret_lon}) exceed snapping tolerance "
                    f"from requested proxy ({proxy['latitude']}, {proxy['longitude']})"
                )
                target_intervals_valid = False

        hourly = loc.get("hourly")
        if not isinstance(hourly, dict):
            failure_reasons.append(f"Missing or malformed 'hourly' object for {loc_name}")
            target_intervals_valid = False
            continue

        times = hourly.get("time")
        if not isinstance(times, list) or len(times) < 48:
            horizon_len = len(times) if isinstance(times, list) else 0
            failure_reasons.append(f"Horizon too short for {loc_name}: {horizon_len} hours")
            target_intervals_valid = False
            continue

        total_leads = max(total_leads, len(times))

        # Check variable presence
        for v in WEATHER_VARIABLES:
            if v not in hourly:
                failure_reasons.append(f"Missing required variable '{v}' for {loc_name}")
                target_intervals_valid = False

        # Target window interval alignment
        target_times_slice = times[25:49]
        if target_times_slice != expected_targets:
            failure_reasons.append(
                f"Target timestamps mismatch for {loc_name}: "
                f"got {target_times_slice[:2]}..{target_times_slice[-1:]}, "
                f"expected {expected_targets[:2]}..{expected_targets[-1:]}"
            )
            target_intervals_valid = False

        # Value null and finite numeric checks on target intervals
        for v in WEATHER_VARIABLES:
            vals = hourly.get(v, [])
            if not isinstance(vals, list) or len(vals) < 49:
                failure_reasons.append(f"Variable '{v}' array too short for {loc_name}")
                target_intervals_valid = False
                continue
            target_vals_slice = vals[25:49]
            nulls_in_var = 0
            for val in target_vals_slice:
                if val is None or not isinstance(val, (int, float)) or not math.isfinite(val):
                    nulls_in_var += 1
            if nulls_in_var > 0:
                target_nulls_count += nulls_in_var
                failure_reasons.append(
                    f"Variable '{v}' has {nulls_in_var} nulls in target window for {loc_name}"
                )

    is_available = len(failure_reasons) == 0 and target_intervals_valid and target_nulls_count == 0

    return WeatherRunAuditResult(
        run_date=d_str,
        model_id=WEATHER_MODEL_ID,
        available=is_available,
        locations_count=len(data),
        lead_hours_count=total_leads,
        target_intervals_valid=target_intervals_valid,
        target_window_nulls=target_nulls_count,
        failure_reasons=failure_reasons,
    )


def check_eia_api_coverage(
    start_period: str = HOLDOUT_TARGET_START,
    end_period: str = HOLDOUT_TARGET_END,
    timeout_seconds: int = 45,
    workspace_dir: Path | None = None,
    env_path: Path | None = None,
) -> EiaApiAuditResult:
    """Retrieve and verify the full holdout dataset from EIA API v2 row-by-row.

    Never logs, prints, or exposes the API key.
    """
    api_key = get_configured_eia_api_key(workspace_dir=workspace_dir, env_path=env_path)
    if not api_key:
        msg = (
            "EIA_API_KEY environment variable is not configured. "
            "Demand data for the 2026 holdout cannot be queried."
        )
        return EiaApiAuditResult(
            api_key_configured=False,
            status="BLOCKED",
            total_reported=0,
            rows_returned=0,
            start_period=start_period,
            end_period=end_period,
            timestamps_valid=False,
            data_integrity_valid=False,
            null_values_count=0,
            invalid_values_count=0,
            duplicate_timestamps_count=0,
            missing_timestamps_count=EXPECTED_HOLDOUT_HOURS,
            message=msg,
            validation_errors=[msg],
        )

    expected_sequence = compute_expected_eia_hourly_sequence(start_period, end_period)
    expected_count = len(expected_sequence)

    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": "ERCO",
        "facets[type][]": "D",
        "start": start_period,
        "end": end_period,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": str(expected_count),
    }

    url = (
        f"https://api.eia.gov/v2/electricity/rto/region-data/data/?{urllib.parse.urlencode(params)}"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "PowerTide-Preflight/0.1.0"})

    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            raw_bytes = resp.read()
    except urllib.error.HTTPError as e:
        safe_msg = f"EIA API request failed with HTTP {e.code}: {e.reason}"
        status = "BLOCKED" if e.code in (401, 403) else "ERROR"
        return EiaApiAuditResult(
            api_key_configured=True,
            status=status,
            total_reported=0,
            rows_returned=0,
            start_period=start_period,
            end_period=end_period,
            timestamps_valid=False,
            data_integrity_valid=False,
            null_values_count=0,
            invalid_values_count=0,
            duplicate_timestamps_count=0,
            missing_timestamps_count=expected_count,
            message=safe_msg,
            validation_errors=[safe_msg],
        )
    except Exception as e:
        safe_msg = f"EIA API request encountered connection error: {type(e).__name__}"
        return EiaApiAuditResult(
            api_key_configured=True,
            status="ERROR",
            total_reported=0,
            rows_returned=0,
            start_period=start_period,
            end_period=end_period,
            timestamps_valid=False,
            data_integrity_valid=False,
            null_values_count=0,
            invalid_values_count=0,
            duplicate_timestamps_count=0,
            missing_timestamps_count=expected_count,
            message=safe_msg,
            validation_errors=[safe_msg],
        )

    try:
        data = json.loads(raw_bytes.decode())
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        safe_msg = f"EIA API response could not be parsed as JSON: {e}"
        return EiaApiAuditResult(
            api_key_configured=True,
            status="ERROR",
            total_reported=0,
            rows_returned=0,
            start_period=start_period,
            end_period=end_period,
            timestamps_valid=False,
            data_integrity_valid=False,
            null_values_count=0,
            invalid_values_count=0,
            duplicate_timestamps_count=0,
            missing_timestamps_count=expected_count,
            message=safe_msg,
            validation_errors=[safe_msg],
        )

    if not isinstance(data, dict):
        safe_msg = f"EIA API top-level response is not a JSON object (got {type(data).__name__})"
        return EiaApiAuditResult(
            api_key_configured=True,
            status="ERROR",
            total_reported=0,
            rows_returned=0,
            start_period=start_period,
            end_period=end_period,
            timestamps_valid=False,
            data_integrity_valid=False,
            null_values_count=0,
            invalid_values_count=0,
            duplicate_timestamps_count=0,
            missing_timestamps_count=expected_count,
            message=safe_msg,
            validation_errors=[safe_msg],
        )

    response_block = data.get("response")
    if not isinstance(response_block, dict):
        safe_msg = (
            f"EIA API 'response' field is not a JSON object (got {type(response_block).__name__})"
        )
        return EiaApiAuditResult(
            api_key_configured=True,
            status="ERROR",
            total_reported=0,
            rows_returned=0,
            start_period=start_period,
            end_period=end_period,
            timestamps_valid=False,
            data_integrity_valid=False,
            null_values_count=0,
            invalid_values_count=0,
            duplicate_timestamps_count=0,
            missing_timestamps_count=expected_count,
            message=safe_msg,
            validation_errors=[safe_msg],
        )

    validation_errors: list[str] = []
    raw_total = response_block.get("total", 0)
    try:
        total = int(raw_total)
    except (ValueError, TypeError):
        total = 0
        validation_errors.append(f"Invalid 'total' field in response: {raw_total}")

    rows = response_block.get("data")
    if not isinstance(rows, list):
        rows = []
        validation_errors.append("Response 'data' field is not a list")

    rows_returned = len(rows)

    if total != rows_returned:
        validation_errors.append(
            f"Response reported total ({total}) does not match returned rows ({rows_returned})"
        )

    if rows_returned != expected_count:
        validation_errors.append(
            f"Returned rows ({rows_returned}) does not match "
            f"expected holdout count ({expected_count})"
        )

    # Validate row periods and sequence
    actual_periods = [r.get("period") for r in rows if isinstance(r, dict)]
    period_counts = Counter(actual_periods)
    duplicate_timestamps = sum(c - 1 for c in period_counts.values() if c > 1)
    if duplicate_timestamps > 0:
        validation_errors.append(
            f"Detected {duplicate_timestamps} duplicate timestamps in API response"
        )

    missing_periods = set(expected_sequence) - set(actual_periods)
    missing_count = len(missing_periods)
    if missing_count > 0:
        validation_errors.append(
            f"Detected {missing_count} missing hourly timestamps in API response"
        )

    timestamps_valid = (
        actual_periods == expected_sequence and duplicate_timestamps == 0 and missing_count == 0
    )
    if not timestamps_valid and not missing_count and not duplicate_timestamps:
        validation_errors.append(
            "Returned periods are out of order or do not strictly match expected sequence"
        )

    # Validate row-level schema and numeric values
    null_values_count = 0
    invalid_values_count = 0
    invalid_type_count = 0
    invalid_respondent_count = 0

    for idx, r in enumerate(rows):
        if not isinstance(r, dict):
            validation_errors.append(f"Row {idx} is not a valid dictionary")
            continue

        if r.get("respondent") != "ERCO":
            invalid_respondent_count += 1
        if r.get("type") != "D":
            invalid_type_count += 1

        val = r.get("value")
        if val is None:
            null_values_count += 1
        else:
            try:
                num = float(val)
                if not math.isfinite(num) or num < 0:
                    invalid_values_count += 1
            except (ValueError, TypeError):
                invalid_values_count += 1

    if invalid_respondent_count > 0:
        validation_errors.append(
            f"{invalid_respondent_count} rows have invalid respondent (expected 'ERCO')"
        )
    if invalid_type_count > 0:
        validation_errors.append(f"{invalid_type_count} rows have invalid type (expected 'D')")
    if null_values_count > 0:
        validation_errors.append(f"{null_values_count} rows contain null demand values")
    if invalid_values_count > 0:
        validation_errors.append(
            f"{invalid_values_count} rows contain non-finite or negative demand values"
        )

    data_integrity_valid = (
        invalid_respondent_count == 0
        and invalid_type_count == 0
        and null_values_count == 0
        and invalid_values_count == 0
    )

    status = (
        "GO"
        if len(validation_errors) == 0 and timestamps_valid and data_integrity_valid
        else "ERROR"
    )
    message = (
        f"EIA API returned and verified all {rows_returned} actual holdout data rows "
        "with 100% integrity."
        if status == "GO"
        else f"EIA API verification failed: {'; '.join(validation_errors[:2])}"
    )

    return EiaApiAuditResult(
        api_key_configured=True,
        status=status,
        total_reported=total,
        rows_returned=rows_returned,
        start_period=start_period,
        end_period=end_period,
        timestamps_valid=timestamps_valid,
        data_integrity_valid=data_integrity_valid,
        null_values_count=null_values_count,
        invalid_values_count=invalid_values_count,
        duplicate_timestamps_count=duplicate_timestamps,
        missing_timestamps_count=missing_count,
        message=message,
        validation_errors=validation_errors,
    )


def generate_evaluation_date_range() -> list[date]:
    """Generate the 92 initialization dates from 2026-06-30 through 2026-09-29."""
    start = date(2026, 6, 30)
    end = date(2026, 9, 29)
    dates: list[date] = []
    curr = start
    while curr <= end:
        dates.append(curr)
        curr += timedelta(days=1)
    return dates


def run_complete_preflight_audit(
    raw_dir: Path | None = None,
    output_report_path: Path | None = None,
) -> dict[str, Any]:
    """Execute complete reproducible preflight audit covering local data, EIA API, and weather."""
    base_dir = raw_dir if raw_dir else Path("data/raw")
    demand_files = sorted(base_dir.glob("demand_*.json"))
    forecast_files = sorted(base_dir.glob("forecast_*.json"))

    # 1. Local archive audit
    local_demand = audit_local_eia_json_files(
        demand_files,
        expected_type="D",
        expected_respondent="ERCO",
        expected_start_period=LOCAL_DEMAND_EXPECTED_START,
        expected_end_period=LOCAL_DEMAND_EXPECTED_END,
    )
    local_forecast = audit_local_eia_json_files(
        forecast_files,
        expected_type="DF",
        expected_respondent="ERCO",
    )

    # 2. Complete EIA API audit
    eia_api_audit = check_eia_api_coverage()

    # 3. Weather audits for all 92 runs
    eval_dates = generate_evaluation_date_range()
    valid_weather_runs = 0
    failed_weather_runs = 0
    total_target_nulls = 0
    weather_failures: list[dict[str, Any]] = []

    for d in eval_dates:
        res = query_ecmwf_ifs_run_audit(d)
        if res.available:
            valid_weather_runs += 1
        else:
            failed_weather_runs += 1
            weather_failures.append({"run_date": d.isoformat(), "reasons": res.failure_reasons})
        if res.target_window_nulls > 0:
            total_target_nulls += res.target_window_nulls

    overall_status = (
        "GO_FOR_INGESTION"
        if (
            local_demand.status == "PASS"
            and eia_api_audit.status == "GO"
            and valid_weather_runs == len(eval_dates)
            and total_target_nulls == 0
        )
        else "BLOCKED"
    )

    report: dict[str, Any] = {
        "overall_status": overall_status,
        "local_demand": local_demand.model_dump(),
        "local_forecast": local_forecast.model_dump(),
        "eia_api": eia_api_audit.model_dump(),
        "weather": {
            "model_id": WEATHER_MODEL_ID,
            "total_runs": len(eval_dates),
            "valid_runs": valid_weather_runs,
            "failed_runs": failed_weather_runs,
            "total_target_nulls": total_target_nulls,
            "failures_sample": weather_failures[:5],
        },
    }

    if output_report_path:
        output_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    report = run_complete_preflight_audit(
        output_report_path=Path("artifacts/preflight_coverage.json")
    )
    print("=" * 80)
    print("POWER_TIDE REPRODUCIBLE PREFLIGHT AUDIT")
    print("=" * 80)
    print(f"Overall Status: {report['overall_status']}")
    loc_records = report["local_demand"]["total_records"]
    loc_status = report["local_demand"]["status"]
    print(f"Local Demand: {loc_records} rows, status={loc_status}")

    api_rows = report["eia_api"]["rows_returned"]
    api_status = report["eia_api"]["status"]
    print(f"EIA Live API: {api_rows} rows returned, status={api_status}")

    w_valid = report["weather"]["valid_runs"]
    w_tot = report["weather"]["total_runs"]
    w_nulls = report["weather"]["total_target_nulls"]
    print(f"Weather ECMWF: {w_valid} / {w_tot} valid runs, nulls={w_nulls}")
    print("=" * 80)
