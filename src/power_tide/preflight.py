"""Data coverage preflight and verification utilities for PowerTide.

Provides functions to audit:
- Local raw EIA demand and forecast JSON archives
- Reusable EIA v2 API coverage checks with fail-closed key handling
- ECMWF IFS HRES (native 9 km, 'ecmwf_ifs') 00:00 UTC Single Runs via Open-Meteo
- Alignment with the fixed 2026 holdout evaluation window (2,208 hourly intervals)
"""

from __future__ import annotations

import json
import os
import urllib.error
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

# Approved weather model identity on Open-Meteo Single Runs API
WEATHER_MODEL_ID: Final[str] = "ecmwf_ifs"

# Proxy coordinates for Texas load centers
TEXAS_PROXY_LOCATIONS: Final[tuple[dict[str, Any], ...]] = (
    {"name": "Dallas", "latitude": 32.7767, "longitude": -96.7970},
    {"name": "Houston", "latitude": 29.7604, "longitude": -95.3698},
    {"name": "Austin", "latitude": 30.2672, "longitude": -97.7431},
    {"name": "San Antonio", "latitude": 29.4241, "longitude": -98.4936},
    {"name": "Midland", "latitude": 31.9973, "longitude": -102.0779},
)

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
    min_period: str = Field(..., description="Earliest observed timestamp (e.g. 2024-01-01T00)")
    max_period: str = Field(..., description="Latest observed timestamp (e.g. 2025-12-31T23)")
    is_chronological: bool = Field(
        ..., description="True if periods appear in strict ascending order"
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
    """Audit result for live EIA v2 API coverage check."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    api_key_configured: bool = Field(
        ..., description="Whether EIA_API_KEY was found in environment"
    )
    status: str = Field(..., description="'GO', 'BLOCKED', or 'ERROR'")
    total_reported: int = Field(..., description="Total matching records reported by the API")
    start_period: str | None = Field(default=None, description="Queried start period")
    end_period: str | None = Field(default=None, description="Queried end period")
    message: str = Field(..., description="Status summary or blocker explanation")


def compute_expected_weather_target_timestamps(run_date: date) -> list[str]:
    """Generate the 24 consecutive target hourly timestamps for day D+1.

    Following EIA hour-ending convention:
    For initialization date D, day D+1 intervals span from (D+1)T01:00 to (D+2)T00:00.
    """
    day_d1 = run_date + timedelta(days=1)
    timestamps: list[str] = []
    # Hours 1 through 23 of day D+1
    for h in range(1, 24):
        timestamps.append(f"{day_d1.isoformat()}T{h:02d}:00")
    # Hour 24 of day D+1 corresponds to 00:00 of day D+2
    day_d2 = run_date + timedelta(days=2)
    timestamps.append(f"{day_d2.isoformat()}T00:00")
    return timestamps


def audit_local_eia_json_files(
    json_paths: list[Path],
    expected_type: str = "D",
    expected_respondent: str = "ERCO",
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
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        batch = data.get("response", {}).get("data", [])
        records.extend(batch)

    if not records:
        return EiaDatasetSummary(
            dataset_type="UNKNOWN",
            respondent="UNKNOWN",
            total_records=0,
            unique_records=0,
            duplicate_records_count=0,
            unique_periods=0,
            duplicate_periods_count=0,
            null_value_count=0,
            min_period="",
            max_period="",
            is_chronological=False,
            missing_periods_count=0,
            missing_period_samples=[],
            holdout_hours_present=0,
            holdout_coverage_ratio=0.0,
            status="FAIL",
            errors=["No records found in provided files"],
        )

    # 1. Respondent and Type validation
    types_found = {r.get("type") for r in records}
    respondents_found = {r.get("respondent") for r in records}

    if types_found != {expected_type}:
        errors.append(f"Unexpected types found: {types_found} (expected: '{expected_type}')")
    if respondents_found != {expected_respondent}:
        errors.append(
            f"Unexpected respondents found: {respondents_found} (expected: '{expected_respondent}')"
        )

    # 2. Record signatures and duplicates
    signatures = [
        (r.get("period"), r.get("respondent"), r.get("type"), r.get("value")) for r in records
    ]
    sig_counts = Counter(signatures)
    duplicate_records_count = sum(c - 1 for c in sig_counts.values() if c > 1)

    # 3. Periods and timestamp formatting
    periods = [r.get("period") for r in records if "period" in r]
    null_value_count = sum(1 for r in records if r.get("value") is None)

    for p in periods:
        try:
            datetime.strptime(p, "%Y-%m-%dT%H")
        except (ValueError, TypeError):
            errors.append(f"Invalid timestamp format detected: {p}")
            break

    period_counts = Counter(periods)
    duplicate_periods_count = sum(c - 1 for c in period_counts.values() if c > 1)

    # 4. Chronological order
    is_chronological = all(periods[i] <= periods[i + 1] for i in range(len(periods) - 1))
    if not is_chronological:
        errors.append("Records are not in strict ascending chronological order")

    # 5. Missing periods check across the span
    min_p = min(periods)
    max_p = max(periods)
    missing_period_samples: list[str] = []
    missing_periods_count = 0

    try:
        start_dt = datetime.strptime(min_p, "%Y-%m-%dT%H")
        end_dt = datetime.strptime(max_p, "%Y-%m-%dT%H")
        expected_periods: set[str] = set()
        curr = start_dt
        while curr <= end_dt:
            expected_periods.add(curr.strftime("%Y-%m-%dT%H"))
            curr += timedelta(hours=1)

        actual_periods_set = set(periods)
        missing_set = expected_periods - actual_periods_set
        missing_periods_count = len(missing_set)
        missing_period_samples = sorted(missing_set)[:5]
        if missing_periods_count > 0:
            errors.append(f"Detected {missing_periods_count} missing hourly intervals in span")
    except Exception as e:
        errors.append(f"Failed to generate expected period span: {e}")

    # 6. Check holdout overlap
    holdout_hours_present = sum(
        1 for p in set(periods) if HOLDOUT_TARGET_START <= p <= HOLDOUT_TARGET_END
    )
    holdout_coverage_ratio = holdout_hours_present / EXPECTED_HOLDOUT_HOURS

    status = "PASS" if not errors and duplicate_records_count == 0 else "FAIL"

    return EiaDatasetSummary(
        dataset_type=list(types_found)[0] if len(types_found) == 1 else "MIXED",
        respondent=list(respondents_found)[0] if len(respondents_found) == 1 else "MIXED",
        total_records=len(records),
        unique_records=len(sig_counts),
        duplicate_records_count=duplicate_records_count,
        unique_periods=len(period_counts),
        duplicate_periods_count=duplicate_periods_count,
        null_value_count=null_value_count,
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
    timeout_seconds: int = 15,
) -> WeatherRunAuditResult:
    """Audit a specific ECMWF IFS HRES 00:00 UTC run, failing closed on any irregularity."""
    failure_reasons: list[str] = []
    lats = ",".join(str(loc["latitude"]) for loc in TEXAS_PROXY_LOCATIONS)
    lons = ",".join(str(loc["longitude"]) for loc in TEXAS_PROXY_LOCATIONS)
    vars_param = ",".join(WEATHER_VARIABLES)
    d_str = run_date.isoformat()

    url = (
        f"https://single-runs-api.open-meteo.com/v1/forecast?"
        f"latitude={lats}&longitude={lons}&models={WEATHER_MODEL_ID}&"
        f"run={d_str}T00:00&hourly={vars_param}"
    )

    req = urllib.request.Request(url, headers={"User-Agent": "PowerTide-Preflight/0.1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            data = json.loads(resp.read().decode())
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

    # 2. Location-level checks
    for idx, loc in enumerate(data):
        loc_name = (
            TEXAS_PROXY_LOCATIONS[idx]["name"] if idx < len(TEXAS_PROXY_LOCATIONS) else f"Loc_{idx}"
        )
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
        # In ECMWF IFS 00:00 run, day D+1 EIA intervals correspond to indices 25..48
        target_times_slice = times[25:49]
        if target_times_slice != expected_targets:
            failure_reasons.append(
                f"Target timestamps mismatch for {loc_name}: "
                f"got {target_times_slice[:2]}..{target_times_slice[-1:]}, "
                f"expected {expected_targets[:2]}..{expected_targets[-1:]}"
            )
            target_intervals_valid = False

        # Value null checks on target intervals
        for v in WEATHER_VARIABLES:
            vals = hourly.get(v, [])
            if len(vals) < 49:
                failure_reasons.append(f"Variable '{v}' array too short for {loc_name}")
                target_intervals_valid = False
                continue
            target_vals_slice = vals[25:49]
            nulls_in_var = sum(1 for val in target_vals_slice if val is None)
            if nulls_in_var > 0:
                failure_reasons.append(
                    f"Variable '{v}' has {nulls_in_var} nulls in target window for {loc_name}"
                )
                target_nulls_count += nulls_in_var

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
    timeout_seconds: int = 15,
) -> EiaApiAuditResult:
    """Check live EIA v2 API coverage for ERCOT demand, reading key safely from environment."""
    api_key = os.environ.get("EIA_API_KEY", "").strip()
    if not api_key:
        msg = (
            "EIA_API_KEY environment variable is not configured. "
            "Demand data for the 2026 holdout cannot be queried."
        )
        return EiaApiAuditResult(
            api_key_configured=False,
            status="BLOCKED",
            total_reported=0,
            start_period=start_period,
            end_period=end_period,
            message=msg,
        )

    url = (
        f"https://api.eia.gov/v2/electricity/rto/region-data/data/?"
        f"api_key={api_key}&frequency=hourly&data[0]=value&"
        f"facets[respondent][]=ERCO&facets[type][]=D&"
        f"start={start_period}&end={end_period}&length=1"
    )

    req = urllib.request.Request(url, headers={"User-Agent": "PowerTide-Preflight/0.1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return EiaApiAuditResult(
            api_key_configured=True,
            status="BLOCKED",
            total_reported=0,
            start_period=start_period,
            end_period=end_period,
            message=f"EIA API request failed with HTTP {e.code}: {e.reason}",
        )
    except Exception as e:
        return EiaApiAuditResult(
            api_key_configured=True,
            status="ERROR",
            total_reported=0,
            start_period=start_period,
            end_period=end_period,
            message=f"EIA API request encountered error: {e}",
        )

    raw_total = data.get("response", {}).get("total", 0)
    try:
        total = int(raw_total)
    except (ValueError, TypeError):
        total = 0
    status = "GO" if total >= EXPECTED_HOLDOUT_HOURS else "BLOCKED"
    message = (
        f"EIA API returned {total} matching records for ERCOT demand "
        f"(expected: {EXPECTED_HOLDOUT_HOURS})."
    )

    return EiaApiAuditResult(
        api_key_configured=True,
        status=status,
        total_reported=total,
        start_period=start_period,
        end_period=end_period,
        message=message,
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
