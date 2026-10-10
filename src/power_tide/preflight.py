"""Data coverage preflight and verification utilities for PowerTide.

Provides functions to audit:
- Local raw EIA demand and forecast JSON archives
- ECMWF IFS HRES 00:00 UTC Single Run availability via Open-Meteo
- Alignment with the fixed 2026 holdout evaluation window (2,208 hourly intervals)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

# Operational evaluation window constants
HOLDOUT_TARGET_START: Final[str] = "2026-07-01T01"
HOLDOUT_TARGET_END: Final[str] = "2026-10-01T00"
EXPECTED_HOLDOUT_HOURS: Final[int] = 2_208

# Proxy coordinates for Texas load centers
TEXAS_PROXY_LOCATIONS: Final[tuple[dict[str, Any], ...]] = (
    {"name": "Dallas", "latitude": 32.7767, "longitude": -96.7970},
    {"name": "Houston", "latitude": 29.7604, "longitude": -95.3698},
    {"name": "Austin", "latitude": 30.2672, "longitude": -97.7431},
    {"name": "San Antonio", "latitude": 29.4241, "longitude": -98.4936},
    {"name": "Midland", "latitude": 31.9973, "longitude": -102.0779},
)

WEATHER_VARIABLES: Final[tuple[str, ...]] = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
)


class EiaDatasetSummary(BaseModel):
    """Summary of local raw EIA dataset coverage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_type: str = Field(..., description="EIA series type: 'D' (demand) or 'DF' (forecast)")
    total_records: int = Field(..., description="Total records combined across files")
    unique_periods: int = Field(..., description="Count of distinct period timestamps")
    duplicate_periods: int = Field(..., description="Count of repeated period timestamps")
    null_value_count: int = Field(..., description="Count of records with missing values")
    min_period: str = Field(..., description="Earliest observed timestamp (e.g. 2024-01-01T00)")
    max_period: str = Field(..., description="Latest observed timestamp (e.g. 2025-12-31T23)")
    holdout_hours_present: int = Field(
        ..., description="Count of records matching the 2026 holdout window"
    )


class WeatherRunAuditResult(BaseModel):
    """Audit result for a single ECMWF IFS HRES 00:00 UTC model run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_date: str = Field(..., description="Run initialization date (YYYY-MM-DD)")
    available: bool = Field(..., description="Whether run was returned successfully")
    locations_count: int = Field(..., description="Number of locations returned")
    lead_hours_count: int = Field(..., description="Number of forecast hours in horizon")
    target_window_nulls: int = Field(
        ..., description="Null count in the D+1 target window (leads +25 to +48)"
    )


def audit_local_eia_json_files(json_paths: list[Path]) -> EiaDatasetSummary:
    """Inspect and validate local raw EIA JSON files without mutating them."""
    if not json_paths:
        raise ValueError("json_paths list cannot be empty")

    records: list[dict[str, Any]] = []
    dataset_type = "UNKNOWN"

    for path in json_paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        batch = data.get("response", {}).get("data", [])
        records.extend(batch)
        if dataset_type == "UNKNOWN" and batch:
            dataset_type = batch[0].get("type", "UNKNOWN")

    periods = [r["period"] for r in records if "period" in r]
    period_counts = Counter(periods)
    duplicate_count = sum(c - 1 for c in period_counts.values() if c > 1)
    null_count = sum(1 for r in records if r.get("value") is None)

    # Check matching holdout timestamps (2026-07-01T01 through 2026-10-01T00)
    holdout_count = sum(1 for p in periods if HOLDOUT_TARGET_START <= p <= HOLDOUT_TARGET_END)

    return EiaDatasetSummary(
        dataset_type=dataset_type,
        total_records=len(records),
        unique_periods=len(period_counts),
        duplicate_periods=duplicate_count,
        null_value_count=null_count,
        min_period=min(periods) if periods else "",
        max_period=max(periods) if periods else "",
        holdout_hours_present=holdout_count,
    )


def query_ecmwf_ifs_run_audit(
    run_date: date,
    timeout_seconds: int = 15,
) -> WeatherRunAuditResult:
    """Query Open-Meteo Single Runs API to audit a specific ECMWF IFS HRES 00:00 UTC run."""
    lats = ",".join(str(loc["latitude"]) for loc in TEXAS_PROXY_LOCATIONS)
    lons = ",".join(str(loc["longitude"]) for loc in TEXAS_PROXY_LOCATIONS)
    vars_param = ",".join(WEATHER_VARIABLES)
    d_str = run_date.isoformat()

    url = (
        f"https://single-runs-api.open-meteo.com/v1/forecast?"
        f"latitude={lats}&longitude={lons}&models=ecmwf_ifs025&"
        f"run={d_str}T00:00&hourly={vars_param}"
    )

    req = urllib.request.Request(url, headers={"User-Agent": "PowerTide-Preflight/0.1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return WeatherRunAuditResult(
            run_date=d_str,
            available=False,
            locations_count=0,
            lead_hours_count=0,
            target_window_nulls=-1,
        )

    if not isinstance(data, list) or len(data) != len(TEXAS_PROXY_LOCATIONS):
        return WeatherRunAuditResult(
            run_date=d_str,
            available=False,
            locations_count=len(data) if isinstance(data, list) else 0,
            lead_hours_count=0,
            target_window_nulls=-1,
        )

    target_nulls = 0
    total_leads = len(data[0].get("hourly", {}).get("time", []))

    for loc in data:
        hourly = loc.get("hourly", {})
        for v in WEATHER_VARIABLES:
            vals = hourly.get(v, [])
            # Target day D+1 corresponds to leads +25h to +48h from 00:00 UTC initialization
            target_slice = vals[25:49]
            target_nulls += sum(1 for val in target_slice if val is None)

    return WeatherRunAuditResult(
        run_date=d_str,
        available=True,
        locations_count=len(data),
        lead_hours_count=total_leads,
        target_window_nulls=target_nulls,
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
