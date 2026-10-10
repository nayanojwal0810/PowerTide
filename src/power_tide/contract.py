"""Canonical forecast contract and payload schemas for PowerTide.

Defines the locked operational contract:
- Forecast issue time: 07:00 UTC on day D
- Historical demand cutoff: 04:00 UTC on day D (3-hour operational buffer)
- Forecast horizon: 24 consecutive EIA hour-ending intervals for day D+1
  (from 01:00 UTC day D+1 through 00:00 UTC day D+2)
- Lead times: +18 hours through +41 hours inclusive
- Weather run: ECMWF IFS HRES 00:00 UTC Single Run
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Immutable contract constants
ISSUE_HOUR_UTC: Final[int] = 7
DEMAND_CUTOFF_HOUR_UTC: Final[int] = 4
CUTOFF_BUFFER_HOURS: Final[int] = 3
TARGET_HORIZON_HOURS: Final[int] = 24
MIN_LEAD_HOURS: Final[int] = 18
MAX_LEAD_HOURS: Final[int] = 41
WEATHER_MODEL_NAME: Final[str] = "ECMWF IFS HRES"
WEATHER_RUN_HOUR_UTC: Final[int] = 0


def ensure_utc_hour_aligned(dt: datetime, field_name: str = "datetime") -> datetime:
    """Validate that a datetime is timezone-aware UTC and aligned to an exact hour.

    Rejects naive datetimes, non-UTC offsets, fractional hours, minutes, seconds,
    or microseconds. Does not truncate or mutate timestamps.
    """
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware UTC, received naive datetime: {dt}")

    if dt.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must have UTC offset (+00:00), received: {dt.utcoffset()}")

    if dt.minute != 0 or dt.second != 0 or dt.microsecond != 0:
        raise ValueError(
            f"{field_name} must be aligned to exact hour boundary (:00:00.000000), "
            f"received: {dt.isoformat()}"
        )

    return dt


def validate_issue_time(issue_time: datetime) -> datetime:
    """Validate that the issue time conforms to the 07:00 UTC contract."""
    dt = ensure_utc_hour_aligned(issue_time, "issue_time")
    if dt.hour != ISSUE_HOUR_UTC:
        msg = (
            f"Forecast issue time must be {ISSUE_HOUR_UTC:02d}:00 UTC, "
            f"received: {dt.hour:02d}:00 UTC"
        )
        raise ValueError(msg)
    return dt


def compute_demand_cutoff(issue_time: datetime) -> datetime:
    """Compute the maximum allowed historical demand timestamp (04:00 UTC on day D)."""
    valid_issue = validate_issue_time(issue_time)
    cutoff = valid_issue - timedelta(hours=CUTOFF_BUFFER_HOURS)
    if cutoff.hour != DEMAND_CUTOFF_HOUR_UTC:
        msg = (
            f"Expected cutoff hour {DEMAND_CUTOFF_HOUR_UTC:02d}:00 UTC, "
            f"computed {cutoff.hour:02d}:00 UTC"
        )
        raise ValueError(msg)
    return cutoff


def compute_target_intervals(issue_time: datetime) -> list[datetime]:
    """Generate the 24 consecutive EIA hour-ending target timestamps for day D+1.

    Target timestamps range from D+1 01:00 UTC (+18h) to D+2 00:00 UTC (+41h).
    """
    valid_issue = validate_issue_time(issue_time)
    return [
        valid_issue + timedelta(hours=lead) for lead in range(MIN_LEAD_HOURS, MAX_LEAD_HOURS + 1)
    ]


def validate_demand_observation_timestamp(
    observation_time: datetime,
    issue_time: datetime,
) -> datetime:
    """Validate that an observed demand timestamp is hour-aligned and not past the cutoff."""
    obs_dt = ensure_utc_hour_aligned(observation_time, "observation_time")
    cutoff = compute_demand_cutoff(issue_time)
    if obs_dt > cutoff:
        raise ValueError(
            f"Observation timestamp {obs_dt.isoformat()} exceeds demand cutoff {cutoff.isoformat()}"
        )
    return obs_dt


class ForecastIntervalPrediction(BaseModel):
    """Prediction for a single EIA hour-ending target interval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_period_end: datetime = Field(
        ...,
        description="EIA hour-ending target timestamp in UTC",
    )
    lead_hours: int = Field(
        ...,
        description="Lead time relative to forecast issue time (+18 to +41)",
    )
    predicted_demand_mwh: float = Field(
        ...,
        description="Predicted ERCOT demand in MWh",
    )

    @field_validator("target_period_end")
    @classmethod
    def _validate_target_period_end(cls, v: datetime) -> datetime:
        return ensure_utc_hour_aligned(v, "target_period_end")

    @field_validator("lead_hours")
    @classmethod
    def _validate_lead_hours(cls, v: int) -> int:
        if v < MIN_LEAD_HOURS or v > MAX_LEAD_HOURS:
            raise ValueError(
                f"lead_hours must be between +{MIN_LEAD_HOURS} and +{MAX_LEAD_HOURS}, received: {v}"
            )
        return v

    @field_validator("predicted_demand_mwh")
    @classmethod
    def _validate_predicted_demand(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(f"predicted_demand_mwh must be a finite number, received: {v}")
        if v < 0.0:
            raise ValueError(f"predicted_demand_mwh must be non-negative, received: {v}")
        return v


class ForecastSummary(BaseModel):
    """Summary metrics across the 24-hour forecast horizon."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_demand_mwh: float = Field(..., description="Sum of predicted demand across 24 intervals")
    peak_demand_mwh: float = Field(..., description="Maximum predicted demand in the horizon")
    peak_period_end: datetime = Field(..., description="Timestamp of the peak demand interval")

    @field_validator("peak_period_end")
    @classmethod
    def _validate_peak_period_end(cls, v: datetime) -> datetime:
        return ensure_utc_hour_aligned(v, "peak_period_end")

    @field_validator("total_demand_mwh", "peak_demand_mwh")
    @classmethod
    def _validate_finite_positive(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(f"Summary value must be a finite number, received: {v}")
        if v < 0.0:
            raise ValueError(f"Summary value must be non-negative, received: {v}")
        return v


class ForecastPayload(BaseModel):
    """Complete 24-hour operational forecast artifact satisfying the contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    issue_time: datetime = Field(..., description="Forecast issue time (07:00 UTC day D)")
    demand_cutoff: datetime = Field(
        ..., description="Historical demand observation cutoff (04:00 UTC day D)"
    )
    weather_model: str = Field(
        default=WEATHER_MODEL_NAME,
        description="Numerical weather prediction model identity",
    )
    weather_run_utc: datetime = Field(
        ..., description="ECMWF IFS HRES run initialization timestamp (00:00 UTC day D)"
    )
    predictions: list[ForecastIntervalPrediction] = Field(
        ...,
        description="Exactly 24 hourly predictions spanning day D+1",
    )
    summary: ForecastSummary | None = Field(
        default=None,
        description="Optional summary metrics validated for strict consistency",
    )

    @field_validator("issue_time")
    @classmethod
    def _validate_issue_time(cls, v: datetime) -> datetime:
        return validate_issue_time(v)

    @field_validator("demand_cutoff")
    @classmethod
    def _validate_demand_cutoff(cls, v: datetime) -> datetime:
        return ensure_utc_hour_aligned(v, "demand_cutoff")

    @field_validator("weather_run_utc")
    @classmethod
    def _validate_weather_run(cls, v: datetime) -> datetime:
        dt = ensure_utc_hour_aligned(v, "weather_run_utc")
        if dt.hour != WEATHER_RUN_HOUR_UTC:
            raise ValueError(
                f"weather_run_utc must be initialized at {WEATHER_RUN_HOUR_UTC:02d}:00 UTC, "
                f"received: {dt.hour:02d}:00 UTC"
            )
        return dt

    @field_validator("weather_model")
    @classmethod
    def _validate_weather_model(cls, v: str) -> str:
        if v != WEATHER_MODEL_NAME:
            raise ValueError(
                f"weather_model must be strictly '{WEATHER_MODEL_NAME}', received: '{v}'"
            )
        return v

    @model_validator(mode="after")
    def _validate_full_payload_consistency(self) -> ForecastPayload:
        # Verify demand cutoff matches issue time exactly
        expected_cutoff = compute_demand_cutoff(self.issue_time)
        if self.demand_cutoff != expected_cutoff:
            raise ValueError(
                f"demand_cutoff mismatch: expected {expected_cutoff.isoformat()}, "
                f"received {self.demand_cutoff.isoformat()}"
            )

        # Weather run date must equal issue time date
        if self.weather_run_utc.date() != self.issue_time.date():
            raise ValueError(
                f"weather_run_utc date ({self.weather_run_utc.date()}) must match "
                f"issue_time date ({self.issue_time.date()})"
            )

        # Validate exactly 24 predictions
        if len(self.predictions) != TARGET_HORIZON_HOURS:
            raise ValueError(
                f"predictions must contain exactly {TARGET_HORIZON_HOURS} intervals, "
                f"received {len(self.predictions)}"
            )

        expected_targets = compute_target_intervals(self.issue_time)
        seen_targets: set[datetime] = set()

        for idx, pred in enumerate(self.predictions):
            expected_target = expected_targets[idx]
            expected_lead = MIN_LEAD_HOURS + idx

            if pred.target_period_end != expected_target:
                msg = (
                    f"Prediction interval {idx} mismatch: "
                    f"expected target {expected_target.isoformat()}, "
                    f"received {pred.target_period_end.isoformat()}"
                )
                raise ValueError(msg)

            if pred.lead_hours != expected_lead:
                raise ValueError(
                    f"Prediction interval {idx} lead mismatch: expected +{expected_lead}h, "
                    f"received +{pred.lead_hours}h"
                )

            # Check lead hours consistency with time delta
            lead_delta = pred.target_period_end - self.issue_time
            if lead_delta != timedelta(hours=pred.lead_hours):
                msg = (
                    f"Interval target {pred.target_period_end.isoformat()} is not "
                    f"+{pred.lead_hours}h from issue time {self.issue_time.isoformat()}"
                )
                raise ValueError(msg)

            if pred.target_period_end in seen_targets:
                raise ValueError(
                    f"Duplicate target_period_end detected: {pred.target_period_end.isoformat()}"
                )
            seen_targets.add(pred.target_period_end)

        # Validate summary consistency if provided
        if self.summary is not None:
            actual_total = sum(p.predicted_demand_mwh for p in self.predictions)
            actual_peak = max(p.predicted_demand_mwh for p in self.predictions)

            if not math.isclose(self.summary.total_demand_mwh, actual_total, rel_tol=1e-5):
                raise ValueError(
                    f"summary.total_demand_mwh ({self.summary.total_demand_mwh}) does not match "
                    f"sum of predictions ({actual_total})"
                )

            if not math.isclose(self.summary.peak_demand_mwh, actual_peak, rel_tol=1e-5):
                raise ValueError(
                    f"summary.peak_demand_mwh ({self.summary.peak_demand_mwh}) does not match "
                    f"maximum prediction ({actual_peak})"
                )

            # Check peak period corresponds to an interval having peak demand
            matching_peak_preds = [
                p
                for p in self.predictions
                if math.isclose(p.predicted_demand_mwh, self.summary.peak_demand_mwh, rel_tol=1e-5)
            ]
            matching_peak_times = {p.target_period_end for p in matching_peak_preds}
            if self.summary.peak_period_end not in matching_peak_times:
                raise ValueError(
                    f"summary.peak_period_end {self.summary.peak_period_end.isoformat()} does not "
                    f"match any interval with peak demand"
                )

        return self
