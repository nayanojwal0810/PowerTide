"""Unit tests for the PowerTide canonical forecast contract.

Verifies:
- Issue-time, timezone, and hourly alignment constraints
- Historical demand cutoff bounds and observation validation
- 24-hour target interval sequence (D+1 calendar day, +18h to +41h leads)
- Month, year, and leap-year transitions
- Numerical constraints (non-negative, finite)
- Payload integrity and summary consistency
- Rejection of invalid offsets, naive datetimes, and mismatched weather runs
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from power_tide.contract import (
    CUTOFF_BUFFER_HOURS,
    DEMAND_CUTOFF_HOUR_UTC,
    ISSUE_HOUR_UTC,
    MAX_LEAD_HOURS,
    MIN_LEAD_HOURS,
    TARGET_HORIZON_HOURS,
    WEATHER_MODEL_NAME,
    WEATHER_RUN_HOUR_UTC,
    ForecastIntervalPrediction,
    ForecastPayload,
    ForecastSummary,
    compute_demand_cutoff,
    compute_target_intervals,
    ensure_utc_hour_aligned,
    validate_demand_observation_timestamp,
    validate_issue_time,
)


def make_valid_predictions(
    issue_time: datetime, base_demand: float = 40_000.0
) -> list[ForecastIntervalPrediction]:
    """Helper to generate a valid sequence of 24 interval predictions."""
    predictions: list[ForecastIntervalPrediction] = []
    for lead in range(MIN_LEAD_HOURS, MAX_LEAD_HOURS + 1):
        target = issue_time + timedelta(hours=lead)
        # Vary demand so peak is uniquely at lead 24 (+15h into target day)
        demand = base_demand + (10_000.0 if lead == 30 else lead * 100.0)
        predictions.append(
            ForecastIntervalPrediction(
                target_period_end=target,
                lead_hours=lead,
                predicted_demand_mwh=demand,
            )
        )
    return predictions


def make_valid_payload(issue_time: datetime, include_summary: bool = True) -> ForecastPayload:
    """Helper to construct a fully valid ForecastPayload."""
    cutoff = compute_demand_cutoff(issue_time)
    weather_run = datetime(
        issue_time.year,
        issue_time.month,
        issue_time.day,
        WEATHER_RUN_HOUR_UTC,
        0,
        0,
        tzinfo=UTC,
    )
    predictions = make_valid_predictions(issue_time)

    summary: ForecastSummary | None = None
    if include_summary:
        total = sum(p.predicted_demand_mwh for p in predictions)
        peak_pred = max(predictions, key=lambda p: p.predicted_demand_mwh)
        summary = ForecastSummary(
            total_demand_mwh=total,
            peak_demand_mwh=peak_pred.predicted_demand_mwh,
            peak_period_end=peak_pred.target_period_end,
        )

    return ForecastPayload(
        issue_time=issue_time,
        demand_cutoff=cutoff,
        weather_model=WEATHER_MODEL_NAME,
        weather_run_utc=weather_run,
        predictions=predictions,
        summary=summary,
    )


class TestTimestampAndTzValidation:
    """Tests for strict UTC and hour-alignment enforcement."""

    def test_valid_utc_hour_aligned(self) -> None:
        dt = datetime(2026, 7, 15, 7, 0, 0, tzinfo=UTC)
        assert ensure_utc_hour_aligned(dt) == dt

    def test_reject_naive_datetime(self) -> None:
        dt = datetime(2026, 7, 15, 7, 0, 0)
        with pytest.raises(ValueError, match="naive"):
            ensure_utc_hour_aligned(dt)

    def test_reject_non_utc_offset(self) -> None:
        non_utc = timezone(timedelta(hours=5, minutes=30))
        dt = datetime(2026, 7, 15, 7, 0, 0, tzinfo=non_utc)
        with pytest.raises(ValueError, match="UTC offset"):
            ensure_utc_hour_aligned(dt)

    def test_reject_fractional_minutes(self) -> None:
        dt = datetime(2026, 7, 15, 7, 30, 0, tzinfo=UTC)
        with pytest.raises(ValueError, match="exact hour boundary"):
            ensure_utc_hour_aligned(dt)

    def test_reject_sub_second_fraction(self) -> None:
        dt = datetime(2026, 7, 15, 7, 0, 0, 123456, tzinfo=UTC)
        with pytest.raises(ValueError, match="exact hour boundary"):
            ensure_utc_hour_aligned(dt)

    def test_reject_seconds(self) -> None:
        dt = datetime(2026, 7, 15, 7, 0, 15, tzinfo=UTC)
        with pytest.raises(ValueError, match="exact hour boundary"):
            ensure_utc_hour_aligned(dt)


class TestIssueTimeAndCutoff:
    """Tests for issue time validation and demand cutoff computation."""

    def test_valid_issue_time(self) -> None:
        dt = datetime(2026, 7, 15, ISSUE_HOUR_UTC, 0, 0, tzinfo=UTC)
        assert validate_issue_time(dt) == dt

    @pytest.mark.parametrize("hour", [0, 4, 6, 8, 12, 18, 23])
    def test_reject_invalid_issue_hours(self, hour: int) -> None:
        dt = datetime(2026, 7, 15, hour, 0, 0, tzinfo=UTC)
        with pytest.raises(ValueError, match="Forecast issue time must be 07:00 UTC"):
            validate_issue_time(dt)

    def test_compute_demand_cutoff(self) -> None:
        issue = datetime(2026, 7, 15, 7, 0, 0, tzinfo=UTC)
        cutoff = compute_demand_cutoff(issue)
        assert cutoff == datetime(2026, 7, 15, DEMAND_CUTOFF_HOUR_UTC, 0, 0, tzinfo=UTC)
        assert (issue - cutoff) == timedelta(hours=CUTOFF_BUFFER_HOURS)

    def test_validate_demand_observation_at_cutoff(self) -> None:
        issue = datetime(2026, 7, 15, 7, 0, 0, tzinfo=UTC)
        obs_at_cutoff = datetime(2026, 7, 15, 4, 0, 0, tzinfo=UTC)
        assert validate_demand_observation_timestamp(obs_at_cutoff, issue) == obs_at_cutoff

    def test_validate_demand_observation_before_cutoff(self) -> None:
        issue = datetime(2026, 7, 15, 7, 0, 0, tzinfo=UTC)
        obs_before = datetime(2026, 7, 15, 3, 0, 0, tzinfo=UTC)
        assert validate_demand_observation_timestamp(obs_before, issue) == obs_before

    def test_reject_demand_observation_after_cutoff(self) -> None:
        issue = datetime(2026, 7, 15, 7, 0, 0, tzinfo=UTC)
        obs_after = datetime(2026, 7, 15, 5, 0, 0, tzinfo=UTC)
        with pytest.raises(ValueError, match="exceeds demand cutoff"):
            validate_demand_observation_timestamp(obs_after, issue)

    def test_reject_non_aligned_observation_timestamp(self) -> None:
        issue = datetime(2026, 7, 15, 7, 0, 0, tzinfo=UTC)
        obs_unaligned = datetime(2026, 7, 15, 3, 30, 0, tzinfo=UTC)
        with pytest.raises(ValueError, match="exact hour boundary"):
            validate_demand_observation_timestamp(obs_unaligned, issue)


class TestTargetHorizonAndIntervals:
    """Tests for 24-interval EIA hour-ending horizon."""

    def test_compute_target_intervals_count_and_leads(self) -> None:
        issue = datetime(2026, 7, 15, 7, 0, 0, tzinfo=UTC)
        targets = compute_target_intervals(issue)
        assert len(targets) == TARGET_HORIZON_HOURS

        # Day D+1 starts at 01:00 UTC D+1 and ends at 00:00 UTC D+2
        assert targets[0] == datetime(2026, 7, 16, 1, 0, 0, tzinfo=UTC)
        assert targets[-1] == datetime(2026, 7, 17, 0, 0, 0, tzinfo=UTC)

        for idx, target in enumerate(targets):
            lead = MIN_LEAD_HOURS + idx
            assert (target - issue) == timedelta(hours=lead)

    def test_month_transition(self) -> None:
        # Issue on July 31 07:00 UTC -> Day D+1 is August 1
        issue = datetime(2026, 7, 31, 7, 0, 0, tzinfo=UTC)
        targets = compute_target_intervals(issue)
        assert targets[0] == datetime(2026, 8, 1, 1, 0, 0, tzinfo=UTC)
        assert targets[-1] == datetime(2026, 8, 2, 0, 0, 0, tzinfo=UTC)

    def test_year_transition(self) -> None:
        # Issue on Dec 31 07:00 UTC -> Day D+1 is Jan 1
        issue = datetime(2026, 12, 31, 7, 0, 0, tzinfo=UTC)
        targets = compute_target_intervals(issue)
        assert targets[0] == datetime(2027, 1, 1, 1, 0, 0, tzinfo=UTC)
        assert targets[-1] == datetime(2027, 1, 2, 0, 0, 0, tzinfo=UTC)

    def test_leap_year_transition(self) -> None:
        # 2028 is leap year. Issue Feb 28 07:00 UTC -> Day D+1 is Feb 29
        issue = datetime(2028, 2, 28, 7, 0, 0, tzinfo=UTC)
        targets = compute_target_intervals(issue)
        assert targets[0] == datetime(2028, 2, 29, 1, 0, 0, tzinfo=UTC)
        assert targets[-1] == datetime(2028, 3, 1, 0, 0, 0, tzinfo=UTC)


class TestForecastPayloadValidation:
    """Tests for Pydantic v2 ForecastPayload schema validation."""

    def test_valid_payload_creation(self) -> None:
        issue = datetime(2026, 7, 15, 7, 0, 0, tzinfo=UTC)
        payload = make_valid_payload(issue)
        assert payload.issue_time == issue
        assert len(payload.predictions) == 24
        assert payload.summary is not None
        assert payload.weather_model == WEATHER_MODEL_NAME

    def test_reject_negative_demand(self) -> None:
        target = datetime(2026, 7, 16, 1, 0, 0, tzinfo=UTC)
        with pytest.raises(ValidationError, match="non-negative"):
            ForecastIntervalPrediction(
                target_period_end=target,
                lead_hours=18,
                predicted_demand_mwh=-10.0,
            )

    @pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
    def test_reject_non_finite_demand(self, bad_val: float) -> None:
        target = datetime(2026, 7, 16, 1, 0, 0, tzinfo=UTC)
        with pytest.raises(ValidationError, match="finite"):
            ForecastIntervalPrediction(
                target_period_end=target,
                lead_hours=18,
                predicted_demand_mwh=bad_val,
            )

    def test_reject_invalid_lead_hour(self) -> None:
        target = datetime(2026, 7, 16, 0, 0, 0, tzinfo=UTC)
        with pytest.raises(ValidationError, match="lead_hours must be between"):
            ForecastIntervalPrediction(
                target_period_end=target,
                lead_hours=17,
                predicted_demand_mwh=35_000.0,
            )

    def test_reject_missing_interval_count(self) -> None:
        issue = datetime(2026, 7, 15, 7, 0, 0, tzinfo=UTC)
        preds = make_valid_predictions(issue)[:23]  # Only 23 intervals
        weather_run = datetime(2026, 7, 15, 0, 0, 0, tzinfo=UTC)
        cutoff = compute_demand_cutoff(issue)
        with pytest.raises(ValidationError, match="exactly 24 intervals"):
            ForecastPayload(
                issue_time=issue,
                demand_cutoff=cutoff,
                weather_run_utc=weather_run,
                predictions=preds,
            )

    def test_reject_duplicate_target_intervals(self) -> None:
        issue = datetime(2026, 7, 15, 7, 0, 0, tzinfo=UTC)
        preds = make_valid_predictions(issue)
        # Duplicate first prediction at index 1
        preds[1] = preds[0]
        weather_run = datetime(2026, 7, 15, 0, 0, 0, tzinfo=UTC)
        cutoff = compute_demand_cutoff(issue)
        with pytest.raises(ValidationError):
            ForecastPayload(
                issue_time=issue,
                demand_cutoff=cutoff,
                weather_run_utc=weather_run,
                predictions=preds,
            )

    def test_reject_inconsistent_summary_total(self) -> None:
        issue = datetime(2026, 7, 15, 7, 0, 0, tzinfo=UTC)
        payload = make_valid_payload(issue)
        assert payload.summary is not None
        bad_summary = ForecastSummary(
            total_demand_mwh=payload.summary.total_demand_mwh + 1000.0,
            peak_demand_mwh=payload.summary.peak_demand_mwh,
            peak_period_end=payload.summary.peak_period_end,
        )
        with pytest.raises(ValidationError, match="does not match sum of predictions"):
            ForecastPayload(
                issue_time=payload.issue_time,
                demand_cutoff=payload.demand_cutoff,
                weather_run_utc=payload.weather_run_utc,
                predictions=payload.predictions,
                summary=bad_summary,
            )

    def test_reject_inconsistent_summary_peak(self) -> None:
        issue = datetime(2026, 7, 15, 7, 0, 0, tzinfo=UTC)
        payload = make_valid_payload(issue)
        assert payload.summary is not None
        bad_summary = ForecastSummary(
            total_demand_mwh=payload.summary.total_demand_mwh,
            peak_demand_mwh=payload.summary.peak_demand_mwh + 500.0,
            peak_period_end=payload.summary.peak_period_end,
        )
        with pytest.raises(ValidationError, match="does not match maximum prediction"):
            ForecastPayload(
                issue_time=payload.issue_time,
                demand_cutoff=payload.demand_cutoff,
                weather_run_utc=payload.weather_run_utc,
                predictions=payload.predictions,
                summary=bad_summary,
            )

    def test_reject_invalid_weather_model_name(self) -> None:
        issue = datetime(2026, 7, 15, 7, 0, 0, tzinfo=UTC)
        payload = make_valid_payload(issue)
        with pytest.raises(ValidationError, match="weather_model must be strictly"):
            ForecastPayload(
                issue_time=payload.issue_time,
                demand_cutoff=payload.demand_cutoff,
                weather_model="GFS-0.25",
                weather_run_utc=payload.weather_run_utc,
                predictions=payload.predictions,
            )

    def test_reject_mismatched_weather_run_date(self) -> None:
        issue = datetime(2026, 7, 15, 7, 0, 0, tzinfo=UTC)
        payload = make_valid_payload(issue)
        # Weather run from previous day
        wrong_date_run = datetime(2026, 7, 14, 0, 0, 0, tzinfo=UTC)
        with pytest.raises(ValidationError, match="weather_run_utc date"):
            ForecastPayload(
                issue_time=payload.issue_time,
                demand_cutoff=payload.demand_cutoff,
                weather_run_utc=wrong_date_run,
                predictions=payload.predictions,
            )

    def test_reject_wrong_weather_run_hour(self) -> None:
        issue = datetime(2026, 7, 15, 7, 0, 0, tzinfo=UTC)
        payload = make_valid_payload(issue)
        # Weather run at 06:00 UTC instead of 00:00 UTC
        wrong_hour_run = datetime(2026, 7, 15, 6, 0, 0, tzinfo=UTC)
        with pytest.raises(
            ValidationError, match="weather_run_utc must be initialized at 00:00 UTC"
        ):
            ForecastPayload(
                issue_time=payload.issue_time,
                demand_cutoff=payload.demand_cutoff,
                weather_run_utc=wrong_hour_run,
                predictions=payload.predictions,
            )
