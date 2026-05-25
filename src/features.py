"""Per-(vehicle, day) feature matrix for the 7-day prediction task.

The output of :func:`build_feature_matrix` is a DataFrame with one row
per (vehicle, day) pair within the modeling window. Each row contains:

* engineered features computed from data **strictly before** that day
  (rolling fault counts, rolling driving aggregates, days-since-last-X);
* calendar features (day of week, is-holiday);
* trailing weather and forward-looking weather (the forward look uses
  historical weather as a proxy for the 7-day forecast, per the
  assumption documented in the README);
* the binary target ``y_unscheduled_next7`` — 1 if any unscheduled work
  order is opened in ``(day, day + 7d]``, else 0.

All rolling windows are *left-closed, right-open* relative to ``day`` so
no information from ``day`` itself leaks into the features. This makes
the feature matrix safe to use with ``TimeSeriesSplit``.
"""

from __future__ import annotations

import logging
from datetime import date

import holidays
import numpy as np
import pandas as pd

from config import (
    MODEL_END_DATE,
    MODEL_START_DATE,
    PREDICTION_HORIZON_DAYS,
    PROCESSED_DIR,
)

logger = logging.getLogger(__name__)

ONTARIO_HOLIDAYS = holidays.country_holidays("CA", subdiv="ON", years=[2024, 2025, 2026])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _daily_index(start: str, end: str) -> pd.DatetimeIndex:
    """Closed-closed daily date index for the modeling window."""
    return pd.date_range(start=start, end=end, freq="D", name="date")


def _events_to_daily_counts(
    events: pd.DataFrame,
    *,
    date_col: str,
    vehicles: list[str],
    daily_index: pd.DatetimeIndex,
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Aggregate an event table to a (vehicle, day) grid of counts.

    If ``weight_col`` is provided the function sums that column instead of
    counting rows (useful for total speeding seconds or distance).
    """
    df = events.copy()
    df["day"] = df[date_col].dt.normalize()
    grouped = (
        df.groupby(["VEHICLE", "day"], observed=True)[weight_col].sum()
        if weight_col is not None
        else df.groupby(["VEHICLE", "day"], observed=True).size()
    )
    grouped.name = "value"
    # Pivot to (day x vehicle), reindex to fill missing days with zero.
    wide = grouped.unstack("VEHICLE", fill_value=0).reindex(daily_index, fill_value=0)
    # Keep only requested vehicles, in deterministic order.
    wide = wide.reindex(columns=vehicles, fill_value=0)
    return wide.astype(float)


def _trailing_sum(wide: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling sum over the trailing ``window`` days, **not including the current day**.

    Equivalent to ``rolling(window).sum().shift(1)``. Result has the same
    shape as the input.
    """
    return wide.rolling(window=window, min_periods=1).sum().shift(1).fillna(0)


def _days_since_event(wide: pd.DataFrame, fill: int = 999) -> pd.DataFrame:
    """For each cell, the number of days since the last non-zero value.

    Returns ``fill`` for days where no event has yet occurred. The result
    uses *yesterday's* last-seen so the current day's event does not leak
    into its own feature.
    """
    # Mark each day with its own ordinal if an event happened, else NaN.
    ordinals = np.arange(len(wide), dtype="float64")
    occurred = wide.gt(0).astype("float64")
    # The ordinal of the event itself, or NaN.
    event_ordinal = wide.copy()
    event_ordinal.loc[:, :] = occurred.values * ordinals[:, None]
    event_ordinal = event_ordinal.where(occurred.astype(bool))
    # Shift by 1 so "today's" feature uses only past events.
    last_event_ord = event_ordinal.shift(1).ffill()
    today_ord = pd.DataFrame(
        np.broadcast_to(ordinals[:, None], wide.shape),
        index=wide.index,
        columns=wide.columns,
        dtype="float64",
    )
    days_since = today_ord - last_event_ord
    return days_since.fillna(fill).astype("float64")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_feature_matrix(
    drives: pd.DataFrame,
    faults: pd.DataFrame,
    workorders: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    cohort: list[str] | None = None,
    start: str = MODEL_START_DATE,
    end: str = MODEL_END_DATE,
    horizon_days: int = PREDICTION_HORIZON_DAYS,
) -> pd.DataFrame:
    """Construct the per-(vehicle, day) feature matrix."""
    if cohort is None:
        cohort = sorted(
            set(drives["VEHICLE"])
            & set(faults["VEHICLE"])
            & set(workorders["VEHICLE"])
        )
    logger.info("Building feature matrix for %d vehicles over %s..%s", len(cohort), start, end)

    daily = _daily_index(start, end)
    cohort = sorted(cohort)

    # ------------------------------------------------------------------
    # Faults
    # ------------------------------------------------------------------
    f_all = faults[faults["VEHICLE"].isin(cohort)]
    fault_daily = _events_to_daily_counts(f_all, date_col="DATE_TIME", vehicles=cohort, daily_index=daily)

    severity_high = f_all[f_all["FAULT_SEVERITY"].isin(["Critical", "High"])]
    high_daily = _events_to_daily_counts(severity_high, date_col="DATE_TIME", vehicles=cohort, daily_index=daily)

    severity_critical = f_all[f_all["FAULT_SEVERITY"] == "Critical"]
    critical_daily = _events_to_daily_counts(severity_critical, date_col="DATE_TIME", vehicles=cohort, daily_index=daily)

    # ------------------------------------------------------------------
    # Driving events
    # ------------------------------------------------------------------
    d_all = drives[drives["VEHICLE"].isin(cohort)]
    drive_daily = _events_to_daily_counts(d_all, date_col="DATETIME_START", vehicles=cohort, daily_index=daily)
    drive_duration_daily = _events_to_daily_counts(
        d_all, date_col="DATETIME_START", vehicles=cohort, daily_index=daily, weight_col="DURATION_SECONDS"
    )
    drive_distance_daily = _events_to_daily_counts(
        d_all, date_col="DATETIME_START", vehicles=cohort, daily_index=daily, weight_col="DISTANCE"
    )

    # ------------------------------------------------------------------
    # Work orders (used for both features and labels)
    # ------------------------------------------------------------------
    wo_all = workorders[workorders["VEHICLE"].isin(cohort)]
    wo_unsched = wo_all[wo_all["SCHEDULE_CLASS"] == "unscheduled"]
    wo_sched = wo_all[wo_all["SCHEDULE_CLASS"] == "scheduled"]

    unsched_daily = _events_to_daily_counts(wo_unsched, date_col="JOB_OPEN_DT", vehicles=cohort, daily_index=daily)
    sched_daily = _events_to_daily_counts(wo_sched, date_col="JOB_OPEN_DT", vehicles=cohort, daily_index=daily)

    # ------------------------------------------------------------------
    # Trailing-window features
    # ------------------------------------------------------------------
    # Lifetime-to-yesterday cumulative counts and rates for each vehicle.
    # These give the GBM the same long-memory signal as the frequency
    # baseline (per-vehicle WO rate over all training data).
    cum_unsched = unsched_daily.cumsum().shift(1).fillna(0)
    cum_faults = fault_daily.cumsum().shift(1).fillna(0)
    days_observed = pd.DataFrame(
        np.broadcast_to(
            (np.arange(len(daily), dtype="float64"))[:, None],
            unsched_daily.shape,
        ),
        index=unsched_daily.index,
        columns=unsched_daily.columns,
    )
    lifetime_unsched_rate = cum_unsched / days_observed.replace(0, np.nan)
    lifetime_fault_rate = cum_faults / days_observed.replace(0, np.nan)

    parts = {
        "fault_count_1d": _trailing_sum(fault_daily, 1),
        "fault_count_7d": _trailing_sum(fault_daily, 7),
        "fault_count_30d": _trailing_sum(fault_daily, 30),
        "high_sev_count_7d": _trailing_sum(high_daily, 7),
        "critical_count_30d": _trailing_sum(critical_daily, 30),
        "speeding_count_7d": _trailing_sum(drive_daily, 7),
        "speeding_count_30d": _trailing_sum(drive_daily, 30),
        "speeding_seconds_7d": _trailing_sum(drive_duration_daily, 7),
        "speeding_km_7d": _trailing_sum(drive_distance_daily, 7),
        "unsched_wo_count_30d": _trailing_sum(unsched_daily, 30),
        "unsched_wo_count_90d": _trailing_sum(unsched_daily, 90),
        "sched_wo_count_30d": _trailing_sum(sched_daily, 30),
        "days_since_unsched_wo": _days_since_event(unsched_daily),
        "days_since_sched_wo": _days_since_event(sched_daily),
        "days_since_critical_fault": _days_since_event(critical_daily),
        # Per-vehicle priors (lifetime up to yesterday).
        "lifetime_unsched_rate": lifetime_unsched_rate.fillna(0),
        "lifetime_fault_rate": lifetime_fault_rate.fillna(0),
        "lifetime_unsched_count": cum_unsched,
    }

    # ------------------------------------------------------------------
    # Long form (one row per vehicle-day)
    # ------------------------------------------------------------------
    long = (
        pd.concat({name: w.stack().rename("value") for name, w in parts.items()}, axis=0)
        .reset_index()
        .rename(columns={"level_0": "feature", "level_1": "date", "VEHICLE": "vehicle"})
        .pivot(index=["vehicle", "date"], columns="feature", values="value")
        .reset_index()
    )

    # ------------------------------------------------------------------
    # Calendar
    # ------------------------------------------------------------------
    long["dow"] = long["date"].dt.dayofweek
    long["month"] = long["date"].dt.month
    long["is_weekend"] = (long["dow"] >= 5).astype(int)
    long["is_holiday"] = long["date"].dt.date.map(lambda d: 1 if d in ONTARIO_HOLIDAYS else 0)

    def days_to_next_holiday(d: date) -> int:
        for offset in range(0, 30):
            if (d + pd.Timedelta(days=offset)).date() in ONTARIO_HOLIDAYS:
                return offset
        return 30
    long["days_to_next_holiday"] = long["date"].map(days_to_next_holiday)

    # ------------------------------------------------------------------
    # Weather — trailing window + forward-looking 7-day window
    # ------------------------------------------------------------------
    w = weather.copy()
    w["date"] = pd.to_datetime(w["date"]).dt.normalize()
    w = w.set_index("date").sort_index()

    # Trailing 7-day weather (shifted to exclude the current day, no leakage).
    trailing = w[["temp_mean_c", "precip_mm", "snow_cm", "below_freezing", "freeze_thaw"]].rolling(7, min_periods=1).agg({
        "temp_mean_c": "mean",
        "precip_mm": "sum",
        "snow_cm": "sum",
        "below_freezing": "sum",
        "freeze_thaw": "sum",
    }).shift(1).add_prefix("trailing7_")

    # Forward-looking 7-day weather (the "forecast" proxy in production).
    forward = (
        w[["temp_mean_c", "precip_mm", "snow_cm", "below_freezing", "freeze_thaw"]]
        .iloc[::-1]
        .rolling(7, min_periods=1)
        .agg({
            "temp_mean_c": "mean",
            "precip_mm": "sum",
            "snow_cm": "sum",
            "below_freezing": "sum",
            "freeze_thaw": "sum",
        })
        .iloc[::-1]
        .add_prefix("forecast7_")
    )

    today = w[["temp_max_c", "temp_min_c", "temp_mean_c"]].rename(
        columns=lambda c: f"today_{c.replace('_c', '_c')}"
    )

    wx = pd.concat([today, trailing, forward], axis=1).reset_index()
    long = long.merge(wx, on="date", how="left")

    # ------------------------------------------------------------------
    # Label: any unscheduled WO opened in (date, date + horizon] ?
    # ------------------------------------------------------------------
    label_window = pd.concat(
        [unsched_daily.shift(-h) for h in range(1, horizon_days + 1)]
    ).groupby(level=0).sum()
    label_long = (
        label_window.stack()
        .rename("y_unscheduled_next7")
        .reset_index()
        .rename(columns={"VEHICLE": "vehicle"})
    )
    label_long["y_unscheduled_next7"] = (label_long["y_unscheduled_next7"] > 0).astype(int)
    long = long.merge(label_long, on=["vehicle", "date"], how="left")
    long["y_unscheduled_next7"] = long["y_unscheduled_next7"].fillna(0).astype(int)

    # Drop the last ``horizon_days`` because the label is undefined there.
    cutoff = pd.Timestamp(end) - pd.Timedelta(days=horizon_days)
    long = long[long["date"] <= cutoff].reset_index(drop=True)

    logger.info(
        "Feature matrix shape: %s  positive rate: %.3f",
        long.shape,
        long["y_unscheduled_next7"].mean(),
    )
    return long


def cache_feature_matrix(df: pd.DataFrame, name: str = "features") -> None:
    """Persist the feature matrix as parquet so re-runs are fast."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
    except ImportError:
        df.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
