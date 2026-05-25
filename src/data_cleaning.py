"""Domain-specific cleaning on top of the raw loaders.

Functions in this module operate on already-loaded DataFrames and produce
analysis-ready frames. They are kept separate from ``data_loading`` so the
two responsibilities (I/O + dtype normalization vs. business-domain
parsing) can be tested and reused independently.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from config import (
    EXCLUDED_REASONS,
    SCHEDULED_REASONS,
    UNSCHEDULED_REASONS,
)

# ---------------------------------------------------------------------------
# EXTRA_INFO parsing (Driving Events)
# ---------------------------------------------------------------------------
# Observed formats in the raw data (sample of three distinct patterns):
#   (Max Speed: 70 km/h (Max road speed: 50 km/h, Max truck road speed: 50 km/h))
#   (Max Speed: 68 km/h (Max road speed: 50 km/h))
#   (Max Speed: 63 km/h (Max road speed: 50 km/h, estimate))
#
# The regex below tolerates the optional truck-road-speed token and the
# optional ", estimate" marker. Any string that does not match any pattern
# is surfaced via :func:`unparsed_extra_info_examples` so the analyst can
# inspect outliers.

_MAX_SPEED_RE = re.compile(r"Max Speed:\s*(?P<max_speed>\d+)\s*km/h", re.IGNORECASE)
_ROAD_SPEED_RE = re.compile(
    r"Max road speed:\s*(?P<max_road_speed>\d+)\s*km/h", re.IGNORECASE
)
_TRUCK_SPEED_RE = re.compile(
    r"Max truck road speed:\s*(?P<max_truck_road_speed>\d+)\s*km/h", re.IGNORECASE
)
_ESTIMATE_RE = re.compile(r"\bestimate\b", re.IGNORECASE)


def parse_extra_info(series: pd.Series) -> pd.DataFrame:
    """Parse the free-text ``EXTRA_INFO`` column into structured fields.

    Returns a DataFrame with the same index as ``series`` and the columns:

    * ``max_speed_kmh`` (Int64): observed peak speed for the incident.
    * ``max_road_speed_kmh`` (Int64): the posted limit on that road.
    * ``max_truck_road_speed_kmh`` (Int64, often null): truck-specific limit.
    * ``is_estimate`` (bool): whether the posted limit was tagged "estimate".
    * ``speed_over_limit_kmh`` (Int64): ``max_speed - max_road_speed``.
    """
    s = series.fillna("").astype(str)

    def _extract(regex: re.Pattern, group: str) -> pd.Series:
        return s.str.extract(regex)[group]

    out = pd.DataFrame(index=series.index)
    out["max_speed_kmh"] = pd.to_numeric(_extract(_MAX_SPEED_RE, "max_speed"), errors="coerce").astype("Int64")
    out["max_road_speed_kmh"] = pd.to_numeric(_extract(_ROAD_SPEED_RE, "max_road_speed"), errors="coerce").astype("Int64")
    out["max_truck_road_speed_kmh"] = pd.to_numeric(_extract(_TRUCK_SPEED_RE, "max_truck_road_speed"), errors="coerce").astype("Int64")
    out["is_estimate"] = s.str.contains(_ESTIMATE_RE)
    out["speed_over_limit_kmh"] = (out["max_speed_kmh"] - out["max_road_speed_kmh"]).astype("Int64")
    return out


def extra_info_template(value: str) -> str:
    """Return a template form of an ``EXTRA_INFO`` string for novelty detection.

    All digits are collapsed to ``#``, so two strings that differ only in
    their numeric values map to the same template. This makes it easy to
    audit *how many distinct shapes* the EXTRA_INFO column has, separate
    from the actual numeric values.
    """
    if not isinstance(value, str):
        return ""
    return re.sub(r"\d+", "#", value)


def unparsed_extra_info_examples(df: pd.DataFrame, n: int = 5) -> pd.Series:
    """Return up to ``n`` example EXTRA_INFO strings that failed to parse."""
    parsed = parse_extra_info(df["EXTRA_INFO"])
    mask = parsed["max_speed_kmh"].isna() | parsed["max_road_speed_kmh"].isna()
    return df.loc[mask, "EXTRA_INFO"].head(n)


# ---------------------------------------------------------------------------
# Job History: scheduling classification
# ---------------------------------------------------------------------------


def classify_jobs(jobs: pd.DataFrame) -> pd.DataFrame:
    """Add a ``SCHEDULE_CLASS`` column to the jobs frame.

    Values:

    * ``"unscheduled"`` — counts toward the positive label.
    * ``"scheduled"``   — preventive / capital / recall / etc.
    * ``"excluded"``    — administrative or external work that we choose
      not to classify as either (see ``EXCLUDED_REASONS`` in ``config``).

    Any ``JOB_REASON`` value not in any of the three sets is mapped to
    ``"unknown"`` and surfaced for review.
    """
    out = jobs.copy()
    reason = out["JOB_REASON"].astype("string")

    conditions = [
        reason.isin(UNSCHEDULED_REASONS),
        reason.isin(SCHEDULED_REASONS),
        reason.isin(EXCLUDED_REASONS),
    ]
    choices = ["unscheduled", "scheduled", "excluded"]
    out["SCHEDULE_CLASS"] = np.select(conditions, choices, default="unknown")
    return out


def workorder_events(jobs: pd.DataFrame) -> pd.DataFrame:
    """Collapse jobs to one row per work order for labelling.

    A single ``WO_NO`` typically opens with the same ``JOB_OPEN_DT`` for
    all of its sub-jobs. We aggregate to the work-order grain because the
    "unscheduled repair event" we want to predict is the *opening* of an
    unscheduled work order, not each individual line on it.

    The work order's ``SCHEDULE_CLASS`` is set to ``"unscheduled"`` if any
    of its line items is unscheduled, otherwise ``"scheduled"`` if any is
    scheduled, otherwise ``"excluded"``.
    """
    classified = classify_jobs(jobs)
    grouped = classified.groupby("WO_NO", as_index=False).agg(
        VEHICLE=("VEHICLE", "first"),
        JOB_OPEN_DT=("JOB_OPEN_DT", "min"),
        JOB_COMPLETED_DT=("JOB_COMPLETED_DT", "max"),
        n_jobs=("JOB", "count"),
        has_unscheduled=("SCHEDULE_CLASS", lambda s: (s == "unscheduled").any()),
        has_scheduled=("SCHEDULE_CLASS", lambda s: (s == "scheduled").any()),
    )
    grouped["SCHEDULE_CLASS"] = np.where(
        grouped["has_unscheduled"],
        "unscheduled",
        np.where(grouped["has_scheduled"], "scheduled", "excluded"),
    )
    return grouped.drop(columns=["has_unscheduled", "has_scheduled"])
