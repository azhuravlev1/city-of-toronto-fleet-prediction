"""Loaders for the three raw datasets.

Responsibilities of this module:

1. Read the three Excel files provided by the City of Toronto.
2. Normalize the cross-dataset join key ``VEHICLE`` so that it has a
   consistent representation everywhere (zero-padded 6-character string).
   This is necessary because the raw files store the column with different
   dtypes (string-with-leading-zeros in the fault file, integer in the
   other two), which would otherwise silently break any join.
3. Parse datetime columns to timezone-naive ``datetime64[ns]``.
4. Normalize a few sentinel values (e.g. ``FAILURE_MODE == "0"`` which is
   used for telematics-device-internal events that are not engine faults).
5. Cache to Parquet under ``data/processed/`` so that re-runs of the EDA
   notebook are fast.

All functions are deterministic and side-effect free except for the
optional cache write.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from config import (
    DRIVING_EVENTS_FILE,
    FAULTS_FILE,
    JOBS_FILE,
    PROCESSED_DIR,
)

VEHICLE_ID_WIDTH = 6


def _normalize_vehicle_id(series: pd.Series) -> pd.Series:
    """Return a zero-padded 6-character string representation of vehicle IDs.

    Handles inputs that are integers, floats (e.g. when Excel coerces a
    column to float because of missing values), or strings with or without
    leading zeros.
    """
    # Strip any whitespace and a possible trailing ``.0`` from numeric coerce.
    as_str = (
        series.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    return as_str.str.zfill(VEHICLE_ID_WIDTH)


def _cache_path(name: str) -> Path:
    return PROCESSED_DIR / f"{name}.parquet"


def _read_with_cache(
    name: str,
    raw_path: Path,
    parser,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Read ``raw_path`` via ``parser`` with a parquet cache layer.

    The cache is invalidated when the raw file's mtime is newer than the
    cached parquet's mtime, so editing the source xlsx triggers a fresh
    parse automatically. If pyarrow / fastparquet are not installed the
    cache layer is silently skipped (the parse still works, just slower).
    """
    cache = _cache_path(name)
    if (
        use_cache
        and cache.exists()
        and cache.stat().st_mtime >= raw_path.stat().st_mtime
    ):
        try:
            return pd.read_parquet(cache)
        except ImportError:
            pass  # No parquet engine; fall through to a fresh parse.

    df = parser(raw_path)
    if use_cache:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        try:
            df.to_parquet(cache, index=False)
        except ImportError:
            # pyarrow not installed; the project still works without the cache.
            pass
    return df


# ---------------------------------------------------------------------------
# Individual dataset parsers
# ---------------------------------------------------------------------------


def _duration_to_seconds(value) -> float:
    """Convert a duration cell (datetime.time, str, or timedelta) to seconds."""
    import datetime as _dt

    if value is None or pd.isna(value):
        return float("nan")
    if isinstance(value, _dt.time):
        return (
            value.hour * 3600.0
            + value.minute * 60.0
            + value.second
            + value.microsecond / 1_000_000.0
        )
    if isinstance(value, _dt.timedelta):
        return value.total_seconds()
    return pd.to_timedelta(str(value)).total_seconds()


def _parse_driving_events(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["VEHICLE"] = _normalize_vehicle_id(df["VEHICLE"])
    df["DATETIME_START"] = pd.to_datetime(df["DATETIME_START"])
    # openpyxl returns durations < 24h as ``datetime.time``. Normalize to
    # floating-point seconds and a timedelta column for downstream resampling.
    df["DURATION_SECONDS"] = df["DURATION"].map(_duration_to_seconds).astype(float)
    df["DURATION_TD"] = pd.to_timedelta(df["DURATION_SECONDS"], unit="s")
    # Replace the original ``DURATION`` column (which contained un-pickleable
    # ``datetime.time`` objects) with a string form that survives parquet.
    df["DURATION"] = df["DURATION_TD"].astype(str)
    return df


def _parse_faults(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["VEHICLE"] = _normalize_vehicle_id(df["VEHICLE"])
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])

    # FAILURE_MODE == "0" is used by the Telematics Device source to indicate
    # an internal device event that does not carry an engine failure mode.
    # We normalize it to NaN so it does not pollute counts of "real" failure
    # modes.
    df.loc[df["FAILURE_MODE"].astype("string").str.strip() == "0", "FAILURE_MODE"] = pd.NA

    # FAULT_SEVERITY is missing for ~55% of rows. We add an explicit
    # "Unrated" category to keep it as a usable feature later on.
    df["FAULT_SEVERITY_FILLED"] = df["FAULT_SEVERITY"].fillna("Unrated")
    return df


def _parse_jobs(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df["VEHICLE"] = _normalize_vehicle_id(df["VEHICLE"])
    df["JOB_OPEN_DT"] = pd.to_datetime(df["JOB_OPEN_DT"])
    df["JOB_COMPLETED_DT"] = pd.to_datetime(df["JOB_COMPLETED_DT"])
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_driving_events(use_cache: bool = True) -> pd.DataFrame:
    """Return the cleaned Driving Events dataset."""
    return _read_with_cache(
        "driving_events", DRIVING_EVENTS_FILE, _parse_driving_events, use_cache=use_cache
    )


def load_faults(use_cache: bool = True) -> pd.DataFrame:
    """Return the cleaned Engine Device Fault Alerts dataset."""
    return _read_with_cache(
        "faults", FAULTS_FILE, _parse_faults, use_cache=use_cache
    )


def load_jobs(use_cache: bool = True) -> pd.DataFrame:
    """Return the cleaned Job History dataset."""
    return _read_with_cache(
        "jobs", JOBS_FILE, _parse_jobs, use_cache=use_cache
    )


def load_all(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Convenience wrapper used by both the notebook and ``main.py``."""
    return {
        "driving_events": load_driving_events(use_cache=use_cache),
        "faults": load_faults(use_cache=use_cache),
        "jobs": load_jobs(use_cache=use_cache),
    }


def vehicle_sets(frames: dict[str, pd.DataFrame]) -> dict[str, set[str]]:
    """Return the unique vehicle ID set for each frame, for diagnostics."""
    return {name: set(df["VEHICLE"].unique()) for name, df in frames.items()}


def summarize(frames: Iterable[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Compact row/vehicle/date-range summary used at the top of EDA."""
    rows = []
    for name, df in frames:
        date_col = next(
            (c for c in ("DATETIME_START", "DATE_TIME", "JOB_OPEN_DT") if c in df.columns),
            None,
        )
        rows.append(
            {
                "dataset": name,
                "rows": len(df),
                "unique_vehicles": df["VEHICLE"].nunique(),
                "date_min": df[date_col].min() if date_col else pd.NaT,
                "date_max": df[date_col].max() if date_col else pd.NaT,
            }
        )
    return pd.DataFrame(rows)
