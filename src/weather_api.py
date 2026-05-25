"""Open-Meteo historical-weather client.

Pulls daily weather (temperature, precipitation, snowfall, wind) for a
single point (default: Toronto City Hall) over a date range, and caches
the result as a parquet file under ``data/external/``. No API key is
required; the endpoint is the free Open-Meteo *archive* API which
serves ERA5 reanalysis values.

Public functions:

* :func:`fetch_toronto_weather` — convenience wrapper used by ``main.py``.
* :func:`fetch_weather`        — generic client with explicit args.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from config import (
    EXTERNAL_DIR,
    MODEL_END_DATE,
    MODEL_START_DATE,
    TORONTO_LAT,
    TORONTO_LON,
)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "wind_speed_10m_max",
]

logger = logging.getLogger(__name__)


def _cache_path(lat: float, lon: float, start: str, end: str) -> Path:
    return EXTERNAL_DIR / f"weather_{lat:.4f}_{lon:.4f}_{start}_{end}.parquet"


def fetch_weather(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    *,
    use_cache: bool = True,
    timeout: int = 30,
) -> pd.DataFrame:
    """Fetch daily weather for ``lat, lon`` over the closed range ``[start, end]``."""
    cache = _cache_path(lat, lon, start_date, end_date)
    if use_cache and cache.exists():
        try:
            return pd.read_parquet(cache)
        except ImportError:
            pass

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "America/Toronto",
    }
    logger.info("Calling Open-Meteo archive: lat=%.4f lon=%.4f %s..%s", lat, lon, start_date, end_date)
    response = requests.get(ARCHIVE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    daily = payload["daily"]

    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(
        columns={
            "time": "date",
            "temperature_2m_max": "temp_max_c",
            "temperature_2m_min": "temp_min_c",
            "temperature_2m_mean": "temp_mean_c",
            "precipitation_sum": "precip_mm",
            "rain_sum": "rain_mm",
            "snowfall_sum": "snow_cm",
            "wind_speed_10m_max": "wind_max_kmh",
        }
    )
    df["below_freezing"] = (df["temp_min_c"] < 0).astype(int)
    df["freeze_thaw"] = ((df["temp_min_c"] < 0) & (df["temp_max_c"] > 0)).astype(int)

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(cache, index=False)
    except ImportError:
        pass
    return df


def fetch_toronto_weather(use_cache: bool = True) -> pd.DataFrame:
    """Daily Toronto weather for the project's modeling window."""
    return fetch_weather(
        TORONTO_LAT,
        TORONTO_LON,
        MODEL_START_DATE,
        MODEL_END_DATE,
        use_cache=use_cache,
    )
