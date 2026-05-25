"""End-to-end pipeline for the Toronto Fleet 7-day unscheduled-repair model.

Stages
------
1. Load     — read the three raw .xlsx files, normalize vehicle IDs.
2. Clean    — parse ``EXTRA_INFO``, classify jobs, collapse to work-orders.
3. Weather  — fetch Toronto daily weather via Open-Meteo.
4. Features — build the per-(vehicle, day) feature matrix.
5. Model    — walk-forward training (HistGradientBoosting vs. frequency baseline).
6. Explain  — SHAP beeswarm summary of the final-fold model.

Run with the raw files in ``data/raw/``:

    python main.py            # full pipeline using cached intermediates where available
    python main.py --no-cache # force re-parse of the xlsx files
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import FIGURES_DIR, PROCESSED_DIR
from src import data_cleaning as dc
from src import data_loading as dl
from src import weather_api
from src.features import build_feature_matrix, cache_feature_matrix
from src.modeling import run_walk_forward, shap_summary

LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(message)s"


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt="%H:%M:%S",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cache", action="store_true", help="ignore parquet cache")
    parser.add_argument("--skip-shap", action="store_true", help="skip the SHAP summary step")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    use_cache = not args.no_cache

    logging.info("Stage 1/6 — loading raw datasets")
    frames = dl.load_all(use_cache=use_cache)
    for name, df in frames.items():
        logging.info("  %-15s rows=%-7d vehicles=%d", name, len(df), df["VEHICLE"].nunique())

    logging.info("Stage 2/6 — cleaning and classifying")
    workorders = dc.workorder_events(frames["jobs"])
    logging.info("  work orders: %d (%s)", len(workorders), workorders["SCHEDULE_CLASS"].value_counts().to_dict())

    logging.info("Stage 3/6 — fetching Toronto weather (Open-Meteo)")
    weather = weather_api.fetch_toronto_weather(use_cache=use_cache)
    logging.info("  %d daily rows  below_freezing=%d  freeze_thaw=%d  total_snow=%.1f cm",
                 len(weather), int(weather["below_freezing"].sum()),
                 int(weather["freeze_thaw"].sum()), weather["snow_cm"].sum())

    logging.info("Stage 4/6 — building per-(vehicle, day) features")
    features = build_feature_matrix(
        drives=frames["driving_events"],
        faults=frames["faults"],
        workorders=workorders,
        weather=weather,
    )
    cache_feature_matrix(features)
    logging.info("  feature matrix: %s rows × %s cols  positive_rate=%.3f",
                 f"{len(features):,}", features.shape[1], features["y_unscheduled_next7"].mean())

    logging.info("Stage 5/6 — walk-forward training and evaluation")
    results = run_walk_forward(features)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    results["per_fold"].to_csv(PROCESSED_DIR / "metrics_per_fold.csv", index=False)
    summary = results["summary"]
    logging.info("Cross-validated metrics (mean ± std across folds):\n%s", summary.to_string())
    summary.to_csv(PROCESSED_DIR / "metrics_summary.csv")

    if args.skip_shap or results["last_model"] is None:
        logging.info("Stage 6/6 — SHAP step skipped")
    else:
        logging.info("Stage 6/6 — computing SHAP summary on the final-fold test set")
        out = shap_summary(results["last_model"], results["last_test"])
        logging.info("  saved %s", out.relative_to(ROOT))

    logging.info("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
