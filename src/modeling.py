"""Walk-forward training, evaluation and explanation.

The split strategy is date-based so a single calendar day is never
shared between a train and test fold (which can happen with the default
``sklearn.model_selection.TimeSeriesSplit`` on panel data sorted by
date). A custom :func:`walk_forward_splits` carves the modeling year
into ``N_CV_FOLDS`` rolling test windows of equal length.

For each fold we report:

* ``roc_auc`` — discrimination on a balanced threshold-independent scale.
* ``pr_auc``  — discrimination weighted toward the positive class.
* ``brier``   — calibration of the predicted probabilities.

The baseline is a per-vehicle frequency model: for each test row, the
predicted probability is the fraction of training-window days on which
that vehicle had an unscheduled WO in the next 7 days. New vehicles
fall back to the fleet base rate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from config import FIGURES_DIR, N_CV_FOLDS, RANDOM_SEED

logger = logging.getLogger(__name__)

LABEL_COL = "y_unscheduled_next7"
ID_COLS = ["vehicle", "date"]


# ---------------------------------------------------------------------------
# Walk-forward splits
# ---------------------------------------------------------------------------


@dataclass
class FoldSpec:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walk_forward_splits(
    dates: pd.Series,
    n_folds: int = N_CV_FOLDS,
    min_train_days: int = 60,
) -> Iterator[FoldSpec]:
    """Yield expanding-window train/test splits over the date range.

    The first ``min_train_days`` days are reserved for the initial
    training set; the remainder is split into ``n_folds`` test windows
    of equal length. Fold ``k`` trains on everything before the start
    of its test window.
    """
    dates = pd.to_datetime(dates)
    d_min, d_max = dates.min(), dates.max()
    initial_train_end = d_min + pd.Timedelta(days=min_train_days - 1)
    test_total_days = (d_max - initial_train_end).days
    test_window = test_total_days // n_folds

    for k in range(n_folds):
        test_start = initial_train_end + pd.Timedelta(days=1 + k * test_window)
        if k == n_folds - 1:
            test_end = d_max
        else:
            test_end = test_start + pd.Timedelta(days=test_window - 1)
        train_end = test_start - pd.Timedelta(days=1)
        yield FoldSpec(
            fold=k + 1,
            train_start=d_min,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS + [LABEL_COL]]


def frequency_baseline_predict(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Predicted probability = per-vehicle base rate in the training window."""
    per_vehicle = train.groupby("vehicle")[LABEL_COL].mean()
    fleet_rate = train[LABEL_COL].mean()
    return test["vehicle"].map(per_vehicle).fillna(fleet_rate).to_numpy()


def fit_gbm(train: pd.DataFrame) -> HistGradientBoostingClassifier:
    X = train[feature_columns(train)]
    y = train[LABEL_COL]
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    )
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "brier": brier_score_loss(y_true, y_score),
    }


def run_walk_forward(features: pd.DataFrame) -> dict:
    """Run walk-forward CV, return per-fold metrics and the last-fold model."""
    df = features.sort_values("date").reset_index(drop=True)
    rows = []
    last_model = None
    last_test = None

    for spec in walk_forward_splits(df["date"]):
        train = df[(df["date"] >= spec.train_start) & (df["date"] <= spec.train_end)]
        test = df[(df["date"] >= spec.test_start) & (df["date"] <= spec.test_end)]
        if len(train) == 0 or len(test) == 0 or test[LABEL_COL].nunique() < 2:
            logger.warning("Fold %d skipped — empty or single-class test set", spec.fold)
            continue

        baseline_pred = frequency_baseline_predict(train, test)
        model = fit_gbm(train)
        gbm_pred = model.predict_proba(test[feature_columns(test)])[:, 1]

        for name, pred in [("baseline_freq", baseline_pred), ("gbm", gbm_pred)]:
            metrics = evaluate(test[LABEL_COL].to_numpy(), pred)
            rows.append({
                "fold": spec.fold,
                "model": name,
                "train_start": spec.train_start.date(),
                "train_end": spec.train_end.date(),
                "test_start": spec.test_start.date(),
                "test_end": spec.test_end.date(),
                "n_train": len(train),
                "n_test": len(test),
                "pos_rate_test": test[LABEL_COL].mean(),
                **metrics,
            })
            logger.info(
                "fold %d  %-13s  ROC-AUC=%.3f  PR-AUC=%.3f  Brier=%.3f  (test=%s..%s, n=%d)",
                spec.fold, name, metrics["roc_auc"], metrics["pr_auc"], metrics["brier"],
                spec.test_start.date(), spec.test_end.date(), len(test),
            )
        last_model = model
        last_test = test

    metrics_df = pd.DataFrame(rows)
    summary = (
        metrics_df.groupby("model")[["roc_auc", "pr_auc", "brier"]]
        .agg(["mean", "std"])
        .round(3)
    )
    return {
        "per_fold": metrics_df,
        "summary": summary,
        "last_model": last_model,
        "last_test": last_test,
    }


# ---------------------------------------------------------------------------
# SHAP explanation
# ---------------------------------------------------------------------------


def shap_summary(
    model: HistGradientBoostingClassifier,
    test: pd.DataFrame,
    out_path: Path = FIGURES_DIR / "16_shap_summary.png",
    max_display: int = 15,
    sample_size: int = 2000,
) -> Path:
    """Save a SHAP beeswarm summary for the model on a test-set sample."""
    import shap
    import matplotlib.pyplot as plt

    X = test[feature_columns(test)]
    if len(X) > sample_size:
        X = X.sample(sample_size, random_state=RANDOM_SEED)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
