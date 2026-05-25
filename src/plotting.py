"""Plotting helpers shared by the EDA notebook and the modeling reports.

These are deliberately thin wrappers around matplotlib / seaborn. The goal
is consistent styling across the deliverable (one set of fonts, sizes, and
colors) and the ability to save each figure into ``reports/figures`` with
a single line of code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import seaborn as sns

from config import FIGURES_DIR


# Colorblind-friendly categorical palette (Okabe-Ito ordering).
PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#000000",  # black
]


def setup_style() -> None:
    """Apply a consistent matplotlib/seaborn style for the whole project."""
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        palette=PALETTE,
        rc={
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )


def save_fig(fig: plt.Figure, name: str, subdir: str | None = None) -> Path:
    """Save ``fig`` under ``reports/figures[/subdir]/<name>.png`` and return the path."""
    target_dir = FIGURES_DIR if subdir is None else FIGURES_DIR / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / f"{name}.png"
    fig.savefig(out)
    return out


def horizontal_bar(
    counts,
    *,
    title: str,
    xlabel: str = "Count",
    top_n: int | None = 20,
    truncate_labels: int | None = 80,
    figsize: tuple[float, float] = (9, 6),
) -> plt.Figure:
    """Plot a horizontal bar chart from a Series of value counts.

    Designed for long categorical labels (e.g. fault descriptions).
    """
    s = counts.sort_values(ascending=True)
    if top_n is not None:
        s = s.tail(top_n)
    if truncate_labels is not None:
        s = s.rename(
            index=lambda x: (str(x)[: truncate_labels - 1] + "\u2026")
            if isinstance(x, str) and len(str(x)) > truncate_labels
            else x
        )

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(s.index.astype(str), s.values, color=PALETTE[0])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig
