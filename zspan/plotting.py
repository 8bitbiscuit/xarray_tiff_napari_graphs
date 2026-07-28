"""Figures for comparing segmentation runs by how far their labels span in z.

* :func:`plot_variant_summary` -- which variant wins?  (one bar per variant in a
  single hue, with per-volume dots, because with a handful of volumes per
  variant the spread is the story)
* :func:`plot_span_distribution` -- where does the difference live?  (share of
  labels at each z-span, categorical hues because the variants are the subject)

Colours come from a validated palette, exported here so notebook-side figures
stay consistent with these: :data:`CATEGORICAL` is a fixed-order set whose
adjacent pairs clear colour-vision-deficiency thresholds -- assign slots in
order and never generate new ones -- while :data:`SEQUENTIAL_BLUE` and
:data:`BLUES` are the single-hue ramp for magnitude and for *ordinal* scales
such as mask thickness.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

__all__ = [
    "BLUES",
    "CATEGORICAL",
    "SEQUENTIAL_BLUE",
    "apply_theme",
    "plot_variant_summary",
    "plot_span_distribution",
]

# Fixed slot order -- the ordering itself is the CVD-safety mechanism.
CATEGORICAL = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

# Single-hue ramp, light -> dark, for continuous magnitude.
SEQUENTIAL_BLUE = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e3e2de"

BLUES = LinearSegmentedColormap.from_list("zspan_blues", SEQUENTIAL_BLUE)


def apply_theme() -> None:
    """Recessive axes, quiet grid, readable type.  Safe to call repeatedly."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
            "figure.dpi": 110,
        }
    )


def _categorical(groups: Sequence[str]) -> dict[str, str]:
    """Map group -> hue by fixed slot order, so a hue always means one entity."""
    return {group: CATEGORICAL[i] for i, group in enumerate(groups)}


def _limit_groups(groups: list[str], limit: int, what: str) -> list[str]:
    if len(groups) <= limit:
        return groups
    warnings.warn(
        f"{len(groups)} {what} exceeds the {limit}-colour ceiling; showing the "
        f"first {limit}. Subset explicitly or facet rather than adding hues.",
        stacklevel=3,
    )
    return groups[:limit]


# --------------------------------------------------------------------------- #
# 1. which variant wins?
# --------------------------------------------------------------------------- #


def plot_variant_summary(
    summary: pd.DataFrame,
    *,
    group: str = "variant",
    value: str = "pct_multi_layer",
    title: str = "Masks spanning multiple z-layers, by segmentation variant",
    value_label: str = "% of masks spanning multiple z-layers",
    show_points: bool = True,
    highlight: str | None = None,
) -> Figure:
    """One bar per variant, sorted, in a single hue.

    Deliberately *not* colour-ramped by value -- bar length already encodes
    magnitude, so a ramp would spend the colour channel on nothing.  Individual
    volumes are overlaid as dots because a mean alone hides the spread.
    """
    if group not in summary.columns:
        raise KeyError(f"column {group!r} not in summary; have {list(summary.columns)}")

    stats = (
        summary.groupby(group)[value]
        .agg(["mean", "count"])
        .sort_values("mean")
        .reset_index()
    )
    positions = np.arange(len(stats))
    colors = [
        CATEGORICAL[0] if (highlight is None or name == highlight) else INK_MUTED
        for name in stats[group]
    ]

    data_max = max(100.0, float(summary[value].max()))
    label_x = data_max * 1.26  # reserved column, so values never sit on the marks

    fig, ax = plt.subplots(figsize=(8.6, 0.66 * len(stats) + 2.0))
    ax.barh(positions, stats["mean"], height=0.5, color=colors, zorder=2)

    if show_points:
        rng = np.random.default_rng(0)
        for i, name in enumerate(stats[group]):
            points = summary.loc[summary[group] == name, value].to_numpy(float)
            ax.scatter(
                points,
                i + rng.uniform(-0.16, 0.16, points.size),
                s=26, facecolor=INK, edgecolor=SURFACE,  # surface ring keeps
                linewidth=1.1, zorder=3,                 # overlapping dots apart
            )

    for i, (mean, count) in enumerate(zip(stats["mean"], stats["count"])):
        ax.text(label_x, i, f"{mean:.1f}%  (n={count})", va="center", ha="right",
                fontsize=9, color=INK_SECONDARY, zorder=4)

    ax.set_yticks(positions, stats[group])
    ax.set_xlabel(value_label)
    ax.set_xlim(0, label_x)
    # Keep ticks on the data range; the reserved label column is not measurable.
    ax.set_xticks([t for t in ax.get_xticks() if 0 <= t <= data_max])
    ax.set_title(title, pad=10, loc="left")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_visible(False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# 2. where does the difference live?
# --------------------------------------------------------------------------- #


def plot_span_distribution(
    labels: pd.DataFrame,
    *,
    group: str = "variant",
    max_span: int = 5,
    title: str = "Distribution of per-mask z-span",
    order: Sequence[str] | None = None,
) -> Figure:
    """Share of labels at each z_span, grouped bars, one hue per variant.

    Normalised within each variant so runs with different label counts compare
    directly.  The final bin is inclusive ("5+").
    """
    if group not in labels.columns:
        raise KeyError(f"column {group!r} not in labels; have {list(labels.columns)}")
    if "z_span" not in labels.columns:
        raise KeyError("labels frame needs a 'z_span' column")

    groups = list(order) if order is not None else sorted(labels[group].unique())
    groups = _limit_groups(groups, len(CATEGORICAL), "groups")
    colors = _categorical(groups)

    binned = labels["z_span"].clip(upper=max_span)
    bins = np.arange(1, max_span + 1)
    tick_labels = [str(b) for b in bins[:-1]] + [f"{max_span}+"]

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    width = 0.8 / len(groups)

    for i, name in enumerate(groups):
        mask = labels[group] == name
        counts = np.array([(binned[mask] == b).sum() for b in bins], float)
        total = counts.sum()
        share = 100.0 * counts / total if total else counts
        offset = (i - (len(groups) - 1) / 2) * width
        ax.bar(
            bins + offset, share, width=width * 0.88,  # gap keeps fills distinct
            color=colors[name], label=f"{name}  (n={int(total):,})", zorder=2,
        )

    ax.set_xticks(bins, tick_labels)
    ax.set_xlabel("z-span of mask (planes, inclusive)")
    ax.set_ylabel("% of masks in variant")
    ax.set_title(title, pad=10, loc="left")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper right", ncols=1)
    fig.tight_layout()
    return fig
