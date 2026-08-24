"""The scorecard rubric, one function per line.

Every rubric line has a scorer that takes a FOV and returns a number, and a plot
that draws the distribution behind it.

    Span    1 - share of masks a ``min_z`` filter would discard
    Jitter  1 - share of masks whose area-through-z profile wanders up and down
    Sway    1 - share whose profile jumps once and holds (an over-merge)
    Signal  P(a random masked voxel outshines a random unmasked one)
    Total   the four, weighted, out of 1.00

Masks living on a single z plane are imaging artifacts and are dropped before
anything is measured, which also returns their voxels to the unmasked pool.

    fov = load_fov("masks.tif", "DAPI_decon_z*.tif")
    jitter(fov)                                  # 0.94
    plot_jitter(fov)                             # the distribution behind it
    scorecard({"3D true": a, "3D stitched": b})  # every line, one row per FOV
    plot_total({"3D true": a, "3D stitched": b})
"""

from __future__ import annotations

import glob
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.figure import Figure

# Cutoffs are absolute, never quantiles of the FOV they are applied to, so a
# number means the same thing in every field.  The two profile cutoffs are
# provisional -- set them from the knee in plot_jitter / plot_sway.
THIN_MAX = 2          # z_span <= this is what a min_z = 3 filter discards
JITTER_CUTOFF = 1.75  # total reversal travel, as a factor; ordinary masks sit at 1.1-1.3
SWAY_CUTOFF = 3.0     # worst corner, as a factor; a healthy arc ~2, a stitching step 10+
AREA_FLOOR = 20       # px; ratios between handfuls of pixels are noise
SLIVER_FRAC = 0.05    # a plane under this share of the mask's own peak is a partial volume

WEIGHTS = {"Span": 0.10, "Jitter": 0.50, "Sway": 0.10, "Signal": 0.30}

PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
FLAG, INK, GRID = "#e34948", "#52514e", "#d8d7d3"


# =============================================================================
# The data one FOV is scored from
# =============================================================================

@dataclass
class FOV:
    """One mask volume, and the DAPI stack it was segmented from."""

    masks: np.ndarray
    dapi: np.ndarray | None = None
    name: str = "fov"

    @cached_property
    def areas(self) -> np.ndarray:
        """(n_z, n_labels + 1) pixel count per label per plane, single-slice masks removed."""
        n = int(self.masks.max()) + 1
        areas = np.stack([np.bincount(plane.ravel(), minlength=n) for plane in self.masks])
        areas[:, 0] = 0                                   # background is not a mask
        areas[:, (areas > 0).sum(axis=0) == 1] = 0        # nor is a one-plane artifact
        return areas

    @cached_property
    def kept(self) -> np.ndarray:
        """The volume the artifacts have been zeroed out of."""
        return np.where((self.areas > 0).any(axis=0)[self.masks], self.masks, 0)

    @cached_property
    def hist(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(masked, unmasked, edges) — voxel brightness split by mask presence."""
        values = self.dapi.ravel()
        inside = (self.kept > 0).ravel()
        if np.issubdtype(values.dtype, np.integer):
            n = int(values.max()) + 1
            total = np.bincount(values, minlength=n)
            masked = np.bincount(values[inside], minlength=n)
            edges = np.arange(n + 1) - 0.5
        else:
            edges = np.linspace(values.min(), values.max(), 4097)
            total, _ = np.histogram(values, edges)
            masked, _ = np.histogram(values[inside], edges)
        return masked, total - masked, edges

    @cached_property
    def table(self) -> pd.DataFrame:
        """One row per surviving mask: its z-span, its jitter and its sway."""
        return mask_table(self)


def load_fov(masks_path, dapi_glob=None, name=None) -> FOV:
    """Read a mask volume, plus the DAPI planes matching ``dapi_glob`` if given."""
    masks = tifffile.imread(str(masks_path))
    masks = masks[None] if masks.ndim == 2 else masks

    dapi = None
    if dapi_glob:
        def z_of(f):
            m = re.search(r"z(\d+)", Path(f).stem)      # z10 sorts after z9
            return int(m.group(1)) if m else 0
        files = sorted(glob.glob(str(dapi_glob)), key=z_of)
        if not files:
            raise FileNotFoundError(f"no DAPI files matched {dapi_glob}")
        dapi = np.stack([tifffile.imread(f) for f in files])
        if dapi.shape != masks.shape:
            raise ValueError(f"masks {masks.shape} != dapi {dapi.shape}")

    return FOV(masks, dapi, name or Path(masks_path).parent.name)


def mask_table(fov: FOV, area_floor=AREA_FLOOR, sliver_frac=SLIVER_FRAC) -> pd.DataFrame:
    """Per-mask z-span, jitter and sway, all read off the per-plane areas.

    A plane holding under ``area_floor`` px, or under ``sliver_frac`` of the
    mask's own peak, is a partial-volume sliver and is left out of the profile —
    every nucleus enters and leaves the stack through one.  Both profile metrics
    need three consecutive surviving planes and are NaN without them.
    """
    areas = fov.areas
    present = areas > 0
    labels = np.flatnonzero(present.any(axis=0))

    z = np.arange(areas.shape[0])[:, None]
    z_span = (np.where(present, z, -1).max(axis=0)
              - np.where(present, z, areas.shape[0]).min(axis=0) + 1)

    keep = present & (areas >= area_floor) & (areas >= sliver_frac * areas.max(axis=0))
    log_area = np.log10(np.maximum(areas, 1))

    step = keep[:-1] & keep[1:]                       # both ends of this step survive
    triple = step[:-1] & step[1:]                     # three consecutive kept planes
    measurable = triple.any(axis=0)

    # jitter: how far the profile travels back on itself.  d is per-step log growth;
    # a reversal is worth the smaller of its two limbs, so a ramp costs nothing and
    # many small wobbles outscore one large excursion.
    d = np.where(step, log_area[1:] - log_area[:-1], 0.0)
    reversed_here = triple & (np.sign(d[:-1]) * np.sign(d[1:]) < 0)
    depth = np.where(reversed_here, np.minimum(np.abs(d[:-1]), np.abs(d[1:])), 0.0)
    jitter = np.where(measurable, 10.0 ** depth.sum(axis=0), np.nan)

    # sway: the sharpest corner, as a second difference of log area.  A flat run,
    # a jump, another flat run is monotone, so jitter cannot see it and this can.
    curve = np.where(triple, np.abs(log_area[2:] - 2 * log_area[1:-1] + log_area[:-2]), -1.0)
    sway = np.where(measurable, 10.0 ** curve.max(axis=0, initial=-1.0), np.nan)

    return pd.DataFrame({
        "label": labels,
        "z_span": z_span[labels],
        "n_planes": present.sum(axis=0)[labels],
        "area_max": areas.max(axis=0)[labels],
        "jitter": jitter[labels],
        "sway": sway[labels],
    })


# =============================================================================
# The rubric: one scorer per line
# =============================================================================

def _flagged(values: pd.Series, cutoff: float) -> float:
    """Share over the cutoff, among the masks long enough to have the metric."""
    values = values.dropna()
    return float((values > cutoff).mean()) if len(values) else np.nan


def span(fov: FOV, thin_max: float = THIN_MAX) -> float:
    """Share of masks a ``min_z`` filter would keep."""
    return 1.0 - float((fov.table["z_span"] <= thin_max).mean())


def jitter(fov: FOV, cutoff: float = JITTER_CUTOFF) -> float:
    """Share of masks whose area profile does not wander up and down."""
    return 1.0 - _flagged(fov.table["jitter"], cutoff)


def sway(fov: FOV, cutoff: float = SWAY_CUTOFF) -> float:
    """Share of masks whose area profile has no over-merge corner in it."""
    return 1.0 - _flagged(fov.table["sway"], cutoff)


def signal(fov: FOV) -> float:
    """P(a random masked voxel is brighter than a random unmasked one).

    Rank-based, so the two groups being wildly different sizes costs nothing.
    0.5 is a technique whose masks say nothing about brightness — which is the
    floor this line scores on, not 0.
    """
    masked, unmasked, _ = fov.hist
    if not masked.sum() or not unmasked.sum():
        return np.nan
    strictly_below = np.concatenate([[0.0], np.cumsum(unmasked)[:-1]])
    wins = (masked * (strictly_below + 0.5 * unmasked)).sum()
    return float(wins / (masked.sum() * unmasked.sum()))


def total(fov: FOV, weights: dict | None = None) -> float:
    """The four lines, weighted, out of 1.00.  Not clipped: -0.10 is not 0.00."""
    return scores(fov, weights)["Total"]


def scores(fov: FOV, weights: dict | None = None) -> pd.Series:
    """Every rubric line for one FOV, plus the ``Total`` they earn."""
    weights = pd.Series(WEIGHTS if weights is None else weights, dtype=float)
    line = pd.Series({"Span": span(fov), "Jitter": jitter(fov), "Sway": sway(fov),
                      "Signal": signal(fov) if fov.dapi is not None else np.nan})
    # a missing line drops out of both the sum and the weight it would have
    # carried, so a scan without DAPI still totals out of 1.00
    have = line.notna()
    line["Total"] = ((line[have] * weights[have]).sum() / weights[have].sum()
                     if have.any() else np.nan)
    return line


def scorecard(fovs, weights: dict | None = None) -> pd.DataFrame:
    """One row per FOV, one column per rubric line."""
    return pd.DataFrame({name: scores(fov, weights)
                         for name, fov in _named(fovs).items()}).T


# =============================================================================
# One plot per line
# =============================================================================

def _named(fovs) -> dict:
    """A single FOV, a list of them or a ``{name: FOV}`` mapping — always a mapping."""
    if isinstance(fovs, FOV):
        return {fovs.name: fovs}
    if isinstance(fovs, dict):
        return dict(fovs)
    return {fov.name: fov for fov in fovs}


def _axes(figsize=(8.2, 4.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=9, colors=INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    return fig, ax


def _finish(ax, title, xlabel, ylabel="share of masks"):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK)
    ax.legend(frameon=False, fontsize=9)
    ax.figure.tight_layout()
    return ax.figure


def plot_span(fovs, thin_max: float = THIN_MAX) -> Figure:
    """Where the masks sit in z; the shaded bars are what a ``min_z`` filter deletes."""
    fovs = _named(fovs)
    fig, ax = _axes()
    top = max(int(f.table["z_span"].max()) for f in fovs.values())
    centres = np.arange(1, top + 1)
    width = 0.8 / len(fovs)

    for i, (name, fov) in enumerate(fovs.items()):
        share = np.bincount(fov.table["z_span"], minlength=top + 1)[1:] / len(fov.table)
        offset = (i - (len(fovs) - 1) / 2) * width
        ax.bar(centres + offset, share, width=width, color=PALETTE[i % len(PALETTE)],
               label=f"{name}  (span {span(fov, thin_max):.2f})")

    ax.axvspan(0.5, thin_max + 0.5, color=FLAG, alpha=0.08, linewidth=0, zorder=0)
    ax.set_xticks(centres)
    return _finish(ax, "How far each mask reaches through z",
                   f"z_span in planes  ·  shaded: thin, <= {thin_max:g}")


def _plot_profile(fovs, column: str, cutoff: float, title: str, xlabel: str) -> Figure:
    """Log-binned per-mask ``jitter`` or ``sway``, one step outline per FOV."""
    fovs = _named(fovs)
    fig, ax = _axes()
    pooled = np.concatenate([f.table[column].dropna().to_numpy() for f in fovs.values()])
    top = max(pooled.max(initial=1.1), cutoff * 1.3)
    bins = np.logspace(0, np.log10(top), 46)
    centres = np.sqrt(bins[:-1] * bins[1:])

    for i, (name, fov) in enumerate(fovs.items()):
        values = fov.table[column].dropna().to_numpy()
        if not values.size:
            continue
        share = np.histogram(np.clip(values, 1.0, top), bins=bins)[0] / values.size
        color = PALETTE[i % len(PALETTE)]
        ax.fill_between(centres, share, step="mid", color=color, alpha=0.16, linewidth=0)
        ax.step(centres, share, where="mid", color=color, linewidth=2, zorder=3,
                label=f"{name}  (n={values.size:,}, {100 * _flagged(fov.table[column], cutoff):.0f}% over)")

    ax.axvspan(cutoff, top, color=FLAG, alpha=0.07, linewidth=0, zorder=0)
    ax.axvline(cutoff, color=FLAG, linestyle="--", linewidth=1.4, zorder=4,
               label=f"cutoff = {cutoff:g}x")
    ax.set_xscale("log")
    ax.set_xlim(1.0, top)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:g}x"))
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    return _finish(ax, title, xlabel)


def plot_jitter(fovs, cutoff: float = JITTER_CUTOFF) -> Figure:
    """The jitter distribution: 1.0x is a profile that never turns around."""
    return _plot_profile(fovs, "jitter", cutoff,
                         "How much each mask's area profile wanders up and down",
                         "jitter = total reversal travel, as a factor (log scale)")


def plot_sway(fovs, cutoff: float = SWAY_CUTOFF) -> Figure:
    """The sway distribution: 1.0x is a profile whose growth rate never changes."""
    return _plot_profile(fovs, "sway", cutoff,
                         "How sharp the worst corner in each area profile is",
                         "sway = worst change in growth rate, as a factor (log scale)")


def _display_bins(masked, unmasked, edges, n_bins=120, clip_quantile=0.999):
    """Collapse the fine histogram onto readable bins, cut at the clip quantile."""
    pooled = masked + unmasked
    centres = 0.5 * (edges[:-1] + edges[1:])
    cut = min(int(np.searchsorted(np.cumsum(pooled) / pooled.sum(), clip_quantile)),
              centres.size - 1)
    new = np.linspace(edges[0], centres[cut], n_bins + 1)
    return (np.histogram(centres, new, weights=masked)[0],
            np.histogram(centres, new, weights=unmasked)[0], new)


def plot_signal(fovs, clip_quantile: float = 0.999) -> Figure:
    """Voxel brightness inside and outside the masks, one panel per FOV."""
    fovs = _named(fovs)
    fig, axes = plt.subplots(len(fovs), 1, figsize=(8.2, 2.6 * len(fovs)), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, (name, fov) in zip(axes, fovs.items()):
        masked, unmasked, edges = _display_bins(*fov.hist, clip_quantile=clip_quantile)
        centres = 0.5 * (edges[:-1] + edges[1:])
        for counts, color, label in ((unmasked, PALETTE[1], "outside every mask"),
                                     (masked, PALETTE[0], "inside a mask")):
            share = counts / max(counts.sum(), 1)
            ax.fill_between(centres, share, step="mid", color=color, alpha=0.35, linewidth=0)
            ax.step(centres, share, where="mid", color=color, linewidth=1.6, label=label)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_title(f"{name}  ·  separability {signal(fov):.3f}",
                     loc="left", fontsize=10, fontweight="bold", pad=8)
        ax.set_ylabel("share of voxels", fontsize=9.5, color=INK)
        ax.tick_params(length=0, labelsize=9, colors=INK)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

    axes[0].legend(frameon=False, fontsize=9)
    axes[-1].set_xlabel("voxel brightness", fontsize=9.5, color=INK)
    fig.tight_layout()
    return fig


def plot_total(fovs, weights: dict | None = None) -> Figure:
    """The scorecard: a bar per line per FOV, with ``Total`` set apart at the right."""
    weights = WEIGHTS if weights is None else weights
    card = scorecard(fovs, weights)
    columns = list(card.columns)
    n_rows, n_cols = len(card), len(columns)

    fig, ax = plt.subplots(figsize=(1.5 * n_cols + 2.2, 0.5 * n_rows + 1.9))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(n_rows - 0.5, -0.5)

    for j, name in enumerate(columns):
        is_total = name == "Total"
        for i, value in enumerate(card[name].to_numpy()):
            if not np.isfinite(value):
                ax.text(j + 0.5, i, "–", ha="center", va="center", color=INK, fontsize=9)
                continue
            drawn = min(max(value, 0.0), 1.0)     # the bar clips, the label does not
            ax.barh(i, 0.9 * drawn, left=j + 0.05, height=0.5, zorder=2,
                    color="#184f95" if is_total else PALETTE[0])
            inside = value >= 0.30
            ax.text(j + 0.05 + 0.9 * drawn + (-0.04 if inside else 0.04), i, f"{value:.2f}",
                    ha="right" if inside else "left", va="center", fontsize=8.5, zorder=3,
                    color="#fcfcfb" if inside else INK,
                    fontweight="bold" if is_total else "normal")
        ax.axvline(j, color=INK if is_total else GRID,
                   linewidth=1.2 if is_total else 0.8, zorder=1)

    for i in range(1, n_rows):
        ax.axhline(i - 0.5, color=GRID, linestyle=":", linewidth=1.0, zorder=0)
    ax.set_xticks(np.arange(n_cols) + 0.5,
                  [c if c == "Total" else f"{c}\n×{weights[c]:.2f}" for c in columns],
                  fontsize=9)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(n_rows), list(card.index), fontsize=9, fontweight="bold")
    ax.tick_params(length=0, colors=INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig
