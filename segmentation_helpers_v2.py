"""Helpers for ``segmentation_comparison_v2.ipynb`` — every script's measurement
on one scorecard.

Three measurements, taken from the repo's scripts and run on the same volumes:

===========================================  ==========================================
z-span (Part 0-1 below)                      how far does each mask reach through z
``segmentation_z_area_sway_claude.py``       does the area-through-z profile wander up
                                             and down, or step in one jump
``segmentation_z_brightness_claude.py``      do the voxels a technique claims actually
                                             look like signal
===========================================  ==========================================

The z-span half is Part 0 below — the palette, the theme, the region walk, the
per-plane areas and the span figure — so this module and the two scripts above
are the whole dependency chain.  The z bounding box is read off the areas array
by :func:`span_table` rather than by a second ``find_objects`` pass, since the
areas are what every other measurement here already needs.

Two scripts are deliberately not aggregated.

``segmentation_z_kanai_only.py`` does ``import napari`` at module scope, which a
headless kernel cannot do, and its z-span maths is the first row of the table.

**Per-transition area change is not measured here at all** — the second half of
``segmentation_z_claude.py`` and the whole of
``segmentation_z_claude_log_cutoff.py``.  |Δ area| / area asks *how much did this
mask's cross-section change between two planes*, which is large for a healthy
nucleus growing steeply toward its mid-plane, so it ranks the healthiest steep
growers above real defects.  ``jitter`` and ``sway`` between them ask the two
questions worth asking — did the profile turn around, and did it jump — and both
score a monotone grower at exactly 1.0 however fast it grows.

**A note on what ``sway`` used to be.**  Until this rework a single metric,
``sway = 10 ** max |Δ² log a|``, stood for both.  It could not tell them apart:
a magnitude of curvature is blind to *direction*, so a violent monotone ramp and
a genuine up-down-up scored alike, and ``max`` made it an extreme-value statistic
that one transition decided — ten alternating wobbles could never outscore one
jump.  Worse, ``AREA_FLOOR`` *clamped* the areas rather than excluding them, so
the partial-volume sliver every nucleus enters and leaves the stack through
stayed in the profile and contributed a ~100x step by construction.  The result
was a flag that ranked masks by onset steepness.  ``jitter`` is the replacement
for the question it was supposed to answer, ``sway`` keeps the name and the old
maths for the over-merge it really did catch, and both now exclude slivers.

This module reduces the three to **one row per (region, FOV, technique)** —
individual statistics, then three group scores, then a weighted total out of
1.00.  It re-implements none of the
measurements: ``check_area_jitter`` / ``check_area_sway``, ``vectorise_pixels`` /
``accumulate_histograms`` and the histogram statistics are imported from the
scripts and called.  What is new here is the three things none of them has.

1. The single-slice filter
--------------------------
A mask living on exactly one z plane is an imaging artifact, not a nucleus.
:func:`filter_single_slice` zeroes those labels out of the volume **before any
metric runs**, so they stop counting as masks *and* their voxels go back into
the unmasked brightness pool — the filter has to happen upstream of everything
or the brightness figures would still be scoring them.

They are produced by per-plane stitching, so ``DROP_SINGLE_SLICE_IN = None``
(the default) applies the filter to *every* technique rather than only to the
one that makes them.  Dropping them from one arm and not the other would move
every rate in that arm's favour — the surviving masks are the thicker ones, so
its thin rate, its median z-span and its jitter rate all improve for free.  Set
``DROP_SINGLE_SLICE_IN = ("3D stitched",)`` for the literal one-arm reading; the
per-technique dropped counts are printed either way, and are carried in the
metrics table (``n_dropped``, ``pct_dropped``) and shown on the scorecard under
its own **Reported** header — listed, never scored, because an artifact of the
imaging is not a defect of the segmenter.  It is on the card because it is the
one number that says how much of each arm's population was removed before any
score was computed, and every score below is read against it.

2. Non-normal statistics, everywhere
------------------------------------
Every distribution here is assumed non-normal, and most of them visibly are: a
jitter is a spike at 1.0x with a sparse tail running out past 100, z-span is a
small integer count, and voxel brightness is bimodal by construction.  So:

* **centre and spread** are the median and the IQR, never the mean and the SD.
  The scripts' means are still computed and still printed, because that is what
  they printed, and ``mean_masked`` is drawn on the card because "how bright is
  a voxel this technique claimed" is worth having in the units of the image —
  but no mean drives a score here.
* **the profile cutoffs** are absolute (``JITTER_CUTOFF``, ``SWAY_CUTOFF``)
  rather than the script's default 90th percentile.  Both are ratios of ratios
  and therefore dimensionless, so a fixed number means the same thing in every
  FOV — where a quantile cutoff flags the top 10% of every FOV *by construction*,
  which makes a fixed-size shortlist to look at and a useless number to compare
  techniques on, since both would score 10.0% every time.  **Both shipped values
  are provisional**, taken from worked profiles rather than from data; derive
  them from :func:`compare_profile_distribution` over a full scan.
* **comparing two techniques** is paired **Wilcoxon signed-rank** across FOVs,
  with the Hodges–Lehmann median shift, the matched-pairs rank-biserial
  correlation, and a percentile **bootstrap** CI — no t-test, no normal-theory
  interval, and the pairing is what handles FOV-to-FOV variation being larger
  than the effect.
* **separability** is the rank-based AUC the brightness script already used,
  under a name that says what it measures — P(a random masked voxel outshines a
  random unmasked one) — plus two rank rates the script did not have: the share
  of masked voxels below the unmasked median, and the share of unmasked voxels
  above the masked median.

3. The scorecard
----------------
The card this replaced scored two groups of columns, Thinness and Stability.
:func:`plot_scorecard` scores four — Span, Jitter, Sway, Signal — and adds them
up as a **rubric**::

    Total = 0.20 * Span + 0.30 * Jitter + 0.20 * Sway + 0.30 * Signal

The weights sum to 1, so a clean run earns 1.00 and every point lost is
traceable to the line that lost it.  They are unequal on purpose and they are
declared in one constant (:data:`SCORE_WEIGHTS`) rather than implied: a corner
in an area profile is a segmenter gluing two objects together, a wrong answer
nothing downstream can repair, so the two profile lines carry 0.50 between them
— 0.30 for a mask that wanders up and down, 0.20 for one clean over-merge step;
brightness separability is real but softer evidence at 0.30; and a thin mask is
the cheapest defect of the three, deleted by a ``min_z`` filter in one line, so
Span carries 0.20.  An equal split would have been a weighting too — this one
just says what it is.

Every subscore is **absolute**: 1.00 is a clean run and no score moves when a
FOV is added to the table.  Span, Jitter and Sway are rates, so each point of
defect costs them one hundredth, and enough defect drives any of them below zero —
only the drawn bar clips at 0, because a 0.00 and a −0.40 are not the same run.

Signal is the odd one: it is the brightness AUC used directly, and an AUC's
no-information point is **0.5**, not 0.  So the Signal line effectively banks
half its 0.30 for any technique whose masks are not actively anti-correlated
with brightness, and the range worth reading on it is about 0.8 to 1.0.  Read a
Total against other Totals rather than as a percentage earned.

**Span pays for the thin rate alone.**  ``pct_deep``, ``pct_z_gap`` and
``median_z_span`` are still measured, still in ``fov_metrics`` and still
available to the paired comparison — they are simply off the card and out of the
score.  A deep mask is a *suspicion* of two nuclei read as one and not a
demonstration of it (a genuinely tall nucleus scores the same), and z-gaps are
rare enough that they moved the score by fractions of a point while taking a
quarter of the card's width.  A thin mask is the one unambiguous defect of the
three: a ``min_z`` filter discards it outright, so its rate is exactly the share
of a technique's output that a downstream pipeline throws away.  Scoring one
clean thing beats scoring three things of mixed sharpness and calling the sum a
measurement.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.figure import Figure
from matplotlib.transforms import blended_transform_factory, offset_copy
from scipy import stats as sps

import segmentation_z_area_sway_claude as sw
import segmentation_z_brightness_claude as b

# CONFIG DEFAULTS
# ----------------------------------------------------------------------------
DAPI_PATTERN = "DAPI_decon_z*.tif"   # one file per z index, inside the FOV directory

# --- the single-slice filter -------------------------------------------------
DROP_SINGLE_SLICE = True
DROP_SINGLE_SLICE_IN = None          # None -> every technique; or e.g. ("3D stitched",)

# --- z-span ------------------------------------------------------------------
THIN_MAX = 2                         # z_span <= this is what a min_z = 3 filter discards
DEEP_MIN = 6                         # z_span >= this is a mask reaching through the stack

# --- area profile: jitter (wobble) and sway (jump) ---------------------------
# Both are dimensionless factors starting at 1.0, so an absolute cutoff means the
# same thing in every FOV.  Both are PROVISIONAL -- worked profiles, not this
# data; derive them from the pooled distributions before trusting them.
JITTER_CUTOFF = 1.75                 # reversal travel; ordinary unimodal masks sit at 1.1-1.3.
                                     # For a mask with a single reversal this says the
                                     # shallower limb of the V is more than a 1.75x area
                                     # change: plateau noise (~1.5x) stays clear, a sharply
                                     # peaked but healthy nucleus (~2.5x) does not
SWAY_CUTOFF = 3.0                    # worst corner; a healthy arc ~2, a stitching step 10+
AREA_FLOOR = 20                      # px; ratios between handfuls of pixels are noise
SLIVER_FRAC = 0.05                   # a plane under this share of the mask's peak area is a
                                     # partial-volume sliver and is left out of the profile

# --- brightness --------------------------------------------------------------
OVERLAP_QUANTILE = 0.5               # the napari bright-unmasked / dim-masked thresholds
# ----------------------------------------------------------------------------


# =============================================================================
# Part 0 -- palette, theme, the region walk, and the per-plane areas
# =============================================================================
# The figure style and the file walk, so this module carries its own.  A
# technique keeps its hue across every figure the repo draws, which is what
# makes a colour mean the same thing in all of them.

# Fixed slot order is the colour-vision-deficiency safety mechanism: assign
# slots in order, never generate a new hue.
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

# Single-hue ramp, light -> dark, for magnitude.
SEQUENTIAL_BLUE = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e3e2de"

MARKERS = ("o", "s", "^", "D")  # identity also travels on shape, not colour alone


def apply_theme() -> None:
    """Recessive axes, quiet grid, readable type.  Safe to call repeatedly."""
    mpl.rcParams.update({
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
    })


def method_colors(methods) -> dict:
    """Hue per technique, fixed slot order.

    A technique keeps its colour across every figure in the repo, so a colour
    never changes meaning between them.
    """
    return {name: CATEGORICAL[i] for i, name in enumerate(methods)}


def tag(frame: pd.DataFrame, method: str, column: str = "method") -> pd.DataFrame:
    """Copy of ``frame`` carrying the technique it came from as a column."""
    out = frame.copy()
    out[column] = method
    return out


def measure_slice_areas(mask_volume) -> np.ndarray:
    """Pixel count of every label in every z slice -> array of shape (n_z, n_labels + 1).

    One bincount pass per plane, and the array it returns is what the span table,
    the single-slice filter and both profile metrics all read — so the volume is
    scanned once and every measurement below comes off the result.
    """
    n_labels = int(mask_volume.max())
    areas = np.zeros((mask_volume.shape[0], n_labels + 1), dtype=np.int64)

    for z in range(mask_volume.shape[0]):
        counts = np.bincount(mask_volume[z].ravel(), minlength=n_labels + 1)
        areas[z] = counts[:n_labels + 1]

    return areas


def available_regions(technique_roots, region_glob="region_*") -> list:
    """Region directories present under *every* technique root, sorted."""
    per_root = [{d.name for d in Path(root).glob(region_glob) if d.is_dir()}
                for root in technique_roots.values()]
    return sorted(set.intersection(*per_root)) if per_root else []


def find_region_fovs(technique_roots, region, fov_glob="fov_*",
                     filename="masks.tif", paired_only=True) -> pd.DataFrame:
    """Locate ``<technique_root>/<region>/<fov>/masks.tif`` for every FOV found.

    ``technique_roots`` maps a technique name to the directory *above* the
    region, i.e. ``data/segmentations_3d_true/cpdino/decon/VePo``.  Returns one
    row per (technique, FOV) with the mask path; with ``paired_only`` a FOV that
    only one technique produced is dropped rather than compared against nothing.
    """
    rows = []
    for method, root in technique_roots.items():
        for fov_dir in sorted((Path(root) / region).glob(fov_glob)):
            path = fov_dir / filename
            if path.exists():
                rows.append({"method": method, "fov": fov_dir.name, "path": path})

    found = pd.DataFrame(rows, columns=["method", "fov", "path"])
    if found.empty:
        raise FileNotFoundError(
            f"no {filename} under any {list(technique_roots.values())}/{region}/{fov_glob}")

    if paired_only:
        per_fov = found.groupby("fov")["method"].nunique()
        complete = set(per_fov[per_fov == len(technique_roots)].index)
        dropped = sorted(set(found["fov"]) - complete)
        if dropped:
            print(f"skipping {len(dropped)} FOV(s) missing from a technique: "
                  f"{', '.join(dropped)}")
        found = found[found["fov"].isin(complete)]

    # technique order follows technique_roots, not the alphabet, so the table
    # rows pair up the same way in every region
    found["method"] = pd.Categorical(found["method"], list(technique_roots), ordered=True)
    return found.sort_values(["fov", "method"]).reset_index(drop=True)


def span_distribution(labels: pd.DataFrame, group: str = "method",
                      max_span: int = 6) -> pd.DataFrame:
    """Share of masks (%) at each z-span, per technique.  Last bin is inclusive."""
    binned = labels["z_span"].clip(upper=max_span)
    counts = (
        pd.crosstab(labels[group], binned, normalize="index")
        # first-appearance order, so a technique keeps its hue slot across figures
        .reindex(index=list(dict.fromkeys(labels[group])))
        .reindex(columns=range(1, max_span + 1), fill_value=0.0)
        * 100.0
    )
    counts.columns = [str(c) for c in counts.columns[:-1]] + [f"{max_span}+"]
    return counts


def compare_span_distribution(labels: pd.DataFrame, group: str = "method",
                              max_span: int = 6,
                              title: str = "How far each mask reaches through z") -> Figure:
    """Share of masks at each z-span, grouped bars, one hue per technique.

    Normalised within technique so runs with different mask counts compare
    directly; the tall bar at z_span=1 is the share a ``min_z`` filter throws away.
    """
    shares = span_distribution(labels, group=group, max_span=max_span)
    methods = list(shares.index)
    colors = method_colors(methods)
    x = np.arange(shares.shape[1])
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for i, name in enumerate(methods):
        offset = (i - (len(methods) - 1) / 2) * width
        n = int((labels[group] == name).sum())
        ax.bar(x + offset, shares.loc[name].to_numpy(), width=width * 0.88,
               color=colors[name], label=f"{name}  (n={n:,})", zorder=2)

    ax.set_xticks(x, list(shares.columns))
    ax.set_xlabel("z-span of mask (planes, inclusive)")
    ax.set_ylabel("% of masks in technique")
    ax.set_title(title, pad=10, loc="left")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# =============================================================================
# Part 1 -- z-span, z-gaps, and the single-slice filter
# =============================================================================

SPAN_COLUMNS = ["label", "z_start", "z_end", "z_span", "n_planes", "has_z_gap",
                "area_min", "area_median", "area_max", "n_voxels"]


def span_table(areas: np.ndarray) -> pd.DataFrame:
    """One row per label, read off the per-plane areas rather than the volume.

    ``z_start``/``z_end``/``z_span`` are the z bounding box, read off
    :func:`measure_slice_areas` output rather than from ``find_objects``, because
    the same array is what the single-slice filter and the profile metrics need,
    and one pass over the volume is enough for all.

    ``n_planes`` is how many planes the label actually occupies.  It differs from
    ``z_span`` only when the label has a hole in z, and ``has_z_gap`` flags that:
    a mask that vanishes for a plane and comes back is two objects stitched into
    one, which is a harder error than plain over-merging and one the z bounding
    box alone cannot show.
    """
    present = areas > 0
    labels = np.flatnonzero(present.any(axis=0))
    labels = labels[labels > 0]                       # drop background
    if labels.size == 0:
        return pd.DataFrame(columns=SPAN_COLUMNS)

    n_z = areas.shape[0]
    z_index = np.arange(n_z, dtype=np.int64)[:, None]
    z_start = np.where(present, z_index, n_z).min(axis=0)
    z_end = np.where(present, z_index, -1).max(axis=0)
    n_planes = present.sum(axis=0)

    # Label ids are sparse (a filtered volume has holes in its numbering), so the
    # area statistics are taken on the used columns only -- an unused column is
    # all-absent, and a median of nothing is a warning, not a number.
    used, seen = areas[:, labels], present[:, labels]
    on_plane = np.where(seen, used, np.nan).astype(float)   # absent != zero area
    absent_high = np.where(seen, used, np.iinfo(np.int64).max)
    span = (z_end - z_start + 1)[labels]

    return pd.DataFrame({
        "label": labels.astype(np.int64),
        "z_start": z_start[labels],
        "z_end": z_end[labels],
        "z_span": span,
        "n_planes": n_planes[labels],
        "has_z_gap": n_planes[labels] < span,
        "area_min": absent_high.min(axis=0),
        "area_median": np.nanmedian(on_plane, axis=0),
        "area_max": used.max(axis=0),
        "n_voxels": used.sum(axis=0),
    })


def drop_labels(masks: np.ndarray, labels) -> np.ndarray:
    """Copy of ``masks`` with ``labels`` zeroed.  Surviving labels keep their ids.

    Ids are deliberately not compacted: a label number in this table is still the
    label number in the napari layer, so a flagged mask can be found by eye.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if labels.size == 0:
        return masks
    return np.where(np.isin(masks, labels), 0, masks).astype(masks.dtype)


def filter_single_slice(masks: np.ndarray, areas: np.ndarray | None = None,
                        drop: bool = DROP_SINGLE_SLICE):
    """Zero every mask that lives on exactly one z plane.

    Returns ``(kept, dropped, areas, report)`` — the filtered volume, a volume
    holding *only* the removed labels (``None`` when nothing was removed, so an
    all-zero volume is never allocated), the per-plane areas of the filtered
    volume, and a dict of counts.

    Both volumes are returned rather than a boolean mask because every downstream
    metric wants the filtered one and the viewer wants the other, and rebuilding
    either from a label list means another pass over the volume.
    """
    if areas is None:
        areas = measure_slice_areas(masks)

    spans = span_table(areas)
    single = spans.loc[spans["z_span"] == 1, "label"].to_numpy()
    n_total = len(spans)

    report = {"n_total": n_total, "n_dropped": int(single.size) if drop else 0,
              "n_single_slice": int(single.size),
              "pct_dropped": (100.0 * single.size / n_total) if n_total else np.nan}

    if not drop or single.size == 0:
        report["n_dropped"] = 0
        return masks, None, areas, report

    kept = drop_labels(masks, single)
    dropped = np.where(np.isin(masks, single), masks, 0).astype(masks.dtype)
    # areas of the filtered volume: the single-slice columns are simply removed,
    # so there is no need to re-scan the volume
    areas = areas.copy()
    areas[:, single] = 0
    return kept, dropped, areas, report


def drops_here(method, drop_in=DROP_SINGLE_SLICE_IN, drop=DROP_SINGLE_SLICE) -> bool:
    """Does the single-slice filter apply to this technique?"""
    if not drop:
        return False
    return drop_in is None or str(method) in set(drop_in)


# =============================================================================
# Part 2 -- brightness statistics that do not assume a shape
# =============================================================================

def hist_cdf_at(counts: np.ndarray, edges: np.ndarray, x: float) -> float:
    """Fraction of a binned distribution at or below ``x``.

    The inverse of ``b.hist_quantile``, interpolating inside the straddled bin
    the same way, so ``hist_cdf_at(c, e, hist_quantile(c, e, q)) == q``.
    """
    total = counts.sum()
    if total == 0 or not np.isfinite(x):
        return float("nan")

    i = int(np.searchsorted(edges, x, side="right")) - 1
    if i < 0:
        return 0.0
    if i >= counts.size:
        return 1.0

    width = edges[i + 1] - edges[i]
    frac = 0.0 if width <= 0 else (x - edges[i]) / width
    frac = min(max(frac, 0.0), 1.0)
    return float((counts[:i].sum() + frac * counts[i]) / total)


BRIGHTNESS_COLUMNS = ["n_voxels", "n_masked", "frac_masked", "separability",
                      "mean_masked", "mean_unmasked",
                      "median_masked", "median_unmasked", "median_shift_iqr",
                      "dim_masked_rate", "bright_unmasked_rate",
                      "point_biserial_r"]


def brightness_stats(counts: np.ndarray, edges: np.ndarray) -> pd.Series:
    """The brightness script's pooled statistics, plus three rank-based ones.

    ``separability`` is the brightness script's ``auc``, renamed for what it
    measures: P(a random masked voxel is brighter than a random unmasked one).
    0.5 is a technique whose masks say nothing about brightness and 1.0 is one
    whose every masked voxel outshines every unmasked one.  It is
    distribution-free and the one number that survives the two groups being
    wildly different sizes.  Added here:

    ``dim_masked_rate``      share of masked voxels below the *unmasked* median.
                             Masks placed at random would score 0.5; background
                             pulled into a mask pushes it up.
    ``bright_unmasked_rate`` share of unmasked voxels above the *masked* median —
                             signal the technique left behind.  It is bounded
                             below by how little of the volume is nucleus, so read
                             it against ``frac_masked`` rather than on its own.
    ``median_shift_iqr``     (median masked − median unmasked) / IQR of unmasked,
                             a robust standardised separation: the same quantity
                             a Cohen's d would report, with medians and an IQR in
                             place of means and an SD.
    """
    pooled = b.summarise_pooled(counts, edges)
    masked = counts[1].sum(axis=0)
    unmasked = counts[0].sum(axis=0)

    iqr_unmasked = pooled["p75_unmasked"] - pooled["p25_unmasked"]
    shift = pooled["median_masked"] - pooled["median_unmasked"]

    return pd.Series({
        "n_voxels": pooled["n_pixels"],
        "n_masked": pooled["n_masked"],
        "frac_masked": pooled["frac_masked"],
        "separability": pooled["auc"],      # the script's key, this table's name
        # the script's own means, carried through and shown: they are a mean of a
        # bimodal distribution and so describe neither mode, which is why nothing
        # is scored on them -- but "how bright is a voxel this technique claimed"
        # is the plainest reading of a mask there is, and it is in the units of
        # the image rather than a rate
        "mean_masked": pooled["mean_masked"],
        "mean_unmasked": pooled["mean_unmasked"],
        "median_masked": pooled["median_masked"],
        "median_unmasked": pooled["median_unmasked"],
        "median_shift_iqr": (shift / iqr_unmasked) if iqr_unmasked > 0 else np.nan,
        "dim_masked_rate": hist_cdf_at(masked, edges, pooled["median_unmasked"]),
        "bright_unmasked_rate": 1.0 - hist_cdf_at(unmasked, edges,
                                                  pooled["median_masked"]),
        "point_biserial_r": pooled["point_biserial_r"],
    }, index=BRIGHTNESS_COLUMNS)


# =============================================================================
# Part 3 -- the distribution figure
# =============================================================================

def _log_distribution(ax, values_by_method, lo, top, n_bins=45, marks=("median",)):
    """Log-binned share of a positive, heavy-tailed quantity, one step per technique.

    Step outlines rather than filled bars: two filled histograms on one axes hide
    each other, and the whole point is reading them against each other.  Log bins
    cannot hold zero, so both tails fold into the edge bins — the caller says so
    in an annotation rather than letting them vanish.
    """
    bins = np.logspace(np.log10(lo), np.log10(top), n_bins + 1)
    centres = np.sqrt(bins[:-1] * bins[1:])
    colors = method_colors(list(values_by_method))
    styles = ("-", "--", "-.", ":")

    for i, (name, values) in enumerate(values_by_method.items()):
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        share, _ = np.histogram(np.clip(values, lo, top), bins=bins)
        share = share / values.size

        ax.fill_between(centres, share, step="mid", color=colors[name],
                        alpha=0.16, linewidth=0)
        ax.step(centres, share, where="mid", color=colors[name],
                linestyle=styles[i % len(styles)], linewidth=2, zorder=3,
                label=f"{name}  (n={values.size:,})")

        for mark in marks:
            q = 0.5 if mark == "median" else float(mark)
            ax.axvline(np.quantile(values, q), color=colors[name],
                       linestyle=styles[i % len(styles)], linewidth=1.2,
                       alpha=0.6, zorder=2)

    ax.set_xscale("log")
    ax.set_xlim(lo, top)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _pos: f"{v:g}"))
    ax.xaxis.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["left"].set_visible(False)
    return ax


def compare_profile_distribution(labels: pd.DataFrame, column: str = "jitter",
                                 group: str = "method",
                                 cutoff: float = JITTER_CUTOFF, n_bins: int = 45,
                                 title: str | None = None) -> Figure:
    """Per-mask ``jitter`` (or ``sway``), log-binned, one curve per technique.

    Both quantities are dimensionless and start at 1.0x, so the absolute cutoff
    means the same thing in every FOV — where a quantile cutoff flags the top 10%
    of every FOV by construction, which cannot compare two techniques.

    **This figure is how the cutoffs get set.**  Draw it over the full scan and
    look for the knee where the ordinary bulk ends and the tail begins; the
    shipped ``JITTER_CUTOFF`` / ``SWAY_CUTOFF`` are provisional values from worked
    profiles, not from any real data.
    """
    axis_label = {
        "jitter": "jitter = total reversal travel, as a factor (log scale)",
        "sway": "sway = worst change in growth rate, as a factor (log scale)",
    }.get(column, f"{column} (log scale)")
    if title is None:
        title = ("How much each mask's area profile wanders up and down"
                 if column == "jitter" else
                 "How sharp the worst corner in each area profile is")

    methods = list(dict.fromkeys(labels[group]))
    values = {name: labels.loc[labels[group] == name, column].dropna().to_numpy()
              for name in methods}
    pooled = labels[column].dropna().to_numpy()
    top = max(float(np.nanmax(pooled)) if pooled.size else 1.1, cutoff * 1.3, 1.1)

    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    _log_distribution(ax, values, 1.0, top, n_bins=n_bins)
    ax.axvspan(cutoff, top, color=CATEGORICAL[7], alpha=0.06, linewidth=0, zorder=1)
    ax.axvline(cutoff, color=CATEGORICAL[7], linestyle="--", linewidth=1.4,
               zorder=4, label=f"cutoff = {cutoff:g}x")

    n_missing = int(labels[column].isna().sum())
    if n_missing:
        ax.annotate(f"{n_missing:,} mask(s) never get three consecutive\n"
                    f"planes, so have no {column} and are never flagged",
                    xy=(0.99, 0.55), xycoords="axes fraction", ha="right",
                    va="top", fontsize=8.5, color=INK_SECONDARY)

    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _pos: f"{v:g}x"))
    ax.set_xlabel(axis_label)
    ax.set_ylabel("share of masks")
    ax.set_title(title, pad=22, loc="left")
    ax.annotate("1.0x = a profile that never turns around, whatever its slope   ·   "
                "vertical rule = that technique's median",
                xy=(0, 1.03), xycoords="axes fraction", fontsize=9, color=INK_SECONDARY)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# =============================================================================
# Part 4 -- one FOV, all three measurements
# =============================================================================

def load_dapi(dapi_glob):
    """The DAPI stack both techniques segmented, read once.

    Numeric z ordering — ``z10`` sorts after ``z9``, which a plain sort of the
    filenames gets wrong.  Read once per FOV and reused across techniques: the
    scripts' own loaders re-read the stack for every mask volume.
    """
    files = glob.glob(str(dapi_glob))

    def z_index(f):
        m = re.search(r"z(\d+)", Path(f).stem)
        return int(m.group(1)) if m else 0
    files = sorted(files, key=z_index)

    if not files:
        raise FileNotFoundError(f"No DAPI files matched pattern: {dapi_glob}")
    return np.stack([tifffile.imread(f) for f in files], axis=0), files


def load_masks(path) -> np.ndarray:
    masks = tifffile.imread(str(path))
    if masks.ndim == 2:
        masks = masks[np.newaxis]
    return masks


def analyse_fov(masks, dapi=None, edges=None, exact=None, drop=DROP_SINGLE_SLICE,
                jitter_cutoff=JITTER_CUTOFF, sway_cutoff=SWAY_CUTOFF,
                area_floor=AREA_FLOOR, sliver_frac=SLIVER_FRAC,
                thin_max=THIN_MAX, deep_min=DEEP_MIN, keep_volumes=True) -> dict:
    """Every measurement on one mask volume.

    Order matters: the single-slice filter runs first, so the artifacts are gone
    before the spans are counted, before the profile is measured, and before a
    single voxel is histogrammed.

    Returns a dict with ``labels`` (one row per surviving mask, carrying the span
    jitter and sway columns side by side), ``brightness`` and the histogram
    ``counts``/``edges`` when a DAPI stack was given, plus the filter ``report``.
    With ``keep_volumes`` the filtered and removed volumes come back too — the
    region walk turns that off so peak memory stays at one FOV.

    :func:`measure_slice_areas` is called once and its result feeds the span table,
    the filter and both profile metrics: one bincount pass answers all of them.
    """
    areas = measure_slice_areas(masks)
    kept, dropped, areas, report = filter_single_slice(masks, areas, drop=drop)

    spans = span_table(areas)
    spans["thin"] = spans["z_span"] <= thin_max
    spans["deep"] = spans["z_span"] >= deep_min

    # the sway script's own functions, on the filtered areas.  A single-slice mask
    # could never have been measured anyway -- both metrics need three
    # consecutive planes -- so the filter changes the denominator here, not the
    # numerator.  Both read the same sliver rule, so a partial-volume onset plane
    # is out of the profile before either of them looks at it.
    jitters = sw.check_area_jitter(areas, area_floor=area_floor,
                                   sliver_frac=sliver_frac)
    jitters, jitter_cut, jitter_source = sw.flag_jitter(jitters, cutoff=jitter_cutoff)

    sways = sw.check_area_sway(areas, area_floor=area_floor, sliver_frac=sliver_frac)
    sways, sway_cut, sway_source = sw.flag_sway(sways, cutoff=sway_cutoff)

    labels = spans.merge(
        jitters[["label", "n_planes_used", "n_reversals", "jitter", "z_at_jitter",
                 "large_jitter"]], on="label", how="left")
    labels = labels.merge(sways[["label", "sway", "z_at_sway", "large_sway"]],
                          on="label", how="left")
    # a mask too short to be measured is NaN here, and NaN is not True
    labels["large_jitter"] = labels["large_jitter"].eq(True)
    labels["large_sway"] = labels["large_sway"].eq(True)

    out = {"labels": labels, "report": report,
           "jitter_cutoff": jitter_cut, "jitter_source": jitter_source,
           "sway_cutoff": sway_cut, "sway_source": sway_source,
           "brightness": None, "counts": None, "edges": edges}

    if dapi is not None:
        if dapi.shape != kept.shape:
            raise ValueError(f"masks shape {kept.shape} != dapi shape {dapi.shape}")
        if edges is None:
            edges, exact = b.build_bin_edges(dapi)
        elif exact is None:
            # bins of width 1 are what build_bin_edges emits for an integer volume
            exact = bool(np.allclose(np.diff(edges), 1.0))
        brightness, present = b.vectorise_pixels(dapi, kept)
        counts = b.accumulate_histograms(brightness, present, edges, exact=exact)
        out.update(brightness=brightness_stats(counts, edges), counts=counts,
                   edges=edges)

    if keep_volumes:
        out.update(masks=kept, dropped=dropped, areas=areas)
    return out


# =============================================================================
# Part 5 -- every FOV in a region, every region in a run
# =============================================================================

def fov_dapi_glob(dapi_root, fov, pattern=DAPI_PATTERN):
    """``<dapi_root>/<fov>/DAPI_decon_z*.tif`` — the image both techniques saw."""
    return str(Path(dapi_root) / fov / pattern)


def scan_region(found: pd.DataFrame, dapi_root=None, region: str | None = None,
                pattern=DAPI_PATTERN, with_brightness=True, verbose=True,
                drop_in=DROP_SINGLE_SLICE_IN, drop=DROP_SINGLE_SLICE,
                **analysis) -> dict:
    """:func:`analyse_fov` on every row of ``found``; returns tidy frames.

    ``found`` is what :func:`find_region_fovs` returns — one row per (technique,
    FOV) with the mask path, ragged FOVs already dropped.  The DAPI stack is read
    once per FOV and both techniques are histogrammed onto bins built from it:
    two distributions binned differently cannot be laid over each other.

    One DAPI stack and one mask volume are resident at a time, so peak memory is
    set by the largest FOV rather than by how many there are.  Only the tables
    and the histograms survive the loop — a few hundred KB per run against
    hundreds of MB for the volumes.
    """
    label_frames, bright_rows, filter_rows, runs = [], [], [], {}

    for fov, rows in found.groupby("fov", sort=True, observed=True):
        dapi, edges, exact = None, None, None
        if with_brightness and dapi_root is not None:
            dapi, _ = load_dapi(fov_dapi_glob(dapi_root, fov, pattern))
            edges, exact = b.build_bin_edges(dapi)

        counts_by_method = {}
        for row in rows.itertuples():
            method = str(row.method)
            masks = load_masks(row.path)
            result = analyse_fov(masks, dapi=dapi, edges=edges, exact=exact,
                                 drop=drops_here(method, drop_in, drop),
                                 keep_volumes=False, **analysis)
            del masks

            tags = {"fov": fov, "method": method}
            if region is not None:
                tags = {"region": region, **tags}

            label_frames.append(result["labels"].assign(**tags))
            filter_rows.append({**tags, **result["report"]})
            if result["brightness"] is not None:
                bright_rows.append({**tags, **result["brightness"].to_dict()})
                counts_by_method[method] = result["counts"]

            if verbose:
                r = result["report"]
                print(f"  {fov:<10} {method:<12} "
                      f"{len(result['labels']):>5} masks  "
                      f"{int(result['labels']['jitter'].notna().sum()):>5} measurable  "
                      f"({r['n_dropped']:>4} single-slice dropped of "
                      f"{r['n_total']:>5})")

        if counts_by_method:
            runs[fov] = {"edges": edges, "counts": counts_by_method}
        del dapi

    index = ["fov", "method"] if region is None else ["region", "fov", "method"]

    def stack(frames):
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def keyed(rows):
        return pd.DataFrame(rows).set_index(index) if rows else pd.DataFrame()

    return {"labels": stack(label_frames), "brightness": keyed(bright_rows),
            "filters": keyed(filter_rows), "runs": runs}


def scan_regions(technique_roots, dapi_patches=None, regions=None,
                 pattern=DAPI_PATTERN, verbose=True, **kwargs) -> dict:
    """:func:`scan_region` over several regions, concatenated.

    ``regions=None`` scans every region present under *every* technique root.
    ``dapi_patches`` is the directory above the region, the same way
    ``technique_roots`` is, so the region name is appended here and named once.
    """
    if regions is None:
        regions = available_regions(technique_roots)

    parts = []
    for region in regions:
        if verbose:
            print(f"\n### {region}")
        found = find_region_fovs(technique_roots, region)
        dapi_root = None if dapi_patches is None else Path(dapi_patches) / region
        parts.append(scan_region(found, dapi_root, region=region, pattern=pattern,
                                 verbose=verbose, **kwargs))

    if not parts:
        raise FileNotFoundError(f"no regions found under {list(technique_roots.values())}")

    merged = {"runs": {}}
    merged["labels"] = pd.concat([p["labels"] for p in parts], ignore_index=True)
    for key in ("brightness", "filters"):
        frames = [p[key] for p in parts if not p[key].empty]
        merged[key] = pd.concat(frames) if frames else pd.DataFrame()
    for region, part in zip(regions, parts):
        for fov, run in part["runs"].items():
            merged["runs"][(region, fov)] = run
    return merged


# =============================================================================
# Part 6 -- one row per FOV per technique, then the scores
# =============================================================================

def _group_keys(frame: pd.DataFrame) -> list:
    return [c for c in ("region", "fov", "method") if c in frame.columns]


def _align(frame: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Put ``frame``'s index levels in ``index``'s order so a join lines up."""
    if frame.index.nlevels == index.nlevels and list(frame.index.names) != list(index.names):
        frame = frame.reorder_levels(list(index.names))
    return frame.reindex(index)


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def fov_metrics(labels: pd.DataFrame, brightness: pd.DataFrame | None = None,
                filters: pd.DataFrame | None = None, by=None) -> pd.DataFrame:
    """One row per (region, FOV, technique) — every individual statistic.

    Grouping keys default to whichever of ``region``/``fov``/``method`` the tables
    carry, so the same call covers one region or all of them.

    Every "centre" column here is a median and every "spread" an IQR or a
    percentile.  ``n_dropped``/``pct_dropped`` come from the single-slice filter
    and are *listed, not scored* — on the card too, under its own **Reported**
    header: an imaging artifact is not the segmenter's defect, which is the whole
    reason it was filtered.

    Every column survives here whether or not the scorecard draws it.
    ``pct_deep``, ``pct_z_gap`` and ``median_z_span`` no longer appear on the card
    and no longer feed ``Span``; ``dim_masked_rate``, ``bright_unmasked_rate``,
    ``frac_masked`` and ``median_shift_iqr`` are off it too, leaving
    ``separability`` and ``mean_masked`` under Signal.  All of them are still the
    honest per-FOV numbers — this table is the wide one and the card is the
    narrow one, and ``frac_masked`` in particular is what tells you whether a high
    ``separability`` was earned or bought by masking only the brightest voxels.

    ``mean_masked``/``mean_unmasked`` are the brightness script's own means, in
    the units of the image.  They are the one place a mean is quoted here and
    nothing is scored on them: brightness is bimodal by construction, so a mean
    over masked voxels sits between the two modes and describes neither.  Read
    ``mean_masked`` as "how bright is a voxel this technique claimed", against
    ``mean_unmasked`` as the floor it should be well clear of.
    """
    by = list(by) if by is not None else _group_keys(labels)

    per_mask = labels.groupby(by, sort=False, observed=True)
    metrics = pd.DataFrame({
        "n_masks": per_mask.size(),
        "pct_thin": 100 * per_mask["thin"].mean(),
        "pct_deep": 100 * per_mask["deep"].mean(),
        "pct_z_gap": 100 * per_mask["has_z_gap"].mean(),
        "median_z_span": per_mask["z_span"].median(),
        "max_z_span": per_mask["z_span"].max(),
        "n_with_jitter": per_mask["jitter"].count(),
        "median_jitter": per_mask["jitter"].median(),
        "p90_jitter": per_mask["jitter"].quantile(0.90),
        "median_reversals": per_mask["n_reversals"].median(),
        "median_sway": per_mask["sway"].median(),
        "p90_sway": per_mask["sway"].quantile(0.90),
        # denominator is masks that *have* the metric, as the sway script reports
        # it: both need three consecutive planes, and a mask that never gets them
        # is already paid for in pct_thin or pct_z_gap.  Dividing by every mask
        # instead would reward a technique for producing masks too short to
        # measure.
        "pct_jittery": 100 * per_mask["large_jitter"].sum() / per_mask["jitter"].count(),
        "pct_swayed": 100 * per_mask["large_sway"].sum() / per_mask["sway"].count(),
    })

    if brightness is not None and not brightness.empty:
        metrics = metrics.join(_align(brightness, metrics.index))

    if filters is not None and not filters.empty:
        metrics = metrics.join(
            _align(filters[["n_total", "n_dropped", "pct_dropped"]], metrics.index))

    return metrics


# (column, group, header, format).  Polarity is declared in POLARITY below, not
# here, because the same column is read in both places.
#
# Span is one column.  pct_deep / pct_z_gap / median_z_span, and dim_masked_rate
# / frac_masked / median_shift_iqr, are still computed by fov_metrics and still
# in the tables -- they are off the card because nothing scores them, and a
# column nobody scores is a column nobody reads.  Two deliberate exceptions are
# drawn anyway: mean_masked, because "how bright is a voxel this technique
# claimed" is the plainest reading of a mask there is and separability alone
# cannot give it in the units of the image, and pct_dropped, which gets a header
# of its own to say that nothing scores it.
SCORECARD_SPEC = (
    ("pct_thin",        "Span",       f"%\nthin (<={THIN_MAX})", "{:.0f}"),
    ("median_jitter",   "Jitter",     "median\njitter",          "{:.2f}"),
    ("pct_jittery",     "Jitter",     "%\njittery",              "{:.0f}"),
    ("median_sway",     "Sway",       "median\nsway",            "{:.1f}"),
    ("pct_swayed",      "Sway",       "%\nswayed",               "{:.0f}"),
    ("separability",    "Signal",     "separability",            "{:.3f}"),
    ("mean_masked",     "Signal",     "mean\nin mask",           "{:,.0f}"),
    ("pct_dropped",     "Reported",   "%\nsingle-slice",         "{:.1f}"),
)

# +1 = higher is better, -1 = lower is better.  Used by the technique comparison
# to name a winner, and nowhere else -- the scores below are absolute.  0 is
# "no preferred direction", which is what pct_dropped gets: it is an imaging
# artifact rate, so neither arm wins by having more or fewer of them.
POLARITY = {
    "pct_thin": -1, "pct_deep": -1, "pct_z_gap": -1, "median_z_span": 0,
    "max_z_span": -1, "pct_dropped": 0, "n_dropped": 0,
    "median_jitter": -1, "p90_jitter": -1, "pct_jittery": -1,
    "median_reversals": -1, "n_with_jitter": 0,
    "median_sway": -1, "p90_sway": -1, "pct_swayed": -1,
    "separability": +1, "frac_masked": 0, "median_shift_iqr": +1,
    "mean_masked": 0, "mean_unmasked": 0,
    "dim_masked_rate": -1, "bright_unmasked_rate": -1, "n_masks": 0,
    "Span": +1, "Jitter": +1, "Sway": +1, "Signal": +1, "Total": +1,
}

# The rubric.  Each subscore earns up to its weight and the four sum to 1.00, so
# a perfect run totals 1.00 and every point lost is traceable to the line that
# lost it.  The old single Smoothness line at 0.50 is now two, because it was
# measuring one thing and claiming two: Jitter 0.30 for a profile that wanders up
# and down, Sway 0.20 for the flat/jump/flat over-merge.  Jitter carries more
# because a mask that cannot hold a stable cross-section is unusable, where a
# sway is one suspicious transition on an otherwise coherent object.  These
# numbers are the editorial content of the card -- change them here, in one
# place, keep them summing to 1.00, and say so when you do.
SCORE_WEIGHTS = {"Span": 0.10, "Jitter": 0.50, "Sway": 0.10, "Signal": 0.30}
TOTAL_COLUMN = "Total"

SCORE_COLUMNS = ("Span", "Jitter", "Sway", "Signal", TOTAL_COLUMN)


def score_fovs(metrics: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """Three group scores and the weighted ``Total`` they earn out of 1.00.

    ``Span       = 1 - pct_thin / 100``
        masks too short to keep — the share of a technique's output that a
        ``min_z = 3`` filter discards, which makes the score the share it keeps.
        ``pct_deep`` and ``pct_z_gap`` used to be summed into this and are not
        any more: a deep mask only *suggests* two nuclei read as one and a tall
        nucleus is not a defect, while z-gaps are too rare to move a score.  Both
        are still measured — read them in ``fov_metrics``, where being unscored
        costs them nothing.
    ``Jitter     = 1 - pct_jittery / 100``
        Each mask's *jitter* is ``10 ** Σ min(|d[k]|, |d[k+1]|)`` summed over
        every place its per-plane log-area growth ``d`` reverses sign — total
        up-and-down travel, as a factor — and a mask is flagged above
        ``JITTER_CUTOFF``.  ``pct_jittery`` is the share of flagged masks among
        those long enough to be measured, so ``Jitter`` is 1.00 when no profile
        wanders and pays one hundredth for each percent flagged.

        A monotone profile scores exactly 1.0 at any steepness, which is the
        whole point: this replaced a metric that ranked the steepest *onsets* in
        the field as its worst offenders.  Weighting each reversal by the smaller
        of its two limbs is what makes a ramp free, and summing rather than
        maxing is what lets many small wobbles outscore one large excursion.
    ``Sway       = 1 - pct_swayed / 100``
        The original metric, kept because ``Jitter`` cannot see what it catches:
        ``10 ** max |log10 a[z+1] - 2 log10 a[z] + log10 a[z-1]|``, the sharpest
        corner in the profile, flagged above ``SWAY_CUTOFF``.  The
        flat-run/jump/flat-run of an over-merge is *monotone*, so it reverses
        nowhere and scores a clean 1.00 jitter; this is the line that charges for
        it.  What changed is not the maths but the sliver exclusion below, which
        is what stopped it from flagging every mask's onset ramp.

        Both metrics now exclude partial-volume slivers (``SLIVER_FRAC``) rather
        than flooring them, which is what stopped every mask's onset ramp from
        scoring as a corner.
    ``Signal     = separability``
        the brightness AUC, used as the score directly: P(a random masked voxel
        outshines a random unmasked one), so 1.00 is a technique whose every
        masked voxel is brighter than every unmasked one.  Rank-based, so it is
        unaffected by the two groups being wildly different sizes — but a
        technique can buy a high separability by masking only the brightest
        voxels, so read it against ``frac_masked`` in ``fov_metrics``.

        Note the floor this puts under the line.  An AUC of 0.5 is the
        no-information point — masks placed without regard to brightness — and it
        scores 0.50 here rather than 0.00, so ``Signal`` banks half its 0.30
        whatever the technique does, and only outright inversion (masks landing
        on the *dim* voxels) drives it lower.  The rescaling this replaced,
        ``2 * separability - 1``, put the no-information point at 0 and made the
        line cost its full weight; on the raw AUC the interesting range is
        roughly 0.8 to 1.0, and differences inside it are compressed to a third
        of what they were.  That is the trade taken deliberately: the score is
        now the same number as the column beside it.

    Span, Jitter and Sway start at 1.00 and pay one hundredth per percent of
    defect; Signal is an AUC on its own scale, floored at 0.5 for anything that
    is not anti-correlated with brightness.  All four are absolute, not ranked:
    0.82 means the same thing in every region, and adding a FOV to the table
    moves nobody.  Enough defect drives a rate score below zero and the number is
    reported as computed — only the drawn bar clips at 0, because a 0.00 and a
    -0.40 are not the same run.

    ``Total`` — the rubric
    ----------------------
    ``Total = 0.20 * Span + 0.30 * Jitter + 0.20 * Sway + 0.30 * Signal``
    (:data:`SCORE_WEIGHTS`).  The weights sum to 1, so a clean run earns **1.00**
    and every point lost is traceable to the line item that lost it: at most 0.20
    to thin masks, 0.30 to wandering profiles, 0.20 to over-merge steps, 0.30 to
    dim ones — though Signal's 0.5 floor means the last is 0.15 in practice.

    The weighting is the one editorial judgement on the card, and it is stated
    rather than implied — an equal split would have been a weighting too, just an
    unadmitted one.  What it says: a corner in the area profile is a segmenter
    gluing two objects into one, which is a **wrong answer** and cannot be
    repaired downstream.  The old 0.50 for "smoothness" is split evenly-ish: 0.30
    to Jitter, because a mask whose cross-section cannot hold still from plane to
    plane is not a usable object, and 0.20 to Sway, one suspicious transition on
    an otherwise coherent one.  Jitter no longer leads the card outright — it
    ties Signal at 0.30 — so read the four lines as three near-equals over a
    lighter Sway rather than as one dominant defect.  Brightness separability
    at 0.30 is real evidence but softer — a technique can buy it by masking only
    the brightest voxels, which is why ``frac_masked`` and ``mean_masked`` are
    worth a glance before reading it.  Thin masks at 0.20 are the cheapest defect
    of the three: a ``min_z`` filter deletes them in one line, so they cost
    throughput rather than correctness — but at a fifth of the card they are no
    longer a rounding error either.

    ``Total`` is **not** clipped into [0, 1].  A subscore below zero carries into
    it, for the same reason the subscores themselves are reported as computed:
    a run that scores −0.10 is not a run that scores 0.00.  Only the drawn bar
    clips.

    A missing subscore is renormalised over the weights that are present — a scan
    without DAPI has no ``Signal``, so its ``Total`` is Span, Jitter and Sway over
    0.70 rather than a hole.  It is still a number out of 1.00, but it is a
    two-line rubric and not the three-line one, so do not read it against a card
    that has all three.
    """
    weights = SCORE_WEIGHTS if weights is None else weights

    scores = pd.DataFrame(index=metrics.index)
    scores["Span"] = 1 - metrics["pct_thin"] / 100
    scores["Jitter"] = 1 - metrics["pct_jittery"] / 100
    scores["Sway"] = 1 - metrics["pct_swayed"] / 100
    if "separability" in metrics.columns:
        scores["Signal"] = metrics["separability"]

    present = pd.Series({k: v for k, v in weights.items() if k in scores.columns},
                        dtype=float)
    if not present.empty:
        parts = scores[list(present.index)]
        # per row, over the components that are actually there: a NaN subscore
        # drops out of both the numerator and the weight it would have carried,
        # so the total stays a number out of 1.00 instead of becoming NaN itself
        earned = parts.mul(present, axis=1).sum(axis=1, skipna=True)
        available = parts.notna().mul(present, axis=1).sum(axis=1)
        scores[TOTAL_COLUMN] = (earned / available).where(available > 0)
    return scores


# =============================================================================
# Part 7 -- the scorecard figure
# =============================================================================

def _row_labels(index, row_sep=" · ") -> list:
    return [row_sep.join(str(part) for part in idx) if isinstance(idx, tuple) else str(idx)
            for idx in index]


def plot_scorecard(metrics: pd.DataFrame, spec=SCORECARD_SPEC,
                   title: str = "Segmentation scorecard", row_sep: str = " · ",
                   col_width: float = 0.82, score_width: float = 1.25,
                   scores: pd.DataFrame | None = None) -> Figure:
    """Individual statistics on the left, the rubric on the right.

    The layout: raw columns under their group header, then a bar per score.  Each subscore's header carries the weight it
    earns toward the ``Total`` (:data:`SCORE_WEIGHTS`), so the rubric is legible
    from the card without reading the source, and ``Total`` is drawn darker and
    bold behind a heavier rule because it is the sum of the columns beside it
    rather than another one of them.  A missing value prints as an en dash rather
    than ``nan`` — a FOV scanned without its DAPI has no ``Signal`` column, and
    that is not the same as scoring zero.
    """
    scores = score_fovs(metrics) if scores is None else scores
    spec = [s for s in spec if s[0] in metrics.columns]
    score_cols = [c for c in SCORE_COLUMNS if c in scores.columns]

    rows = _row_labels(metrics.index, row_sep)
    n_rows, n_cols, n_score = len(rows), len(spec), len(score_cols)

    fig, (ax, ax_score) = plt.subplots(
        1, 2,
        figsize=(col_width * n_cols + score_width * n_score + 3.4, 0.42 * n_rows + 2.3),
        gridspec_kw={"width_ratios": [col_width * n_cols, score_width * n_score],
                     "wspace": 0.06})

    for axis in (ax, ax_score):
        axis.set_ylim(n_rows - 0.5, -0.5)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.tick_params(length=0)
        for i in range(1, n_rows):          # dotted separators, as in the reference
            axis.axhline(i - 0.5, color=GRID, linestyle=":", linewidth=1.0, zorder=0)
        axis.axhline(-0.5, color=INK, linewidth=1.2, zorder=1)
        axis.axhline(n_rows - 0.5, color=INK, linewidth=1.2, zorder=1)

    # --- left: the raw numbers ------------------------------------------------
    ax.set_xlim(-0.5, n_cols - 0.5)
    for j, (col, _group, _header, fmt) in enumerate(spec):
        for i, value in enumerate(metrics[col].to_numpy()):
            text = fmt.format(value) if _finite(value) else "–"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=INK)

    ax.set_xticks(range(n_cols), [header for _c, _g, header, _f in spec], fontsize=8.5)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(n_rows), rows, fontsize=9, fontweight="bold")

    # --- right: the rubric, bar in cell ---------------------------------------
    ax_score.set_xlim(0, n_score)
    for j, name in enumerate(score_cols):
        total = name == TOTAL_COLUMN
        color = SEQUENTIAL_BLUE[-1] if total else SEQUENTIAL_BLUE[-5]
        for i, value in enumerate(scores[name].to_numpy()):
            if not _finite(value):
                ax_score.text(j + 0.5, i, "–", ha="center", va="center",
                              fontsize=9, color=INK_SECONDARY)
                continue
            drawn = min(max(float(value), 0.0), 1.0)   # the bar clips; the label does not
            ax_score.barh(i, 0.9 * drawn, left=j + 0.05, height=0.52, color=color, zorder=2)
            # label rides the end of the bar: inside while there is room, just
            # outside once the bar is too short to hold the text
            end = j + 0.05 + 0.9 * drawn
            inside = value >= 0.30
            ax_score.text(end + (-0.04 if inside else 0.04), i, f"{value:.2f}",
                          ha="right" if inside else "left", va="center", fontsize=8.5,
                          color=SURFACE if inside else INK_SECONDARY, zorder=3,
                          fontweight="bold" if total else "normal")
    for j in range(1, n_score):
        heavy = score_cols[j] == TOTAL_COLUMN     # the rubric's line, not a divider
        ax_score.axvline(j, color=INK_SECONDARY if heavy else GRID,
                         linewidth=1.2 if heavy else 0.8, zorder=1)

    # the weight each line earns, on the header, so the rubric reads off the card
    ax_score.set_xticks(
        np.arange(n_score) + 0.5,
        [name if name == TOTAL_COLUMN or name not in SCORE_WEIGHTS
         else f"{name}\n×{SCORE_WEIGHTS[name]:.2f}" for name in score_cols],
        fontsize=8.5)
    ax_score.xaxis.set_ticks_position("top")
    ax_score.set_yticks([])

    # --- group headers above the column labels --------------------------------
    spans, start = [], 0
    for j in range(1, n_cols + 1):
        if j == n_cols or spec[j][1] != spec[start][1]:
            spans.append((spec[start][1], start, j - 1))
            start = j
    spans = [(name, lo - 0.35, hi + 0.35, ax) for name, lo, hi in spans]
    if n_score > 1 and score_cols[-1] == TOTAL_COLUMN:
        # the total is what the columns beside it add up to, not another line
        spans.append(("Rubric", 0.05, n_score - 1.05, ax_score))
        spans.append(("out of 1.00", n_score - 0.95, n_score - 0.05, ax_score))
    else:
        spans.append(("Rubric", 0.05, n_score - 0.05, ax_score))

    for name, lo, hi, axis in spans:
        # x in data, y pinned to the top of the axes and pushed above the labels
        anchor = blended_transform_factory(axis.transData, axis.transAxes)
        rule = offset_copy(anchor, fig=fig, x=0, y=30, units="points")
        axis.plot([lo, hi], [1.0, 1.0], transform=rule, color=GRID, linewidth=1.2,
                  clip_on=False, zorder=4)
        axis.annotate(name, xy=((lo + hi) / 2, 1.0), xytext=(0, 34),
                      xycoords=anchor, textcoords="offset points",
                      ha="center", va="bottom", fontsize=9, color=INK_SECONDARY,
                      annotation_clip=False)

    fig.suptitle(title, x=0.01, y=0.98, ha="left", va="top", fontsize=11.5,
                 fontweight="bold", color=INK)
    fig.subplots_adjust(top=0.80, bottom=0.04, left=0.18, right=0.98)
    return fig


# =============================================================================
# Part 8 -- is the difference real?  Paired, rank-based, no normal anywhere
# =============================================================================

def _hodges_lehmann(diffs: np.ndarray) -> float:
    """Median of the Walsh averages — the estimator the signed-rank test inverts.

    The paired counterpart of the test below: where the median of the differences
    is a plain order statistic, this is the location estimate the Wilcoxon
    statistic is consistent with, and it is what should be quoted next to its
    p-value.
    """
    if diffs.size == 0:
        return float("nan")
    i, j = np.triu_indices(diffs.size)          # includes i == j, as Walsh defined it
    return float(np.median((diffs[i] + diffs[j]) / 2.0))


def _rank_biserial(diffs: np.ndarray) -> float:
    """Matched-pairs rank-biserial: (W+ − W−) / (W+ + W−), in [-1, 1].

    A correlation-scaled effect size for the signed-rank test.  ±1 is every pair
    moving the same way, 0 is the positive and negative ranks balancing.  Zero
    differences are dropped, as the test drops them.
    """
    nonzero = diffs[diffs != 0]
    if nonzero.size == 0:
        return float("nan")
    ranks = sps.rankdata(np.abs(nonzero))
    w_plus = ranks[nonzero > 0].sum()
    w_minus = ranks[nonzero < 0].sum()
    total = w_plus + w_minus
    return float((w_plus - w_minus) / total) if total else float("nan")


def _bootstrap_ci(values: np.ndarray, statistic=np.median, n_boot=10000,
                  alpha=0.05, seed=0):
    """Percentile bootstrap interval — no normal approximation, no SE.

    With a handful of FOVs the interval is wide and lumpy; that is the honest
    reading of a handful of FOVs, not a defect of the method.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True)
    stat = statistic(draws, axis=1)
    return (float(np.quantile(stat, alpha / 2)),
            float(np.quantile(stat, 1 - alpha / 2)))


COMPARISON_COLUMNS = ["n_pairs", "median_a", "median_b", "median_diff", "hl_shift",
                      "ci_lo", "ci_hi", "p_wilcoxon", "rank_biserial", "better"]


def compare_techniques(table: pd.DataFrame, a: str | None = None, b: str | None = None,
                       columns=None, method_level: str = "method",
                       n_boot: int = 10000, alpha: float = 0.05,
                       seed: int = 0) -> pd.DataFrame:
    """Paired, rank-based comparison of two techniques over the FOVs they share.

    The pairing is the point: FOV-to-FOV variation is usually larger than the
    difference between techniques, and both techniques segmented the *same*
    image, so the difference within a FOV is the measurement and the spread
    across FOVs is not error to be averaged over.

    Per column: the median under each technique, the median of the paired
    differences and its Hodges–Lehmann counterpart, a percentile bootstrap
    interval for the median difference, the two-sided **Wilcoxon signed-rank**
    p-value (exact while there are no ties and n is small), and the matched-pairs
    rank-biserial effect size.  ``better`` reads the sign against
    :data:`POLARITY`, and is blank for a column with no preferred direction.

    ``p`` is not a verdict on its own: with five or six FOVs the smallest
    attainable two-sided p is 0.06–0.03, so a real effect can be unable to reach
    0.05 no matter how consistent it is.  Read the effect size and the interval,
    and treat the p-value as a check that the pairs point the same way.
    """
    if method_level not in table.index.names:
        raise ValueError(f"{method_level!r} is not an index level of the table")

    methods = list(dict.fromkeys(table.index.get_level_values(method_level)))
    a = methods[0] if a is None else a
    b = methods[1] if b is None else b

    wide = table.unstack(method_level)
    if columns is None:
        columns = [c for c in table.columns
                   if pd.api.types.is_numeric_dtype(table[c])]

    rows = []
    for col in columns:
        if (col, a) not in wide.columns or (col, b) not in wide.columns:
            continue
        pair = wide[[(col, a), (col, b)]].dropna()
        x = pair[(col, a)].to_numpy(dtype=float)
        y = pair[(col, b)].to_numpy(dtype=float)
        diffs = y - x                            # b minus a, so a positive diff favours b

        if diffs.size and np.any(diffs != 0):
            try:
                p = float(sps.wilcoxon(x, y, zero_method="wilcox").pvalue)
            except ValueError:                   # every difference was zero
                p = float("nan")
        else:
            p = float("nan")

        lo, hi = _bootstrap_ci(diffs, n_boot=n_boot, alpha=alpha, seed=seed)
        shift = float(np.median(diffs)) if diffs.size else float("nan")
        polarity = POLARITY.get(col, 0)
        if not polarity or not np.isfinite(shift) or shift == 0:
            better = ""
        else:
            better = b if (shift > 0) == (polarity > 0) else a

        rows.append({
            "metric": col, "n_pairs": int(diffs.size),
            "median_a": float(np.median(x)) if x.size else np.nan,
            "median_b": float(np.median(y)) if y.size else np.nan,
            "median_diff": shift, "hl_shift": _hodges_lehmann(diffs),
            "ci_lo": lo, "ci_hi": hi, "p_wilcoxon": p,
            "rank_biserial": _rank_biserial(diffs), "better": better,
        })

    out = pd.DataFrame(rows).set_index("metric")
    out.columns = pd.Index(COMPARISON_COLUMNS, name=f"{a}  vs  {b}")
    return out


def plot_paired(table: pd.DataFrame, columns=None, method_level: str = "method",
                a: str | None = None, b: str | None = None, n_cols: int = 4,
                title: str = "Every FOV, under both techniques") -> Figure:
    """One panel per metric; one line per FOV between the two techniques.

    The figure the paired test is actually about.  A test says the pairs move one
    way; this says how many of them do, by how much, and whether one FOV is
    carrying the result — none of which a p-value can. The heavy line is the
    median under each technique, so it can be read against the crossing lines
    without being mistaken for a fit.
    """
    methods = list(dict.fromkeys(table.index.get_level_values(method_level)))
    a = methods[0] if a is None else a
    b = methods[1] if b is None else b
    colors = {a: CATEGORICAL[methods.index(a)], b: CATEGORICAL[methods.index(b)]}

    if columns is None:
        columns = [c for c in ("pct_thin", "pct_z_gap", "median_jitter",
                               "pct_jittery", "pct_swayed", "separability",
                               "dim_masked_rate", TOTAL_COLUMN)
                   if c in table.columns]

    wide = table.unstack(method_level)
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.9 * n_rows),
                             squeeze=False)

    for ax, col in zip(axes.ravel(), columns):
        pair = wide[[(col, a), (col, b)]].dropna()
        x = pair[(col, a)].to_numpy(dtype=float)
        y = pair[(col, b)].to_numpy(dtype=float)

        for xi, yi in zip(x, y):
            ax.plot([0, 1], [xi, yi], color=INK_SECONDARY, alpha=0.35,
                    linewidth=1.0, marker="o", markersize=3.5, zorder=2)
        for pos, values, name in ((0, x, a), (1, y, b)):
            if values.size:
                ax.plot([pos - 0.16, pos + 0.16], [np.median(values)] * 2,
                        color=colors[name], linewidth=3, zorder=3)

        ax.set_xlim(-0.45, 1.45)
        ax.set_xticks([0, 1], [a, b], fontsize=8.5)
        ax.set_title(col, fontsize=9.5, loc="left", pad=6)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.spines["bottom"].set_color(GRID)
        ax.spines["left"].set_visible(False)

    for ax in axes.ravel()[len(columns):]:
        ax.set_visible(False)

    fig.suptitle(f"{title}  ·  n = {len(wide)} FOV(s)  ·  bar = median",
                 x=0.005, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def technique_summary(metrics: pd.DataFrame, scores: pd.DataFrame | None = None,
                      method_level: str = "method", n_boot: int = 10000,
                      seed: int = 0) -> pd.DataFrame:
    """Median of every score per technique, with a bootstrap interval.

    The row a reader wants after the scorecard: one line per technique, the
    median across FOVs rather than the mean, because a single bad FOV should not
    be able to decide a technique.
    """
    scores = score_fovs(metrics) if scores is None else scores
    rows = []
    for method, sub in scores.groupby(level=method_level, sort=False, observed=True):
        row = {"method": method, "n_fovs": len(sub)}
        for col in sub.columns:
            values = sub[col].to_numpy(dtype=float)
            lo, hi = _bootstrap_ci(values, n_boot=n_boot, seed=seed)
            row[col] = float(np.nanmedian(values)) if values.size else np.nan
            row[f"{col}_ci"] = f"[{lo:.2f}, {hi:.2f}]" if np.isfinite(lo) else "–"
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


# =============================================================================
# Part 9 -- napari: every flagged mask, on one viewer
# =============================================================================

def build_flag_layers(masks: np.ndarray, labels: pd.DataFrame,
                      dropped: np.ndarray | None = None,
                      dapi: np.ndarray | None = None, counts=None, edges=None,
                      quantile: float = OVERLAP_QUANTILE,
                      thin_max: int = THIN_MAX, deep_min: int = DEEP_MIN) -> dict:
    """Every defect this module measures, as label volumes to toggle over the DAPI.

    Each entry is a copy of the mask volume with everything but the selected
    labels zeroed, so a layer can be switched on and off without recomputing
    anything.  The brightness pair are boolean volumes instead, since "voxel
    outside every mask yet brighter than the median masked voxel" is not a label.
    """
    def subset(condition):
        return np.where(np.isin(masks, labels.loc[condition, "label"].to_numpy()),
                        masks, 0).astype(masks.dtype)

    layers = {
        f"thin (z-span <= {thin_max})": subset(labels["thin"]),
        f"deep (z-span >= {deep_min})": subset(labels["deep"]),
        "z-gapped (hole in z)": subset(labels["has_z_gap"]),
        "flagged: jitter (wanders up and down)": subset(labels["large_jitter"]),
        "flagged: sway (one big jump)": subset(labels["large_sway"]),
    }
    if dropped is not None and int(dropped.max()) > 0:
        layers["dropped: single-slice artifacts"] = dropped

    if dapi is not None and counts is not None and edges is not None:
        bright_thr = b.hist_quantile(counts[1].sum(axis=0), edges, quantile)
        dim_thr = b.hist_quantile(counts[0].sum(axis=0), edges, quantile)
        present = masks > 0
        layers[f"bright, unmasked (>{bright_thr:,.0f})"] = (
            (~present & (dapi >= bright_thr)).astype(np.uint8))
        layers[f"dim, masked (<{dim_thr:,.0f})"] = (
            (present & (dapi <= dim_thr)).astype(np.uint8))

    return layers


def launch_viewer(dapi, masks, layers=None, title=None, z_scale=None):
    """One viewer: DAPI, the filtered masks, and every flagged layer, all hidden.

    Blocks the kernel until the window is closed, which is why the notebook keeps
    this in a cell of its own.
    """
    import napari  # Qt import, kept out of module scope so a headless kernel can import this file

    viewer = napari.Viewer(title=title) if title else napari.Viewer()
    viewer.add_image(
        dapi, name="DAPI (deconvolved)", colormap="gray",
        contrast_limits=[float(dapi.min()), float(np.percentile(dapi, 99.5))],
    )
    viewer.add_labels(masks, name="segmentation masks (filtered)", opacity=0.4)

    for name, volume in (layers or {}).items():
        viewer.add_labels(volume, name=name, opacity=0.6, visible=False,
                          blending="additive")

    if z_scale is not None:
        make_3d(viewer, z_scale)

    print(f"\nLaunching napari viewer{f' — {title}' if title else ''}")
    napari.run()
    return viewer


def make_3d(viewer=None, z_scale: float = 15.0):
    """Stretch z so the stack has roughly isotropic proportions in the 3D view.

    ``napari`` is imported locally so a headless kernel can still import this
    module, and the scale is an argument rather than a literal, since the right
    number is the z step divided by the pixel size.
    """
    import napari

    if viewer is None:
        viewer = napari.current_viewer()
    for layer in viewer.layers:
        layer.scale = (z_scale, 1, 1)
    viewer.reset_view()
    return viewer
