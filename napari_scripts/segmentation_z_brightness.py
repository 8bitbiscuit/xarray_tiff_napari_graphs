"""Correlate mask presence with pixel brightness across the z stack.

Every pixel column (y, x) is vectorised across z, so each voxel contributes one
(brightness, is_masked) pair. The headline output is two histograms side by
side -- brightness of masked voxels vs brightness of unmasked voxels -- drawn on
identical bins and identical axes so the two distributions can be read against
each other directly.

If a segmenter is picking up signal, masked voxels should sit well to the right
of unmasked ones. Overlap between the two is where the interesting failures
live: bright voxels nobody masked (missed signal) and dim voxels inside a mask
(background pulled in). Both are available as napari layers.

The histogram machinery itself lives in `segmentation_helpers_v2.py` and is
called from here, so this script and the notebook's brightness figures cannot
drift apart. Nothing is imported from another script, and napari is only
imported if LAUNCH_NAPARI is on, so the plots work in an environment without Qt.
"""

import glob
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# the helpers sit one directory up.  Running this as
# `python napari_scripts/segmentation_z_brightness.py` puts only napari_scripts/
# on sys.path, so the repo root goes on it too -- `pip install -e .` makes the
# same import work without this, and the line is harmless when it has.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import segmentation_helpers_v2 as v2  # noqa: E402

# CONFIG
# ----------------------------------------------------------------------------
MASKS_DIR = Path("../data/segmentations_3d_true/cpdino/decon/VePo/region_UWA-7648/fov_09")
DAPI_DIR = Path("../data/patches/VePo/region_UWA-7648/fov_09")

MASKS_PATH = MASKS_DIR / "masks.tif"
DAPI_GLOB = str(DAPI_DIR / "DAPI_decon_z*.tif")  # Each z index

# Histogram resolution. Integer volumes get one bin per intensity value (exact)
# as long as the range fits in MAX_EXACT_BINS; anything else gets FLOAT_HIST_BINS
# uniform bins. This is the internal resolution used for every statistic -- the
# plots re-bin down to N_DISPLAY_BINS.
FLOAT_HIST_BINS = 4096
MAX_EXACT_BINS = 1 << 17

# Distribution plot
N_DISPLAY_BINS = 60
CLIP_QUANTILE = 0.999   # x-axis stops here (of the pooled distribution); None -> full range
LOG_Y = False           # log y is worth turning on when the masked peak dwarfs the tail
MAKE_PLOT = True
PLOT_PATH = Path("z_brightness_by_mask.png")        # None -> show interactively

# Per-z breakdown
MAKE_PER_Z_PLOT = True
PER_Z_PLOT_PATH = Path("z_brightness_by_mask_per_z.png")
CSV_PATH = None                                     # e.g. Path("z_brightness_by_mask.csv")

# napari overlays: a voxel is "bright unmasked" if it is outside every mask yet
# brighter than this quantile of the masked distribution, and "dim masked" if it
# is inside a mask yet dimmer than this quantile of the unmasked distribution.
OVERLAP_QUANTILE = 0.5

LAUNCH_NAPARI = True
# ----------------------------------------------------------------------------

# Categorical slots 1 and 2 of the reference palette (validated: CVD ΔE 24.7).
C_MASKED = "#2a78d6"
C_UNMASKED = "#eb6834"
C_INK = "#52514e"
C_GRID = "#d8d7d3"


def load_data(dapi_glob, masks_path):
    files = glob.glob(dapi_glob)

    def z_index(f):
        m = re.search(r"z(\d+)", Path(f).stem)
        return int(m.group(1)) if m else 0
    files = sorted(files, key=z_index)

    if not files:
        raise FileNotFoundError(f"No DAPI files matched pattern: {dapi_glob}")
    dapi = np.stack([tifffile.imread(f) for f in files], axis=0)
    masks = tifffile.imread(masks_path)

    # Make sure both are the right thing & they line up
    if masks.ndim == 2:
        masks = masks[np.newaxis]
    assert masks.shape == dapi.shape, (f"masks shape {masks.shape} != dapi shape {dapi.shape}")
    n_objects = int(masks.max())

    print(f"Volume shape: {dapi.shape}  |  {n_objects} labeled objects")
    return dapi, masks, files


# The histogram machinery lives in ``segmentation_helpers_v2`` and is called from
# here, so this script and the notebook's brightness figures cannot drift apart.
# Re-exported under this module's names, so the rest of the file -- and
# ``segmentation_z_brightness_bimodal.py``, which builds on the same counts --
# reads unchanged.
vectorise_pixels = v2.vectorise_pixels
accumulate_histograms = v2.accumulate_histograms
bin_centers = v2.bin_centers
rebin = v2.rebin
hist_quantile = v2.hist_quantile
hist_moments = v2.hist_moments
hist_auc = v2.hist_auc
hist_point_biserial = v2.hist_point_biserial
summarise_by_z = v2.summarise_by_z
summarise_pooled = v2.summarise_pooled
STAT_COLUMNS = v2.STAT_COLUMNS


def build_bin_edges(brightness, n_bins=FLOAT_HIST_BINS, max_exact_bins=MAX_EXACT_BINS):
    """Bin edges covering the whole volume, one bin per value where that is cheap.

    Wrapped rather than re-exported so the resolution defaults are this script's
    CONFIG values and not the module's -- a default binds at import, so
    re-exporting would quietly ignore the block at the top of this file.
    """
    return v2.build_bin_edges(brightness, n_bins=n_bins, max_exact_bins=max_exact_bins)


def build_histograms(brightness, present, n_bins=FLOAT_HIST_BINS,
                     max_exact_bins=MAX_EXACT_BINS):
    """Per-z brightness histograms, split by mask presence, on bins of their own."""
    return v2.build_histograms(brightness, present, n_bins=n_bins,
                               max_exact_bins=max_exact_bins)


def give_brightness_summary(pooled, per_z):
    print("\n=== BRIGHTNESS vs MASK PRESENCE ===")
    print(f"  n_voxels: {int(pooled['n_pixels']):,}  "
          f"({int(pooled['n_masked']):,} masked, {100 * pooled['frac_masked']:.2f}%)")

    if not pooled["n_masked"] or pooled["n_masked"] == pooled["n_pixels"]:
        print("  one group is empty; nothing to compare")
        return

    print(f"  masked   : mean {pooled['mean_masked']:.1f}  "
          f"median {pooled['median_masked']:.1f}  "
          f"IQR {pooled['p25_masked']:.1f}-{pooled['p75_masked']:.1f}")
    print(f"  unmasked : mean {pooled['mean_unmasked']:.1f}  "
          f"median {pooled['median_unmasked']:.1f}  "
          f"IQR {pooled['p25_unmasked']:.1f}-{pooled['p75_unmasked']:.1f}")
    print(f"  separability (AUC): {pooled['auc']:.3f}   "
          f"point-biserial r: {pooled['point_biserial_r']:.3f}")

    print("\n  per z slice:")
    for row in per_z.itertuples():
        print(f"    z {row.z:>2}: masked {100 * row.frac_masked:5.2f}%  "
              f"median {row.median_masked:8.1f} vs {row.median_unmasked:8.1f}  "
              f"AUC {row.auc:.3f}")


# --- plotting ----------------------------------------------------------------

def _style_axes(ax):
    ax.grid(axis="y", color=C_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors=C_INK, length=3)


def rebin(counts, edges, new_edges):
    """Aggregate fine bins into display bins; anything outside falls in the end bins."""
    centers = bin_centers(edges)
    idx = np.searchsorted(new_edges, centers, side="right") - 1
    idx = np.clip(idx, 0, new_edges.size - 2)
    return np.bincount(idx, weights=counts, minlength=new_edges.size - 1)


def display_edges(counts, edges, n_bins=N_DISPLAY_BINS, clip_quantile=CLIP_QUANTILE):
    """Common bins for both panels: data minimum up to a quantile of the pooled data."""
    pooled = counts[0].sum(axis=0) + counts[1].sum(axis=0)
    nonzero = np.flatnonzero(pooled)
    lo = float(edges[nonzero[0]])
    hi = float(edges[nonzero[-1] + 1])

    if clip_quantile is not None:
        clipped = hist_quantile(pooled, edges, clip_quantile)
        if np.isfinite(clipped) and clipped > lo:
            hi = clipped

    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, n_bins + 1)


def plot_brightness_distributions(counts, edges, pooled_stats, path=PLOT_PATH,
                                  n_bins=N_DISPLAY_BINS, clip_quantile=CLIP_QUANTILE,
                                  log_y=LOG_Y):
    """The headline figure: masked vs unmasked brightness, same bins, same axes."""
    fine = {"masked": counts[1].sum(axis=0), "unmasked": counts[0].sum(axis=0)}
    if fine["masked"].sum() == 0 or fine["unmasked"].sum() == 0:
        print("\nOne group is empty; nothing to plot")
        return None

    new_edges = display_edges(counts, edges, n_bins=n_bins, clip_quantile=clip_quantile)
    top = float(new_edges[-1])
    centers = bin_centers(edges)
    width = new_edges[1] - new_edges[0]

    frac, totals, over = {}, {}, {}
    for key, counts_fine in fine.items():
        totals[key] = int(counts_fine.sum())
        frac[key] = rebin(counts_fine, edges, new_edges) / totals[key]
        over[key] = int(counts_fine[centers > top].sum())

    ymax = max(frac["masked"].max(), frac["unmasked"].max()) * 1.18

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharex=True, sharey=True)
    labels = {"masked": "in a mask", "unmasked": "not in a mask"}
    colors = {"masked": C_MASKED, "unmasked": C_UNMASKED}

    for key, ax in (("masked", axes[0]), ("unmasked", axes[1])):
        other = "unmasked" if key == "masked" else "masked"

        ax.bar(bin_centers(new_edges), frac[key], width=width,
               color=colors[key], edgecolor="white", linewidth=0.5)
        # closed staircase: repeat the last value so the final bin is drawn
        ax.step(new_edges, np.append(frac[other], frac[other][-1]),
                where="post", color=colors[other], linewidth=1.4, alpha=0.75)

        # dashed rule at the group mean, labelled at the top of the line
        mean = pooled_stats[f"mean_{key}"]
        ax.axvline(mean, color=colors[key], linestyle="--", linewidth=1.4)
        right_edge = mean > new_edges[0] + 0.75 * (top - new_edges[0])
        ax.annotate(f"mean {mean:,.0f}", xy=(mean, 0.995),
                    xycoords=ax.get_xaxis_transform(),
                    xytext=(-5 if right_edge else 5, 0), textcoords="offset points",
                    ha="right" if right_edge else "left", va="top",
                    fontsize=9.5, color=colors[key])

        if over[key]:
            ax.annotate(f"{over[key]:,} / {totals[key]:,} "
                        f"({100 * over[key] / totals[key]:.2f}%) > {top:,.0f}\n"
                        f"clipped into last bin",
                        xy=(0.97, 0.86), xycoords="axes fraction",
                        ha="right", va="top", fontsize=9, color=C_INK)

        ax.set_title(f"Voxels {labels[key]}", color=C_INK)
        ax.set_xlabel("pixel brightness")
        _style_axes(ax)

    axes[0].set_ylabel("fraction of voxels in group")
    axes[0].set_xlim(new_edges[0], top)
    if log_y:
        axes[0].set_yscale("log")
    else:
        axes[0].set_ylim(0, ymax)

    # one legend for both panels: filled = the panel's own group, outline = the other
    handles = [Patch(facecolor=colors["masked"], label=labels["masked"]),
               Patch(facecolor=colors["unmasked"], label=labels["unmasked"]),
               Line2D([], [], color=C_INK, alpha=0.6, linewidth=1.4,
                      label="outline: the other group, for reference")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)

    fig.suptitle("Pixel brightness by mask presence, pooled over z", color=C_INK)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    if path is not None:
        fig.savefig(path, dpi=200)
        print(f"\nSaved plot to {path}")
    else:
        plt.show()  # blocks until closed, napari launches afterwards

    return fig


def plot_per_z(per_z, pooled_stats, path=PER_Z_PLOT_PATH):
    """How the same comparison holds up slice by slice."""
    if per_z.empty:
        print("\nNo z slices to plot")
        return None

    z = per_z["z"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- left: median brightness per slice, both groups ----------------------
    ax = axes[0]
    for key, label, color in (("masked", "in a mask", C_MASKED),
                              ("unmasked", "not in a mask", C_UNMASKED)):
        ax.fill_between(z, per_z[f"p25_{key}"], per_z[f"p75_{key}"],
                        color=color, alpha=0.18, linewidth=0)
        ax.plot(z, per_z[f"median_{key}"], marker="o", markersize=5,
                color=color, linewidth=2, label=f"{label} (median, IQR band)")

    ax.set_xlabel("z slice")
    ax.set_ylabel("pixel brightness")
    ax.set_title("Brightness per z slice", color=C_INK)
    ax.set_xticks(z)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    _style_axes(ax)

    # --- right: how separable the two groups are on each slice ---------------
    ax = axes[1]
    # one series, so no legend box -- the two reference rules are direct-labelled
    auc = per_z["auc"].to_numpy(dtype=float)
    pooled_auc = float(pooled_stats["auc"])

    ax.axhline(0.5, color=C_INK, linewidth=1.0, alpha=0.5)
    ax.annotate("0.5 = brightness says nothing", xy=(z[0], 0.5), xytext=(0, 4),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=9, color=C_INK)
    ax.plot(z, auc, marker="o", markersize=5, color=C_MASKED, linewidth=2)
    ax.axhline(pooled_auc, color=C_MASKED, linestyle="--", linewidth=1.4)
    ax.annotate(f"pooled {pooled_auc:.3f}", xy=(z[-1], pooled_auc),
                xytext=(0, -6 if pooled_auc > 0.9 else 4),
                textcoords="offset points", ha="right",
                va="top" if pooled_auc > 0.9 else "bottom",
                fontsize=9, color=C_MASKED)

    floor = 0.45 if np.isnan(auc).all() else min(0.45, float(np.nanmin(auc)) - 0.03)
    ax.set_ylim(floor, 1.01)
    ax.set_xlabel("z slice")
    ax.set_ylabel("P(masked voxel brighter than unmasked)")
    ax.set_title("Separability per z slice", color=C_INK)
    ax.set_xticks(z)
    _style_axes(ax)

    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=200)
        print(f"Saved plot to {path}")
    else:
        plt.show()

    return fig


# --- napari ------------------------------------------------------------------

def build_overlap_layers(dapi, present, counts, edges, quantile=OVERLAP_QUANTILE):
    """The two tails of the overlap, as volumes you can scroll through.

    bright_unmasked: outside every mask, brighter than `quantile` of masked voxels
    dim_masked:      inside a mask, dimmer than `quantile` of unmasked voxels
    """
    masked_thr = hist_quantile(counts[1].sum(axis=0), edges, quantile)
    unmasked_thr = hist_quantile(counts[0].sum(axis=0), edges, quantile)

    if not np.isfinite(masked_thr) or not np.isfinite(unmasked_thr):
        empty = np.zeros(dapi.shape, dtype=np.uint8)
        return empty, empty.copy(), masked_thr, unmasked_thr

    bright_unmasked = (~present & (dapi >= masked_thr)).astype(np.uint8)
    dim_masked = (present & (dapi <= unmasked_thr)).astype(np.uint8)

    print(f"\n  overlap at q={quantile:g}: "
          f"{int(bright_unmasked.sum()):,} bright-but-unmasked voxels (>{masked_thr:,.0f}), "
          f"{int(dim_masked.sum()):,} dim-but-masked voxels (<{unmasked_thr:,.0f})")
    return bright_unmasked, dim_masked, masked_thr, unmasked_thr


def launch_viewer(dapi, masks, bright_unmasked=None, dim_masked=None,
                  masked_thr=None, unmasked_thr=None):
    import napari  # imported here so the analysis runs without a Qt install

    viewer = napari.Viewer()
    viewer.add_image(
        dapi, name="DAPI (deconvolved)", colormap="gray",
        contrast_limits=[float(dapi.min()), float(np.percentile(dapi, 99.5))],
    )

    # Give each mask a random color
    viewer.add_labels(masks, name="segmentation masks", opacity=0.4)

    if bright_unmasked is not None:
        viewer.add_labels(bright_unmasked,
                          name=f"bright, unmasked (>{masked_thr:,.0f})",
                          opacity=0.6, visible=False, blending="additive")
    if dim_masked is not None:
        viewer.add_labels(dim_masked,
                          name=f"dim, masked (<{unmasked_thr:,.0f})",
                          opacity=0.6, visible=False, blending="additive")

    print("\nLaunching napari viewer")
    napari.run()
    return viewer


def main():
    dapi, masks, files = load_data(DAPI_GLOB, MASKS_PATH)

    brightness, present = vectorise_pixels(dapi, masks)
    edges, counts = build_histograms(brightness, present)

    per_z = summarise_by_z(counts, edges)
    pooled = summarise_pooled(counts, edges)
    give_brightness_summary(pooled, per_z)

    if CSV_PATH is not None:
        per_z.to_csv(CSV_PATH, index=False)
        print(f"\nSaved per-slice stats to {CSV_PATH}")

    if MAKE_PLOT:
        plot_brightness_distributions(counts, edges, pooled, path=PLOT_PATH)

    if MAKE_PER_Z_PLOT:
        plot_per_z(per_z, pooled, path=PER_Z_PLOT_PATH)

    if LAUNCH_NAPARI:
        bright_unmasked, dim_masked, masked_thr, unmasked_thr = build_overlap_layers(
            dapi, present.reshape(masks.shape), counts, edges)
        launch_viewer(dapi, masks, bright_unmasked, dim_masked, masked_thr, unmasked_thr)


if __name__ == "__main__":
    main()
