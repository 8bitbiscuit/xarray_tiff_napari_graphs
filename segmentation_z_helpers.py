"""Helpers for ``segmentation_method_comparison.ipynb``.

**Part 1** is ``segmentation_z_claude.py`` moved here as-is: same function
bodies, same defaults, same printed wording.  Everything that changed is listed
under "Changes from the script" below.

**Part 2** is new.  The script analysed one segmentation run; the notebook has to
put *two techniques* (3D true vs 3D stitched) on the same axes to choose between
them, and nothing in the script does that.

Changes from the script
-----------------------
1. ``LAYER_SPAN_CUTOFF``/``AREA_CHANGE_CUTOFF``/``CHANGE_REFERENCE``/
   ``OBJECT_STATISTIC`` were module globals read directly inside
   ``build_span_layers``, ``give_output_summary``, ``give_change_summary`` and
   ``launch_viewer``.  They are now keyword arguments defaulting to the same
   values, because the notebook calls those functions once per technique and a
   global would silently apply one technique's settings to the other.
2. ``PLOT_PATH`` defaults to ``None`` instead of ``Path("z_area_change.png")``.
   In a notebook the figure *is* the output, so the default now renders inline;
   pass ``path=...`` to write the PNG exactly as the script did.
3. ``import napari`` moved from module level into ``launch_viewer``.  napari
   pulls in Qt and is the one dependency a headless kernel usually lacks, and
   importing it at the top would stop the whole analysis from running there.
4. ``main()`` became ``analyse_masks()`` -- the same call sequence with the file
   loading, printing and viewer launch lifted out.  The notebook shows those
   steps as their own cells, and the comparison loop needs to run the analysis
   block twice without re-narrating it.
5. ``plot_area_change``'s left panel and its ``log_hist_range`` helper come from
   ``segmentation_z_claude_log.py`` rather than from ``segmentation_z_claude.py``
   -- the |Δ area| / area distribution is log-binned on a log x axis.  That is
   the only difference between the two scripts, and it is a display transform
   only: the mean and the ``cutoff x mean`` threshold are still plain arithmetic
   means, so the flagged objects and the napari layer are unchanged.

Nothing else was touched: the maths, the column names, the flagging rule and the
figure's contents are the script's.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from matplotlib.figure import Figure
from matplotlib.transforms import blended_transform_factory, offset_copy

# CONFIG DEFAULTS -- same values the script had at module level
# ----------------------------------------------------------------------------
LAYER_SPAN_CUTOFF = 1

# Area-change analysis
AREA_CHANGE_CUTOFF = 3          # flag objects changing > this many x the mean change
CHANGE_REFERENCE = "layer"      # "layer" = mean of that z-transition, "global" = mean of all transitions
OBJECT_STATISTIC = "max"        # collapse each object's transitions with "max" or "mean"
PLOT_PATH = None                # None -> render inline; Path("z_area_change.png") -> write it
# ----------------------------------------------------------------------------


# =============================================================================
# Part 1 -- from segmentation_z_claude.py
# =============================================================================


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


def check_z_span(mask_volume, layer_span_cutoff=LAYER_SPAN_CUTOFF):

    objects = ndi.find_objects(mask_volume)

    results = []

    for label, sl in enumerate(objects, start=1):
        if sl is None:
            continue

        z_slice = sl[0]
        z_start = z_slice.start
        z_end = z_slice.stop - 1
        z_span = z_slice.stop - z_slice.start

        results.append({
            "label": label,
            "z_start": z_start,
            "z_end": z_end,
            "z_span": z_span,
            "spans_multiple_layers": z_span > layer_span_cutoff
        })

    return pd.DataFrame(results)


def measure_slice_areas(mask_volume):
    """Pixel count of every label in every z slice -> array of shape (n_z, n_labels + 1)."""
    n_labels = int(mask_volume.max())
    areas = np.zeros((mask_volume.shape[0], n_labels + 1), dtype=np.int64)

    for z in range(mask_volume.shape[0]):
        counts = np.bincount(mask_volume[z].ravel(), minlength=n_labels + 1)
        areas[z] = counts[:n_labels + 1]

    return areas


CHANGE_COLUMNS = ["label", "z_from", "z_to", "area_from", "area_to",
                  "delta_area", "frac_change", "abs_frac_change"]


def check_area_change(mask_volume):
    """Area change of each object across every consecutive pair of z slices.

    One row per (object, z-transition). Only transitions where the object is
    present in *both* slices are counted, so appearing/disappearing objects
    don't register as a 100% change.
    """
    areas = measure_slice_areas(mask_volume)
    frames = []

    for z in range(areas.shape[0] - 1):
        a0, a1 = areas[z], areas[z + 1]
        labels = np.flatnonzero((a0 > 0) & (a1 > 0))
        labels = labels[labels > 0]  # drop background
        if labels.size == 0:
            continue

        delta = a1[labels] - a0[labels]
        frac = delta / a0[labels]

        frames.append(pd.DataFrame({
            "label": labels,
            "z_from": z,
            "z_to": z + 1,
            "area_from": a0[labels],
            "area_to": a1[labels],
            "delta_area": delta,
            "frac_change": frac,
            "abs_frac_change": np.abs(frac),
        }))

    if not frames:
        return pd.DataFrame(columns=CHANGE_COLUMNS)

    return pd.concat(frames, ignore_index=True)


def summarise_layer_change(changes):
    """Mean |area change| per z-transition (i.e. per layer boundary)."""
    if changes.empty:
        return pd.DataFrame(columns=["z_from", "z_to", "mean_change", "std_change", "n_objects"])

    layer_stats = (
        changes
        .groupby(["z_from", "z_to"], as_index=False)
        .agg(mean_change=("abs_frac_change", "mean"),
             std_change=("abs_frac_change", "std"),
             n_objects=("label", "size"))
    )
    return layer_stats


def flag_area_outliers(changes, layer_stats,
                       cutoff=AREA_CHANGE_CUTOFF,
                       reference=CHANGE_REFERENCE,
                       statistic=OBJECT_STATISTIC):
    """Per-object area-change summary + flag for objects above cutoff x mean.

    reference="layer"  -> each transition is compared to the mean of its own
                          z-transition (controls for systematic drift in z)
    reference="global" -> everything is compared to the mean over all transitions
    """
    if changes.empty:
        return pd.DataFrame(columns=["label", "n_transitions", "mean_abs_change",
                                     "max_abs_change", "change_ratio", "large_area_change"])

    changes = changes.copy()

    if reference == "layer":
        ref = changes["z_from"].map(layer_stats.set_index("z_from")["mean_change"])
    elif reference == "global":
        ref = pd.Series(changes["abs_frac_change"].mean(), index=changes.index)
    else:
        raise ValueError(f"reference must be 'layer' or 'global', got {reference!r}")

    ref = ref.replace(0, np.nan)  # avoid divide-by-zero on a perfectly stable layer
    changes["change_ratio"] = changes["abs_frac_change"] / ref

    per_object = (
        changes
        .groupby("label", as_index=False)
        .agg(n_transitions=("abs_frac_change", "size"),
             mean_abs_change=("abs_frac_change", "mean"),
             max_abs_change=("abs_frac_change", "max"),
             change_ratio=("change_ratio", statistic))
    )
    per_object["large_area_change"] = per_object["change_ratio"] > cutoff

    return per_object


def build_span_layers(masks, df, layer_span_cutoff=LAYER_SPAN_CUTOFF):
    normal_cutoff = df.loc[df["z_span"] == layer_span_cutoff, "label"].to_numpy()
    next_cutoff = df.loc[df["z_span"] == layer_span_cutoff + 1, "label"].to_numpy()

    normal_cutoff_layer = np.where(np.isin(masks, normal_cutoff), masks, 0).astype(masks.dtype)
    next_cutoff_layer = np.where(np.isin(masks, next_cutoff), masks, 0).astype(masks.dtype)

    return normal_cutoff_layer, next_cutoff_layer


def build_change_layer(masks, df):
    flagged = df.loc[df["large_area_change"], "label"].to_numpy()
    return np.where(np.isin(masks, flagged), masks, 0).astype(masks.dtype)


def give_output_summary(df, layer_span_cutoff=LAYER_SPAN_CUTOFF):
    n_single = int((df["z_span"] == 1).sum())
    n_two = int((df["z_span"] == 2).sum())
    n_multi = int(df["spans_multiple_layers"].sum())

    print("\n=== Z-SPAN SUMMARY ===")
    print(f"  n_objects: {len(df)}")
    print(f"  n_single_slice (z_span==1): {n_single}")
    print(f"  n_two_slice (z_span==2): {n_two}")
    print(f"  n_spans_multiple_layers (z_span>{layer_span_cutoff}): {n_multi}")


def give_change_summary(df, changes, layer_stats,
                        cutoff=AREA_CHANGE_CUTOFF,
                        reference=CHANGE_REFERENCE,
                        statistic=OBJECT_STATISTIC):
    print("\n=== AREA-CHANGE SUMMARY ===")
    if changes.empty:
        print("  no objects span more than one z slice; nothing to compare")
        return

    print(f"  n_transitions measured: {len(changes)}")
    print(f"  mean |Δarea|/area: {changes['abs_frac_change'].mean():.3f}")
    print(f"  median |Δarea|/area: {changes['abs_frac_change'].median():.3f}")
    print(f"  reference: {reference}  |  object statistic: {statistic}")
    print(f"  n_large_area_change (> {cutoff}x mean): "
          f"{int(df['large_area_change'].sum())}")

    print("\n  per-layer mean |Δarea|/area:")
    for row in layer_stats.itertuples():
        std = 0.0 if pd.isna(row.std_change) else row.std_change
        print(f"    z {row.z_from}->{row.z_to}: {row.mean_change:.3f} "
              f"± {std:.3f}  (n={row.n_objects})")


def log_hist_range(vals, hist_xmin=None, hist_xmax=None, hist_decades=4):
    """Decide the (lo, top) span of the log-binned histogram.

    ``top`` defaults to the largest observed change: on a log axis the long right
    tail costs almost no width, so unlike the linear version there is no reason
    to truncate it by default.  ``lo`` defaults to the decade holding the
    smallest *positive* change, floored at ``hist_decades`` below ``top`` so one
    near-zero transition can't stretch the axis over ten empty decades.
    """
    top = float(hist_xmax) if hist_xmax is not None else float(vals.max())
    if not np.isfinite(top) or top <= 0:
        top = 1.0  # degenerate: every transition was perfectly stable

    floor_lo = top / 10.0 ** hist_decades

    if hist_xmin is not None:
        lo = float(hist_xmin)
    else:
        positive = vals[vals > 0]
        auto_lo = 10.0 ** np.floor(np.log10(positive.min())) if positive.size else floor_lo
        lo = max(auto_lo, floor_lo)

    lo = min(lo, top / 10.0)  # always show at least one decade
    return lo, top


def plot_area_change(changes, layer_stats, cutoff=AREA_CHANGE_CUTOFF, path=PLOT_PATH,
                     hist_xmax=None, n_bins=50, hist_ymax=None, layer_ymax=2.5,
                     spread="iqr", hist_xmin=None, hist_decades=4):
    """Same figure as the linear version, with a log-scaled left panel.

    ``hist_xmax`` and ``hist_ymax`` now default to ``None`` (autoscale) because
    the linear defaults no longer suit the transformed axis: the log axis fits
    the whole tail without clipping, and log-spaced bins are narrow at the low
    end, so per-bin mass no longer approaches the old 0.60 ceiling.  Passing
    either explicitly still clips/limits exactly as before.
    """

    if changes.empty:
        print("\nNo z-transitions to plot")
        return None

    vals = changes["abs_frac_change"].to_numpy()
    global_mean = vals.mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- left panel: distribution of per-transition area change (log x) ------
    ax = axes[0]
    lo, top = log_hist_range(vals, hist_xmin=hist_xmin, hist_xmax=hist_xmax,
                             hist_decades=hist_decades)
    bins = np.logspace(np.log10(lo), np.log10(top), n_bins + 1)

    n_over = int((vals > top).sum())
    n_under = int((vals < lo).sum())   # includes the exactly-zero transitions
    n_zero = int((vals == 0).sum())

    # Log bins can't hold zero or negative values, so the tails are folded into
    # the edge bins and reported in the annotation rather than silently dropped.
    ax.hist(np.clip(vals, lo, top), bins=bins,
            weights=np.full(vals.size, 1.0 / vals.size),
            color="0.5", edgecolor="white", zorder=2)
    ax.axvline(global_mean, color="tab:blue",
               label=f"mean = {global_mean:.3f}")
    ax.axvline(cutoff * global_mean, color="tab:red", linestyle="--",
               label=f"{cutoff}x mean = {cutoff * global_mean:.3f}")

    notes = []
    if n_under:
        what = ("no area change" if n_zero == n_under
                else f"< {lo:g}" + (f" ({n_zero} of them 0)" if n_zero else ""))
        notes.append(f"{n_under} / {vals.size} "
                     f"({100 * n_under / vals.size:.1f}%) {what}\n"
                     f"clipped into first bin")
    if n_over:
        notes.append(f"{n_over} / {vals.size} "
                     f"({100 * n_over / vals.size:.1f}%) > {top:g}\n"
                     f"clipped into last bin")
    if notes:
        ax.annotate("\n".join(notes),
                    xy=(0.97, 0.95), xycoords="axes fraction",
                    ha="right", va="top", fontsize=9, color="0.3")

    ax.set_xscale("log")
    ax.set_xlim(lo, top)
    # Plain decimals ("0.01") read better than 10^-2 for a fractional change.
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _pos: f"{v:g}"))
    if hist_ymax is not None:
        ax.set_ylim(0, hist_ymax)
    ax.xaxis.grid(True, which="major", color="0.9", linewidth=0.8, zorder=0)
    ax.xaxis.grid(True, which="minor", color="0.94", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("|Δ area| / area  (per z transition, log scale)")
    ax.set_ylabel("fraction of object-transitions")
    ax.set_title("Distribution of between-layer area change (log scale)")
    ax.legend(loc="upper left")

    # --- right panel: mean change per z transition ---------------------------
    ax = axes[1]
    layer_stats = layer_stats.sort_values("z_from")
    z = layer_stats["z_from"].to_numpy()
    mean = layer_stats["mean_change"].to_numpy()
    upper = mean.copy()

    if spread == "iqr":
        q = (changes.groupby("z_from")["abs_frac_change"]
             .quantile([0.25, 0.5, 0.75])
             .unstack()
             .reindex(z))
        lo, med, hi = q[0.25].to_numpy(), q[0.5].to_numpy(), q[0.75].to_numpy()
        ax.fill_between(z, lo, hi, color="tab:blue", alpha=0.20, linewidth=0,
                        label="IQR (25th–75th pct)")
        ax.plot(z, med, color="tab:blue", linestyle=":", linewidth=1.4,
                label="median")
        upper = hi
    elif spread == "sem":
        sd = layer_stats["std_change"].fillna(0).to_numpy()
        n = layer_stats["n_objects"].to_numpy()
        sem = sd / np.sqrt(np.maximum(n, 1))
        ax.errorbar(z, mean, yerr=sem, fmt="none", capsize=3, color="tab:blue",
                    label="mean ± s.e.m.")
        upper = mean + sem
    elif spread == "sd":
        sd = layer_stats["std_change"].fillna(0).to_numpy()
        yerr = np.vstack([np.minimum(sd, mean), sd])  # |Δ|/area can't be < 0
        ax.errorbar(z, mean, yerr=yerr, fmt="none", capsize=3, color="tab:blue",
                    alpha=0.5, label="mean ± sd (clamped at 0)")
        upper = mean + sd
    elif spread is not None:
        raise ValueError(f"spread must be 'iqr', 'sem', 'sd' or None, got {spread!r}")

    ax.plot(z, mean, marker="o", color="tab:blue", label="mean (flag reference)")
    ax.plot(z, cutoff * mean, linestyle="--", color="tab:red",
            label=f"{cutoff}x mean (flag threshold)")

    ax.set_xlim(z.min() - 0.5, z.max() + 0.5)
    if layer_ymax is not None:
        ax.set_ylim(0, layer_ymax)
        n_clipped = int((upper > layer_ymax).sum())
        if n_clipped:
            ax.annotate(f"spread clipped on {n_clipped} transition(s)",
                        xy=(0.97, 0.70), xycoords="axes fraction",
                        ha="right", va="top", fontsize=9, color="0.3")

    ax.set_xlabel("z transition (z -> z+1)")
    ax.set_ylabel("|Δ area| / area")
    ax.set_title("Mean area change per layer")
    ax.set_xticks(z)
    ax.legend(loc="upper right")

    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=200)
        print(f"\nSaved plot to {path}")
    else:
        plt.show()  # blocks until closed, napari launches afterwards

    return fig


def launch_viewer(dapi, masks, single_layer_vol, two_layer_vol, change_vol=None,
                  layer_span_cutoff=LAYER_SPAN_CUTOFF, cutoff=AREA_CHANGE_CUTOFF,
                  title=None):

    import napari  # Qt import, kept out of module scope so a headless kernel can import this file

    viewer = napari.Viewer(title=title) if title else napari.Viewer()
    viewer.add_image(
        dapi, name="DAPI (deconvolved)", colormap="gray",
        contrast_limits=[float(dapi.min()), float(np.percentile(dapi, 99.5))],
    )

    # Give each mask a random color
    viewer.add_labels(masks, name="segmentation masks", opacity=0.4)
    viewer.add_labels(single_layer_vol, name=F"{layer_span_cutoff}-slice nuclei", opacity=0.6, visible=False)
    viewer.add_labels(two_layer_vol, name=F"{layer_span_cutoff + 1}-slice nuclei", opacity=0.6, visible=False)
    if change_vol is not None:
        viewer.add_labels(change_vol, name=F">{cutoff}x mean area change",
                          opacity=0.6, visible=False, blending='additive')

    print("\nLaunching napari viewer")
    napari.run()
    return viewer


def analyse_masks(masks,
                  layer_span_cutoff=LAYER_SPAN_CUTOFF,
                  cutoff=AREA_CHANGE_CUTOFF,
                  reference=CHANGE_REFERENCE,
                  statistic=OBJECT_STATISTIC):
    """The analysis block of the script's ``main()``, with I/O and napari removed.

    Returns ``(df, changes, layer_stats)``: one row per object, one row per
    (object, z-transition), one row per z-transition.
    """
    df = check_z_span(masks, layer_span_cutoff=layer_span_cutoff)

    changes = check_area_change(masks)
    layer_stats = summarise_layer_change(changes)
    per_object = flag_area_outliers(changes, layer_stats,
                                    cutoff=cutoff,
                                    reference=reference,
                                    statistic=statistic)

    df = df.merge(per_object, on="label", how="left")
    # the script's `.fillna(False).astype(bool)`, written so pandas >= 2.2 does
    # not warn about downcasting the object column the merge leaves behind.
    # Objects with no transition to score are NaN here, and NaN is not True.
    df["large_area_change"] = df["large_area_change"].eq(True)

    return df, changes, layer_stats


# =============================================================================
# Part 2 -- comparing two techniques (new; the script only ever saw one run)
# =============================================================================

# Palette lifted from zspan/plotting.py so the notebook's figures match the
# package's.  Fixed slot order is the colour-vision-deficiency safety mechanism:
# assign slots in order, never generate a new hue.
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


def tag(frame: pd.DataFrame, method: str, column: str = "method") -> pd.DataFrame:
    """Copy of ``frame`` carrying the technique it came from as a column."""
    out = frame.copy()
    out[column] = method
    return out


def _method_colors(methods) -> dict:
    return {name: CATEGORICAL[i] for i, name in enumerate(methods)}


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
    colors = _method_colors(methods)
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


def compare_layer_change(changes: pd.DataFrame, group: str = "method",
                         spread: str = "iqr", ymax: float | None = 2.5,
                         title: str = "Area change between adjacent layers") -> Figure:
    """Mean |Δ area|/area per z-transition, one line per technique.

    Same quantity as the right panel of ``plot_area_change``, with the flag
    threshold dropped: two techniques' means and thresholds on one axes is four
    lines to read where two carry the comparison.
    """
    methods = list(dict.fromkeys(changes[group]))
    colors = _method_colors(methods)
    clipped = 0

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for i, name in enumerate(methods):
        sub = changes[changes[group] == name]
        stats = summarise_layer_change(sub).sort_values("z_from")
        z = stats["z_from"].to_numpy()
        mean = stats["mean_change"].to_numpy()

        if spread == "iqr":
            q = (sub.groupby("z_from")["abs_frac_change"]
                 .quantile([0.25, 0.75]).unstack().reindex(z))
            ax.fill_between(z, q[0.25].to_numpy(), q[0.75].to_numpy(),
                            color=colors[name], alpha=0.16, linewidth=0)
        elif spread is not None:
            raise ValueError(f"spread must be 'iqr' or None, got {spread!r}")

        ax.plot(z, mean, marker=MARKERS[i % len(MARKERS)], markersize=6,
                linewidth=2, color=colors[name], label=name, zorder=3)
        if ymax is not None:
            clipped += int((mean > ymax).sum())

    ax.set_xlabel("z transition (z -> z+1)")
    ax.set_ylabel("|Δ area| / area")
    ax.set_title(title, pad=22, loc="left")
    ax.annotate("line = mean per transition, band = IQR (25th–75th pct)",
                xy=(0, 1.03), xycoords="axes fraction", fontsize=9, color=INK_SECONDARY)
    if ymax is not None:
        ax.set_ylim(0, ymax)
        if clipped:  # never let a line leave the axes silently
            ax.annotate(f"{clipped} mean(s) above {ymax:g}, clipped",
                        xy=(0.99, 0.02), xycoords="axes fraction",
                        ha="right", fontsize=9, color=INK_SECONDARY)
    ax.set_xticks(sorted(changes["z_from"].unique()))
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.spines["bottom"].set_color(GRID)
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def compare_headline(labels: pd.DataFrame, changes: pd.DataFrame,
                     group: str = "method",
                     layer_span_cutoff: int = LAYER_SPAN_CUTOFF) -> pd.DataFrame:
    """One row per technique: the numbers the choice actually turns on."""
    flags = labels.assign(
        _single=labels["z_span"] == 1,
        _multi=labels["z_span"] > layer_span_cutoff,
    )
    per_span = flags.groupby(group, sort=False)
    spans = pd.DataFrame({
        "n_masks": per_span.size(),
        "pct_single_slice": 100 * per_span["_single"].mean(),
        "pct_multi_layer": 100 * per_span["_multi"].mean(),
        "mean_z_span": per_span["z_span"].mean(),
        "max_z_span": per_span["z_span"].max(),
        "pct_flagged": 100 * per_span["large_area_change"].mean(),
    })

    per_change = changes.groupby(group, sort=False)
    change_cols = pd.DataFrame({
        "n_transitions": per_change.size(),
        "mean_abs_change": per_change["abs_frac_change"].mean(),
        "median_abs_change": per_change["abs_frac_change"].median(),
    })

    return spans.join(change_cols).round(3)


# =============================================================================
# Part 3 -- every FOV in a region, scored side by side
# =============================================================================


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


def scan_region(found, layer_span_cutoff=LAYER_SPAN_CUTOFF,
                cutoff=AREA_CHANGE_CUTOFF, reference=CHANGE_REFERENCE,
                statistic=OBJECT_STATISTIC, verbose=True):
    """``analyse_masks`` on every row of ``found``; returns tidy (labels, changes).

    Only the mask volume is read -- the DAPI stack is what ``load_data`` checks
    against and is not needed for either metric, and re-reading it per FOV would
    dominate the runtime.  One volume is held at a time, so peak memory is set by
    the largest FOV rather than by how many there are.
    """
    label_frames, change_frames = [], []

    for row in found.itertuples():
        masks = tifffile.imread(row.path)
        if masks.ndim == 2:
            masks = masks[np.newaxis]

        labels_i, changes_i, _ = analyse_masks(
            masks, layer_span_cutoff=layer_span_cutoff, cutoff=cutoff,
            reference=reference, statistic=statistic)
        del masks

        label_frames.append(labels_i.assign(fov=row.fov, method=row.method))
        change_frames.append(changes_i.assign(fov=row.fov, method=row.method))

        if verbose:
            print(f"  {row.fov:<10} {str(row.method):<12} "
                  f"{len(labels_i):>6} masks  {len(changes_i):>6} transitions")

    return (pd.concat(label_frames, ignore_index=True),
            pd.concat(change_frames, ignore_index=True))


def fov_metrics(labels, changes, by=("fov", "method"),
                layer_span_cutoff=LAYER_SPAN_CUTOFF, thin_max=2) -> pd.DataFrame:
    """One row per FOV per technique -- the raw numbers the scorecard draws."""
    by = list(by)
    flags = labels.assign(
        _single=labels["z_span"] == 1,
        _thin=labels["z_span"] <= thin_max,
        _multi=labels["z_span"] > layer_span_cutoff,
    )
    per_span = flags.groupby(by, sort=False, observed=True)
    metrics = pd.DataFrame({
        "pct_single_slice": 100 * per_span["_single"].mean(),
        "pct_thin": 100 * per_span["_thin"].mean(),
        "pct_flagged": 100 * per_span["large_area_change"].mean(),
        "n_masks": per_span.size(),
        "mean_z_span": per_span["z_span"].mean(),
        "max_z_span": per_span["z_span"].max(),
    })

    per_change = changes.groupby(by, sort=False, observed=True)
    metrics = metrics.join(pd.DataFrame({
        "mean_abs_change": per_change["abs_frac_change"].mean(),
        "median_abs_change": per_change["abs_frac_change"].median(),
    }))

    return metrics


# (column, group, header, format).  Every column here is lower-is-better; a
# higher-is-better metric belongs in the spec negated.
FOV_TABLE_SPEC = (
    ("pct_single_slice",  "Thinness",  "% single\nslice", "{:.0f}"),
    ("pct_thin",          "Thinness",  "% thin\n(<=2)",   "{:.0f}"),
    ("mean_abs_change",   "Stability", "mean\n|Δa|/a",    "{:.2f}"),
    ("median_abs_change", "Stability", "median\n|Δa|/a",  "{:.2f}"),
    ("pct_flagged",       "Stability", "%\nflagged",      "{:.0f}"),
)


def score_fovs(metrics: pd.DataFrame) -> pd.DataFrame:
    """Both scores start at 1.00 and pay one point per percent of defect.

    ``Thinness  = 1 - (pct_thin + pct_single_slice) / 100``.  A single-plane mask
    is counted in both columns, so it costs twice what a two-plane one does --
    one plane is a failed mask outright, two is only what a ``min_z = 3`` filter
    happens to discard.

    ``Stability = 1 - pct_flagged / 100``.

    These are *absolute*, not ranked: a score means the same thing in every
    region, and adding a FOV to the table moves nothing.  Enough defect drives a
    score below 0 -- the number is reported as computed and only the drawn bar
    is clipped at 0, since a 0.00 and a -0.40 are not the same run.
    """
    return pd.DataFrame({
        "Thinness": 1 - (metrics["pct_thin"] + metrics["pct_single_slice"]) / 100,
        "Stability": 1 - metrics["pct_flagged"] / 100,
    }, index=metrics.index)


def plot_fov_table(metrics: pd.DataFrame, spec=FOV_TABLE_SPEC,
                   title: str = "FOV comparison", row_sep: str = " · ") -> Figure:
    """Scorecard: one row per FOV per technique, raw values then group scores."""
    aggregate = score_fovs(metrics)

    rows = [row_sep.join(str(part) for part in idx) if isinstance(idx, tuple) else str(idx)
            for idx in metrics.index]
    n_rows, n_cols, n_agg = len(rows), len(spec), aggregate.shape[1]

    fig, (ax, ax_agg) = plt.subplots(
        1, 2, figsize=(0.95 * n_cols + 1.5 * n_agg + 3.2, 0.42 * n_rows + 2.1),
        gridspec_kw={"width_ratios": [n_cols, 1.6 * n_agg], "wspace": 0.06})

    for axis in (ax, ax_agg):
        axis.set_ylim(n_rows - 0.5, -0.5)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.tick_params(length=0)
        for i in range(1, n_rows):        # dotted separators, as in the reference
            axis.axhline(i - 0.5, color=GRID, linestyle=":", linewidth=1.0, zorder=0)
        axis.axhline(-0.5, color=INK, linewidth=1.2, zorder=1)
        axis.axhline(n_rows - 0.5, color=INK, linewidth=1.2, zorder=1)

    # --- left: the raw numbers -----------------------------------------------
    ax.set_xlim(-0.5, n_cols - 0.5)
    for j, (col, _group, _header, fmt) in enumerate(spec):
        for i, value in enumerate(metrics[col].to_numpy()):
            ax.text(j, i, fmt.format(value), ha="center", va="center",
                    fontsize=9.5, color=INK)

    ax.set_xticks(range(n_cols), [header for _c, _g, header, _f in spec], fontsize=9)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(n_rows), rows, fontsize=9.5, fontweight="bold")

    # --- right: group score, bar in cell -------------------------------------
    ax_agg.set_xlim(0, n_agg)
    for j, name in enumerate(aggregate.columns):
        for i, value in enumerate(aggregate[name].to_numpy()):
            drawn = min(max(float(value), 0.0), 1.0)   # the bar clips; the label does not
            ax_agg.barh(i, 0.9 * drawn, left=j + 0.05, height=0.52,
                        color=SEQUENTIAL_BLUE[-5], zorder=2)
            # label rides the end of the bar: inside it while there is room,
            # just outside once the bar is too short to hold the text
            end = j + 0.05 + 0.9 * drawn
            inside = value >= 0.30
            ax_agg.text(end + (-0.04 if inside else 0.04), i, f"{value:.2f}",
                        ha="right" if inside else "left", va="center", fontsize=9,
                        color=SURFACE if inside else INK_SECONDARY, zorder=3)
    for j in range(1, n_agg):
        ax_agg.axvline(j, color=GRID, linewidth=0.8, zorder=1)

    ax_agg.set_xticks(np.arange(n_agg) + 0.5, list(aggregate.columns), fontsize=9)
    ax_agg.xaxis.set_ticks_position("top")
    ax_agg.set_yticks([])

    # --- group headers above the column labels -------------------------------
    spans, start = [], 0
    for j in range(1, n_cols + 1):
        if j == n_cols or spec[j][1] != spec[start][1]:
            spans.append((spec[start][1], start, j - 1))
            start = j
    spans = [(name, lo - 0.35, hi + 0.35, ax) for name, lo, hi in spans]
    spans.append(("Aggregate score", 0.05, n_agg - 0.05, ax_agg))

    for name, lo, hi, axis in spans:
        # x in data, y pinned to the top of the axes and pushed above the labels
        anchor = blended_transform_factory(axis.transData, axis.transAxes)
        rule = offset_copy(anchor, fig=fig, x=0, y=30, units="points")
        axis.plot([lo, hi], [1.0, 1.0], transform=rule, color=GRID, linewidth=1.2,
                  clip_on=False, zorder=4)
        axis.annotate(name, xy=((lo + hi) / 2, 1.0), xytext=(0, 34),
                      xycoords=anchor, textcoords="offset points",
                      ha="center", va="bottom", fontsize=9.5, color=INK_SECONDARY,
                      annotation_clip=False)

    fig.suptitle(title, x=0.02, y=0.97, ha="left", va="top", fontsize=11.5,
                 fontweight="bold", color=INK)
    fig.subplots_adjust(top=0.80, bottom=0.04, left=0.16, right=0.98)
    return fig
