"""z-span / area-change analysis, with objects flagged by an absolute cutoff on
the log scale of |Δ area| / area instead of by a multiple of the mean change.

The script it was written as a variant of, ``segmentation_z_claude.py``, has
since been removed from the repo; this one is the version that was kept.  What
that baseline did, and what changed here, is below — the choices only read as
choices against it.

Changes from that baseline
--------------------------
1. The flag rule.  The script asked "is this transition more than
   ``AREA_CHANGE_CUTOFF`` x the mean change of its layer?".  This version asks
   "is log10(|Δ area| / area) above ``log10(CHANGE_CUTOFF)``?", i.e. it puts a
   fixed line on the log axis of the left-hand histogram and marks everything to
   the right of it.  ``CHANGE_CUTOFF = 1.0`` is the "1" gridline: the object's
   cross-section changed by 100% or more between two adjacent slices.
2. ``CHANGE_REFERENCE`` is gone.  The cutoff no longer depends on the mean of
   anything, so there is nothing to pick a reference for and
   ``flag_area_outliers`` no longer takes ``layer_stats``.  The per-layer means
   are still computed -- they are what the right-hand panel and the printed
   summary show -- they just aren't the threshold any more.
3. Statistics that collapse many transitions into one number are computed in log
   space, since that is the scale the cutoff now lives on: ``OBJECT_STATISTIC =
   "mean"`` averages log10 values (a geometric mean of the ratios) rather than
   the raw ratios, and the per-layer summary reports a geometric mean alongside
   the arithmetic one.  ``OBJECT_STATISTIC = "max"`` is unaffected -- log10 is
   monotonic, so the largest log is the largest ratio.  The two now ask
   different questions: "max" flags an object that *ever* jumped, "mean" flags
   one that is *typically* unstable.
   Areas are integer pixel counts, so a transition can come out at exactly 0 and
   log10(0) is -inf.  Those are censored, not stable-to-infinite-precision: the
   change was smaller than one pixel, so they are floored at half a pixel's
   worth (``0.5 / area_from``) instead of at a global constant, which keeps the
   floor proportional to the object.  It only ever touches values far below any
   usable cutoff, so no flag depends on it.
4. The left panel is log-binned on a log x axis, because the cutoff is only
   readable there, and the right panel is log-scaled in y with a flat cutoff
   line -- the
   threshold is the same at every z now, so the interesting question is how much
   of each layer sits above it.  The right panel's second axis shows exactly
   that: the % of transitions flagged per z.
5. ``import napari`` moved into ``launch_viewer`` so the analysis and the plot
   still run on a machine without Qt.

A note on the metric: ``abs_frac_change`` is |Δ area| / area_from, so shrinkage
can never reach 1.0 (an object that shrank by 100% is gone, and transitions
where the object vanishes are not counted at all) while growth is unbounded.  A
cutoff of 1.0 on that metric therefore only ever flags objects that *grew*.  Set
``CHANGE_METRIC = "sym_frac_change"`` for the symmetric version, where 1.0 means
"doubled or halved" in either direction.
"""

import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

# CONFIG
# ----------------------------------------------------------------------------
MASKS_DIR = Path("../data/segmentations_3d_stitched/cpdino/decon/VePo/region_UCI-5224/fov_07")
DAPI_DIR = Path("../data/patches/VePo/region_UCI-5224/fov_07")

MASKS_PATH = MASKS_DIR / "masks.tif"
DAPI_GLOB = str(DAPI_DIR / "DAPI_decon_z*.tif") # Each z index

LAYER_SPAN_CUTOFF = 1

# Area-change analysis
CHANGE_CUTOFF = 1.0             # flag objects above this |Δarea|/area, i.e. above
                                # log10(CHANGE_CUTOFF) on the log axis. 1.0 = the "1"
                                # gridline. For "log10 of the ratio > 1" instead, set 10.
CHANGE_METRIC = "abs_frac_change"   # "abs_frac_change"  = |Δarea| / area_from (the script's
                                    #   metric; shrinkage tops out at 1, so a 1.0 cutoff
                                    #   flags growth only)
                                    # "sym_frac_change" = |Δarea| / min(area_from, area_to)
                                    #   symmetric: 1.0 = "doubled or halved", either way
OBJECT_STATISTIC = "max"        # collapse each object's transitions with "max" (ever jumped)
                                # or "mean" (typically unstable); both read in log space
MAKE_PLOT = True
PLOT_PATH = Path("z_area_change_log.png")            # e.g. Path(...); None -> show interactively

LAUNCH_NAPARI = True
# ----------------------------------------------------------------------------

METRIC_LABELS = {
    "abs_frac_change": "|Δ area| / area",
    "sym_frac_change": "|Δ area| / min(area)",
}


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


def to_log10(values, floor):
    """log10 of a change metric, with the exactly-stable transitions censored.

    Areas are integer pixel counts, so a transition where the area didn't move
    is exactly 0 and log10(0) is -inf, which would poison every mean taken in
    log space.  It doesn't mean "stable to infinite precision" either -- it means
    the change was under one pixel -- so ``floor`` (scalar or per-row) stands in
    for it.  Everything it touches is decades below any usable cutoff.
    """
    return np.log10(np.maximum(np.asarray(values, dtype=float), floor))


CHANGE_COLUMNS = ["label", "z_from", "z_to", "area_from", "area_to",
                  "delta_area", "frac_change", "abs_frac_change",
                  "sym_frac_change", "change_floor", "log_change"]


def check_area_change(mask_volume, metric=CHANGE_METRIC):
    """Area change of each object across every consecutive pair of z slices.

    One row per (object, z-transition). Only transitions where the object is
    present in *both* slices are counted, so appearing/disappearing objects
    don't register as a 100% change.

    ``log_change`` is log10 of whichever column ``metric`` names -- that is the
    scale the cutoff is applied on.
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
        # Symmetric alternative: dividing by the smaller of the two areas makes a
        # halving score the same as a doubling, which |delta|/area_from can't do.
        sym = np.abs(delta) / np.minimum(a0[labels], a1[labels])

        frames.append(pd.DataFrame({
            "label": labels,
            "z_from": z,
            "z_to": z + 1,
            "area_from": a0[labels],
            "area_to": a1[labels],
            "delta_area": delta,
            "frac_change": frac,
            "abs_frac_change": np.abs(frac),
            "sym_frac_change": sym,
            # Half a pixel: the smallest change the metric could have resolved,
            # and so what an exactly-zero transition stands in for.
            "change_floor": 0.5 / a0[labels],
        }))

    if not frames:
        return pd.DataFrame(columns=CHANGE_COLUMNS)

    changes = pd.concat(frames, ignore_index=True)
    changes["log_change"] = to_log10(changes[metric].to_numpy(),
                                     changes["change_floor"].to_numpy())
    return changes


LAYER_COLUMNS = ["z_from", "z_to", "mean_change", "std_change", "median_change",
                 "mean_log_change", "geo_mean_change", "n_objects", "n_flagged",
                 "frac_flagged"]


def summarise_layer_change(changes, cutoff=CHANGE_CUTOFF, metric=CHANGE_METRIC):
    """Per z-transition summary of the change metric, plus how many it flags.

    ``geo_mean_change`` is the log-space centre (10 ** mean of log10), which is
    the one that lines up with a cutoff read off the log axis; ``mean_change``
    is kept because it is what the original script printed.
    """
    if changes.empty:
        return pd.DataFrame(columns=LAYER_COLUMNS)

    layer_stats = (
        changes
        .assign(_flagged=changes[metric] > cutoff)
        .groupby(["z_from", "z_to"], as_index=False)
        .agg(mean_change=(metric, "mean"),
             std_change=(metric, "std"),
             median_change=(metric, "median"),
             mean_log_change=("log_change", "mean"),
             n_objects=("label", "size"),
             n_flagged=("_flagged", "sum"))
    )
    layer_stats["geo_mean_change"] = 10.0 ** layer_stats["mean_log_change"]
    layer_stats["frac_flagged"] = layer_stats["n_flagged"] / layer_stats["n_objects"]

    return layer_stats[LAYER_COLUMNS]


OBJECT_COLUMNS = ["label", "n_transitions", "mean_abs_change", "max_abs_change",
                  "mean_log_change", "max_log_change", "log_change", "change_stat",
                  "large_area_change"]


def flag_area_outliers(changes,
                       cutoff=CHANGE_CUTOFF,
                       statistic=OBJECT_STATISTIC,
                       metric=CHANGE_METRIC):
    """Per-object area-change summary + flag for objects past the log cutoff.

    The threshold is absolute: ``log_change > log10(cutoff)``, the same line at
    every z.  No ``layer_stats`` argument, because nothing here is measured
    relative to a layer mean any more.

    statistic="max"  -> flag the object if any single transition crossed the line
    statistic="mean" -> flag on the mean of its log10 values, i.e. the geometric
                        mean of its ratios (a straight arithmetic mean would let
                        one huge transition drag a stable object over)
    """
    if changes.empty:
        return pd.DataFrame(columns=OBJECT_COLUMNS)

    if statistic not in ("max", "mean"):
        raise ValueError(f"statistic must be 'max' or 'mean', got {statistic!r}")

    per_object = (
        changes
        .groupby("label", as_index=False)
        .agg(n_transitions=(metric, "size"),
             mean_abs_change=(metric, "mean"),
             max_abs_change=(metric, "max"),
             mean_log_change=("log_change", "mean"),
             max_log_change=("log_change", "max"))
    )

    per_object["log_change"] = per_object[f"{statistic}_log_change"]
    per_object["change_stat"] = 10.0 ** per_object["log_change"]
    per_object["large_area_change"] = per_object["log_change"] > np.log10(cutoff)

    return per_object[OBJECT_COLUMNS]


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
                        cutoff=CHANGE_CUTOFF,
                        statistic=OBJECT_STATISTIC,
                        metric=CHANGE_METRIC):
    label = METRIC_LABELS[metric]

    print("\n=== AREA-CHANGE SUMMARY (log cutoff) ===")
    if changes.empty:
        print("  no objects span more than one z slice; nothing to compare")
        return

    vals = changes[metric]
    n_over = int((vals > cutoff).sum())
    n_zero = int((vals == 0).sum())

    print(f"  n_transitions measured: {len(changes)}")
    print(f"  metric: {label}  |  object statistic: {statistic} (in log space)")
    print(f"  mean {label}: {vals.mean():.3f}")
    print(f"  median {label}: {vals.median():.3f}")
    print(f"  geometric mean {label}: {10 ** changes['log_change'].mean():.3f}"
          + (f"  ({n_zero} sub-pixel transitions censored at 0.5/area)" if n_zero else ""))
    print(f"  cutoff: {label} > {cutoff:g}  (log10 > {np.log10(cutoff):.2f})")
    print(f"  n_transitions over cutoff: {n_over} / {len(changes)} "
          f"({100 * n_over / len(changes):.1f}%)")
    print(f"  n_large_area_change (objects): {int(df['large_area_change'].sum())}")

    if metric == "abs_frac_change" and cutoff >= 1:
        n_halved = int((changes["sym_frac_change"] > cutoff).sum())
        print(f"  note: |Δarea|/area_from can't exceed 1 on the shrinking side, so this "
              f"cutoff\n        flags growth only; the symmetric metric would flag "
              f"{n_halved} transitions")

    print(f"\n  per-layer {label}:")
    for row in layer_stats.itertuples():
        std = 0.0 if pd.isna(row.std_change) else row.std_change
        print(f"    z {row.z_from}->{row.z_to}: mean {row.mean_change:.3f} ± {std:.3f}  "
              f"geo {row.geo_mean_change:.3f}  "
              f"flagged {row.n_flagged}/{row.n_objects} ({100 * row.frac_flagged:.1f}%)")


def log_hist_range(vals, hist_xmin=None, hist_xmax=None, hist_decades=4, cutoff=None):
    """Decide the (lo, top) span of the log-binned histogram.

    ``top`` defaults to the largest observed change: on a log axis the long right
    tail costs almost no width, so there is no reason to truncate it by default.
    ``lo`` defaults to the decade holding the smallest *positive* change, floored
    at ``hist_decades`` below ``top`` so one near-zero transition can't stretch
    the axis over ten empty decades.  ``cutoff``, if given, is kept in frame --
    an off-screen threshold line is the one thing this panel can't afford.
    """
    top = float(hist_xmax) if hist_xmax is not None else float(vals.max())
    if not np.isfinite(top) or top <= 0:
        top = 1.0  # degenerate: every transition was perfectly stable
    if cutoff is not None and hist_xmax is None:
        top = max(top, float(cutoff) * 1.2)

    floor_lo = top / 10.0 ** hist_decades

    if hist_xmin is not None:
        lo = float(hist_xmin)
    else:
        positive = vals[vals > 0]
        auto_lo = 10.0 ** np.floor(np.log10(positive.min())) if positive.size else floor_lo
        lo = max(auto_lo, floor_lo)

    lo = min(lo, top / 10.0)  # always show at least one decade
    if cutoff is not None:
        lo = min(lo, float(cutoff) / 10.0)
    return lo, top


def plot_area_change(changes, layer_stats, cutoff=CHANGE_CUTOFF, path=PLOT_PATH,
                     metric=CHANGE_METRIC, n_bins=50, hist_xmin=None, hist_xmax=None,
                     hist_ymax=None, hist_decades=4, spread="iqr", show_flagged_rate=True):
    """Two panels, both log-scaled in the direction the cutoff runs.

    Left: the distribution of per-transition change, log-binned, with the cutoff
    as a vertical line and everything past it shaded -- the flagged region is
    literally the part of the histogram to the right of the line.

    Right: the per-layer distribution on a log y axis under one flat cutoff line
    (the threshold no longer moves with the layer mean), plus the % of each
    layer's transitions that land above it.
    """

    if changes.empty:
        print("\nNo z-transitions to plot")
        return None

    label = METRIC_LABELS[metric]
    vals = changes[metric].to_numpy()
    global_mean = vals.mean()
    geo_mean = 10.0 ** changes["log_change"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- left panel: distribution of per-transition area change (log x) ------
    ax = axes[0]
    lo, top = log_hist_range(vals, hist_xmin=hist_xmin, hist_xmax=hist_xmax,
                             hist_decades=hist_decades, cutoff=cutoff)
    bins = np.logspace(np.log10(lo), np.log10(top), n_bins + 1)

    n_over_axis = int((vals > top).sum())
    n_under = int((vals < lo).sum())   # includes the exactly-zero transitions
    n_zero = int((vals == 0).sum())
    n_flagged = int((vals > cutoff).sum())

    # Log bins can't hold zero or negative values, so the tails are folded into
    # the edge bins and reported in the annotation rather than silently dropped.
    ax.hist(np.clip(vals, lo, top), bins=bins,
            weights=np.full(vals.size, 1.0 / vals.size),
            color="0.5", edgecolor="white", zorder=2)
    ax.axvspan(cutoff, top, color="tab:red", alpha=0.07, linewidth=0, zorder=1)
    ax.axvline(global_mean, color="tab:blue", label=f"mean = {global_mean:.3f}")
    ax.axvline(geo_mean, color="tab:blue", linestyle=":", linewidth=1.4,
               label=f"geometric mean = {geo_mean:.3f}")
    ax.axvline(cutoff, color="tab:red", linestyle="--",
               label=f"cutoff = {cutoff:g}  (log10 = {np.log10(cutoff):.2f})")

    notes = [f"{n_flagged} / {vals.size} ({100 * n_flagged / vals.size:.1f}%) flagged"]
    if n_under:
        what = ("no area change" if n_zero == n_under
                else f"< {lo:g}" + (f" ({n_zero} of them 0)" if n_zero else ""))
        notes.append(f"{n_under} / {vals.size} "
                     f"({100 * n_under / vals.size:.1f}%) {what}\n"
                     f"clipped into first bin")
    if n_over_axis:
        notes.append(f"{n_over_axis} / {vals.size} "
                     f"({100 * n_over_axis / vals.size:.1f}%) > {top:g}\n"
                     f"clipped into last bin")
    # Sits below the legend rather than beside it: with the cutoff line labelled,
    # the legend is wide enough to meet a right-aligned note at the same height.
    ax.annotate("\n".join(notes),
                xy=(0.97, 0.60), xycoords="axes fraction",
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
    ax.set_xlabel(f"{label}  (per z transition, log scale)")
    ax.set_ylabel("fraction of object-transitions")
    ax.set_title("Distribution of between-layer area change (log scale)")
    ax.legend(loc="upper left", fontsize=9)

    # --- right panel: per-layer distribution against the flat cutoff ---------
    ax = axes[1]
    layer_stats = layer_stats.sort_values("z_from")
    z = layer_stats["z_from"].to_numpy()
    mean = layer_stats["mean_change"].to_numpy()
    geo = layer_stats["geo_mean_change"].to_numpy()
    floor = lo  # a log axis can't reach 0, so everything is clamped to the panel floor
    band_hi = geo.copy()

    if spread == "iqr":
        q = (changes.groupby("z_from")[metric]
             .quantile([0.25, 0.5, 0.75])
             .unstack()
             .reindex(z))
        q_lo, med, hi = q[0.25].to_numpy(), q[0.5].to_numpy(), q[0.75].to_numpy()
        ax.fill_between(z, np.maximum(q_lo, floor), np.maximum(hi, floor),
                        color="tab:blue", alpha=0.20, linewidth=0,
                        label="IQR (25th–75th pct)")
        ax.plot(z, np.maximum(med, floor), color="tab:blue", linestyle=":",
                linewidth=1.4, label="median")
        band_hi = hi
    elif spread in ("sd", "sem"):
        sd = layer_stats["std_change"].fillna(0).to_numpy()
        if spread == "sem":
            sd = sd / np.sqrt(np.maximum(layer_stats["n_objects"].to_numpy(), 1))
        # The lower whisker is clamped so it can't run off the bottom of a log axis.
        yerr = np.vstack([np.minimum(sd, np.maximum(mean - floor, 0)), sd])
        ax.errorbar(z, mean, yerr=yerr, fmt="none", capsize=3, color="tab:blue",
                    alpha=0.5, label=f"mean ± {spread} (clamped)")
        band_hi = mean + sd
    elif spread is not None:
        raise ValueError(f"spread must be 'iqr', 'sd', 'sem' or None, got {spread!r}")

    ax.plot(z, mean, marker="o", color="tab:blue", label="mean")
    ax.plot(z, geo, marker="o", markersize=4, linestyle="--", color="tab:cyan",
            label="geometric mean")
    ax.axhline(cutoff, linestyle="--", color="tab:red",
               label=f"cutoff = {cutoff:g} (flag threshold)")

    ax.set_yscale("log")
    ax.set_ylim(max(floor, np.nanmin(np.append(geo, mean)) / 3),
                max(np.nanmax(np.append(band_hi, mean)), cutoff) * 3)
    ax.set_xlim(z.min() - 0.5, z.max() + 0.5)
    ax.set_xlabel("z transition (z -> z+1)")
    ax.set_ylabel(f"{label}  (log scale)")
    ax.set_title("Area change per layer vs. the fixed cutoff")
    ax.set_xticks(z)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _pos: f"{v:g}"))

    handles, labels = ax.get_legend_handles_labels()

    if show_flagged_rate:
        # The threshold is the same everywhere now, so "how much of this layer is
        # over the line" is the per-layer number worth reading.
        rate_ax = ax.twinx()
        pct = 100 * layer_stats["frac_flagged"].to_numpy()
        bars = rate_ax.bar(z, pct, width=0.55, color="tab:red", alpha=0.18,
                           linewidth=0, zorder=0, label="% flagged")
        rate_ax.set_ylim(0, max(pct.max() * 1.6, 1.0))
        rate_ax.set_ylabel("% of transitions flagged", color="tab:red")
        rate_ax.tick_params(axis="y", colors="tab:red", labelsize=9)
        ax.set_zorder(rate_ax.get_zorder() + 1)
        ax.patch.set_visible(False)
        handles.append(bars)
        labels.append("% flagged (right axis)")

    ax.legend(handles, labels, loc="upper right", fontsize=9)

    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=200)
        print(f"\nSaved plot to {path}")
    else:
        plt.show()  # blocks until closed, napari launches afterwards

    return fig


def launch_viewer(dapi, masks, single_layer_vol, two_layer_vol, change_vol=None,
                  cutoff=CHANGE_CUTOFF, metric=CHANGE_METRIC):
    import napari  # pulls in Qt; kept out of module scope so headless runs work

    viewer = napari.Viewer()
    viewer.add_image(
        dapi, name="DAPI (deconvolved)", colormap="gray",
        contrast_limits=[float(dapi.min()), float(np.percentile(dapi, 99.5))],
    )

    # Give each mask a random color
    viewer.add_labels(masks, name="segmentation masks", opacity=0.4)
    viewer.add_labels(single_layer_vol, name=F"{LAYER_SPAN_CUTOFF}-slice nuclei", opacity=0.6, visible=False)
    viewer.add_labels(two_layer_vol, name=F"{LAYER_SPAN_CUTOFF + 1}-slice nuclei", opacity=0.6, visible=False)
    if change_vol is not None:
        viewer.add_labels(change_vol,
                          name=F"{METRIC_LABELS[metric]} > {cutoff:g}",
                          opacity=0.6, visible=False, blending='additive')

    print("\nLaunching napari viewer")
    napari.run()
    return viewer


def main():
    dapi, masks, files = load_data(DAPI_GLOB, MASKS_PATH)
    df = check_z_span(masks, layer_span_cutoff=LAYER_SPAN_CUTOFF)

    changes = check_area_change(masks, metric=CHANGE_METRIC)
    layer_stats = summarise_layer_change(changes, cutoff=CHANGE_CUTOFF, metric=CHANGE_METRIC)
    per_object = flag_area_outliers(changes,
                                    cutoff=CHANGE_CUTOFF,
                                    statistic=OBJECT_STATISTIC,
                                    metric=CHANGE_METRIC)

    df = df.merge(per_object, on="label", how="left")
    df["large_area_change"] = df["large_area_change"].fillna(False).astype(bool)

    single_layer_vol, two_layer_vol = build_span_layers(masks, df)
    change_vol = build_change_layer(masks, df)

    give_output_summary(df)
    give_change_summary(df, changes, layer_stats)

    if MAKE_PLOT:
        plot_area_change(changes, layer_stats, cutoff=CHANGE_CUTOFF, path=PLOT_PATH)

    if LAUNCH_NAPARI:
        launch_viewer(dapi, masks, single_layer_vol, two_layer_vol, change_vol)


# Might want to change DAPI colormap to magma, reduce opacity to 0.7, and gamma to 1.5

if __name__ == "__main__":
    main()
