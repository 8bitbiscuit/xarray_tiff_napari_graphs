"""Flag masks whose area profile through z *sways*, rather than ones that grow.

3D stitching can glue a nucleus on one plane to something much larger on the
next.  Drawn as area against z, that is a line with a corner in it: a flat run, a
jump, another flat run.  A healthy nucleus is a smooth line — an arc, or a steady
rise — whatever its steepness.

``area_max - area_min`` cannot tell those apart, because a *range* is large for
any mask that ends up much bigger than it started, and the steepest smooth
growers win it outright.  So the metric here is the sharpness of the corner:

    sway = 10 ** max | log10(a[z+1]) - 2 log10(a[z]) + log10(a[z-1]) |

the largest second difference of log area over three consecutive planes, read
back as a factor.  It is the change in *growth rate* from one step to the next,
so a mask growing at a constant rate scores 1.0 no matter how fast it grows, a
gentle arc scores a little over 1, and a mask that steps by 10x in one plane and
is flat either side scores about 10.  Being a ratio of ratios it is
dimensionless: a small nucleus and a large one are held to the same standard.

Areas are floored at ``AREA_FLOOR`` before the log, because a ratio between
handfuls of pixels is noise — 4 px against 1 px is a factor of 4 that means
nothing.  A mask needs three consecutive planes for a second difference to exist;
one that never gets them has ``sway = NaN`` and is never flagged.

The cutoff is the **top 10%** of the FOV's sways (``CUTOFF_QUANTILE = 0.90``), so
the flag is a fixed-size shortlist of the least smooth masks rather than a claim
about any of them.  Worth knowing when reading it: most masks are smooth, so the
sway distribution is a spike at 1.0x with a sparse tail, and a rank cutoff this
low reaches down into masks whose profiles look straight — on a 1,552-mask FOV,
p90 lands near 1.4x while the stitching steps are up at 4-14x.  Because a sway is
dimensionless, ``SWAY_CUTOFF = 3.0`` pins a threshold that means the same thing in
every FOV instead; the quantiles printed by the summary say where that would sit
in this one.

Reference points: a constant growth rate of any steepness scores 1.0x, a steep
rise flattening into a plateau 1.5x, a sharply peaked healthy arc 2.0x, a mask
that steps 14x in one plane 14x, and one that drops out for a plane and comes
back 100x.
"""

import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import tifffile

# CONFIG
# ----------------------------------------------------------------------------
MASKS_DIR = Path("../data/segmentations_3d_true/cpdino/decon/VePo/region_UWA-7648/fov_09")
DAPI_DIR = Path("../data/patches/VePo/region_UWA-7648/fov_09")

MASKS_PATH = MASKS_DIR / "masks.tif"
DAPI_GLOB = str(DAPI_DIR / "DAPI_decon_z*.tif")  # Each z index

SWAY_CUTOFF = None         # a factor pins the cutoff and overrides CUTOFF_QUANTILE.
                           # A sway is dimensionless, so an absolute number means the
                           # same thing in every FOV: 3.0 sits above the sharpest
                           # healthy arc and well below a stitching step
CUTOFF_QUANTILE = 0.90     # None cutoff -> flag the top 10% of this FOV's sways
AREA_FLOOR = 20            # px; area differences below this are noise, not shape
SLIVER_FRAC = 0.05         # a plane holding less than this share of the mask's own peak
                           # area is a partial-volume sliver, not a cross-section, and
                           # is left out of the profile entirely.  The absolute floor
                           # cannot do this job on its own: a mask that onsets at 60 px
                           # and jumps to 2,000 clears AREA_FLOOR and still contributes
                           # a 30x step that is an artifact of where the nucleus met
                           # the edge of the stack, not of how it was segmented
JITTER_CUTOFF = 1.75       # provisional -- see the note in check_area_jitter

MAKE_PLOT = True
PLOT_PATH = Path("z_area_sway.png")   # None -> show interactively

LAUNCH_NAPARI = True
# ----------------------------------------------------------------------------


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

    print(f"Volume shape: {dapi.shape}  |  {int(masks.max())} labeled objects")
    return dapi, masks, files


def measure_slice_areas(mask_volume):
    """Pixel count of every label in every z slice -> array of shape (n_z, n_labels + 1)."""
    n_labels = int(mask_volume.max())
    areas = np.zeros((mask_volume.shape[0], n_labels + 1), dtype=np.int64)

    for z in range(mask_volume.shape[0]):
        counts = np.bincount(mask_volume[z].ravel(), minlength=n_labels + 1)
        areas[z] = counts[:n_labels + 1]

    return areas


SWAY_COLUMNS = ["label", "n_planes", "n_planes_used", "area_min", "area_max",
                "sway", "z_at_sway"]


def kept_planes(areas, area_floor=AREA_FLOOR, sliver_frac=SLIVER_FRAC):
    """Which planes are a real cross-section of each mask, as a boolean array.

    A plane is dropped when it holds fewer than ``area_floor`` px, or less than
    ``sliver_frac`` of that mask's own peak area.  Both tests are needed and the
    relative one does the work: every nucleus enters and leaves the stack through
    a partial-volume sliver, and a sliver against a mid-plane is a ratio of
    hundreds to one that says where the nucleus met the edge of the volume rather
    than anything about the segmentation.

    This replaces flooring the areas before the log.  Flooring *clamped* a 16 px
    onset plane to 20 px and then measured it anyway, which left a ~100x step in
    every mask's profile by construction -- the artifact that made the old sway
    metric a ranking of onset steepness.  Excluding the plane is what the floor
    was always meant to do.
    """
    present = areas > 0
    peak = areas.max(axis=0)
    return present & (areas >= area_floor) & (areas >= sliver_frac * peak)


def check_area_sway(areas, area_floor=AREA_FLOOR, sliver_frac=SLIVER_FRAC):
    """One row per label: the sharpest corner in its area-through-z profile.

    Only triples of *consecutive present* planes count, so a mask stitched across
    a z gap does not score a sway for the hole, and appearing or disappearing is
    not a corner.  ``z_at_sway`` is the middle plane of the worst triple — the
    plane to scroll to in napari.
    """
    present = areas > 0
    labels = np.flatnonzero(present.any(axis=0))
    labels = labels[labels > 0]  # drop background
    if labels.size == 0:
        return pd.DataFrame(columns=SWAY_COLUMNS)

    keep = kept_planes(areas, area_floor, sliver_frac)
    log_area = np.log10(np.maximum(areas, 1))

    if areas.shape[0] >= 3:
        triple = keep[:-2] & keep[1:-1] & keep[2:]
        # second difference: how much the growth rate changed at the middle plane
        curvature = np.abs(log_area[2:] - 2.0 * log_area[1:-1] + log_area[:-2])
        curvature = np.where(triple, curvature, -1.0)
        measurable = triple.any(axis=0)
        sway = np.where(measurable, 10.0 ** curvature.max(axis=0), np.nan)
        z_at_sway = np.where(measurable, curvature.argmax(axis=0) + 1, -1)
    else:
        sway = np.full(areas.shape[1], np.nan)
        z_at_sway = np.full(areas.shape[1], -1)

    absent_high = np.where(present, areas, np.iinfo(np.int64).max)

    return pd.DataFrame({
        "label": labels.astype(np.int64),
        "n_planes": present.sum(axis=0)[labels],
        "n_planes_used": keep.sum(axis=0)[labels],
        "area_min": absent_high.min(axis=0)[labels],
        "area_max": areas.max(axis=0)[labels],
        "sway": sway[labels],
        "z_at_sway": z_at_sway[labels],
    })


JITTER_COLUMNS = ["label", "n_planes", "n_planes_used", "area_min", "area_max",
                  "n_reversals", "jitter", "z_at_jitter"]


def check_area_jitter(areas, area_floor=AREA_FLOOR, sliver_frac=SLIVER_FRAC):
    """One row per label: how much its area profile wanders up and down.

    Sway asks *how sharp is the worst corner*, which is a question about a single
    transition and is answered identically by a corner and by a steep monotone
    ramp.  This asks a different one -- **how often, and how far, does the profile
    reverse direction** -- which is the question a jittery mask fails and a ramp,
    however violent, does not::

        d[k]     = log10 a[k+1] - log10 a[k]        per-step log growth
        reversal = sign(d[k]) != sign(d[k+1])       direction turned around
        w[k]     = min(|d[k]|, |d[k+1]|)            how deep the V goes
        jitter   = 10 ** sum(w over reversals)      a factor, >= 1.0

    Two choices carry the metric.  Weighting a reversal by the **smaller** of its
    two limbs is what makes a ramp cost nothing: a huge rise followed by a tiny
    dip is worth the tiny dip, not the huge rise, so the onset ramp that
    dominated sway cannot dominate this even when a sliver survives the filter.
    And **summing** rather than taking the max is what lets many small wobbles
    outscore one big excursion -- under a max, ten alternating 1.5x wobbles score
    1.5x and a single monotone jump scores 50x, which is exactly backwards.

    A monotone profile scores exactly 1.0 at any steepness.  A healthy nucleus is
    unimodal -- it grows to its mid-plane and shrinks -- so it contains one
    reversal at the peak and lands a little above 1.0, typically 1.1-1.3; that
    peak is **not** exempt, so the cutoff has to sit above the ordinary bulk
    rather than at 1.0.  ``JITTER_CUTOFF = 1.75`` is a provisional value from
    worked profiles, not from this data: derive it from the pooled distribution
    before trusting it, the way the notebook's config note says.

    ``z_at_jitter`` is the middle plane of the deepest reversal -- the plane to
    scroll to in napari -- and is -1 for a mask that never reverses.
    """
    present = areas > 0
    labels = np.flatnonzero(present.any(axis=0))
    labels = labels[labels > 0]  # drop background
    if labels.size == 0:
        return pd.DataFrame(columns=JITTER_COLUMNS)

    keep = kept_planes(areas, area_floor, sliver_frac)
    log_area = np.log10(np.maximum(areas, 1))

    if areas.shape[0] >= 3:
        step_ok = keep[:-1] & keep[1:]              # both ends of this step survive
        d = np.where(step_ok, log_area[1:] - log_area[:-1], 0.0)

        triple = step_ok[:-1] & step_ok[1:]         # three consecutive kept planes
        # a flat step has sign 0, so its product is 0 and it is not a reversal:
        # the profile has to actually turn around, not merely stop rising
        reversed_here = triple & (np.sign(d[:-1]) * np.sign(d[1:]) < 0)
        depth = np.where(reversed_here,
                         np.minimum(np.abs(d[:-1]), np.abs(d[1:])), 0.0)

        measurable = triple.any(axis=0)
        n_reversals = reversed_here.sum(axis=0)
        jitter = np.where(measurable, 10.0 ** depth.sum(axis=0), np.nan)
        z_at_jitter = np.where(n_reversals > 0, depth.argmax(axis=0) + 1, -1)
    else:
        n_reversals = np.zeros(areas.shape[1], dtype=np.int64)
        jitter = np.full(areas.shape[1], np.nan)
        z_at_jitter = np.full(areas.shape[1], -1)

    absent_high = np.where(present, areas, np.iinfo(np.int64).max)

    return pd.DataFrame({
        "label": labels.astype(np.int64),
        "n_planes": present.sum(axis=0)[labels],
        "n_planes_used": keep.sum(axis=0)[labels],
        "area_min": absent_high.min(axis=0)[labels],
        "area_max": areas.max(axis=0)[labels],
        "n_reversals": n_reversals[labels].astype(np.int64),
        "jitter": jitter[labels],
        "z_at_jitter": z_at_jitter[labels],
    })


def flag_metric(df, column, cutoff, quantile=CUTOFF_QUANTILE, flag=None):
    """Add a boolean flag for ``column`` exceeding ``cutoff``.

    The body of the old :func:`flag_sway`, with the two column names lifted out
    so jitter and sway can share it.  ``cutoff=None`` takes the quantile of the
    values that exist -- a rank statistic, so nothing is assumed about the
    distribution's shape and masks too short to be measured cannot drag it down.
    """
    flag = flag or f"large_{column}"
    df = df.copy()
    if df.empty:
        df[flag] = pd.Series(dtype=bool)
        return df, float("inf"), "no masks"

    if cutoff is None:
        values = df[column].dropna()
        cutoff = float(values.quantile(quantile)) if len(values) else float("inf")
        source = f"p{100 * quantile:g} of this FOV's {column} values"
    else:
        cutoff, source = float(cutoff), "set in CONFIG"

    df[flag] = df[column] > cutoff        # NaN compares False
    return df, cutoff, source


def flag_jitter(jitters, cutoff=JITTER_CUTOFF, quantile=CUTOFF_QUANTILE):
    """:func:`flag_metric` on ``jitter``.  Returns ``(df, cutoff, source)``."""
    return flag_metric(jitters, "jitter", cutoff, quantile, flag="large_jitter")


def flag_sway(sways, cutoff=SWAY_CUTOFF, quantile=CUTOFF_QUANTILE):
    """Add ``large_sway``.  Returns ``(sways, cutoff, source)``.

    ``cutoff=None`` takes the quantile of the sways that exist — a rank
    statistic, so nothing is assumed about the distribution's shape, and masks
    too short to have a sway can't drag it down.
    """
    return flag_metric(sways, "sway", cutoff, quantile, flag="large_sway")


def build_sway_layer(masks, df, column="sway"):
    flagged = df.loc[df[f"large_{column}"], "label"].to_numpy()
    return np.where(np.isin(masks, flagged), masks, 0).astype(masks.dtype)


def give_sway_summary(df, cutoff, source, n_worst=8, column="sway"):
    flag, z_col = f"large_{column}", f"z_at_{column}"
    print(f"\n=== AREA-{column.upper()} SUMMARY ===")
    values = df[column].dropna()
    if values.empty:
        print("  no mask has three consecutive planes; nothing to measure")
        return

    print(f"  n_objects: {len(df)}  |  with a measurable {column}: {len(values)}")
    print(f"  {column} quantiles: "
          + "  ".join(f"p{100 * q:g} {values.quantile(q):,.1f}x"
                      for q in (0.5, 0.75, 0.9, 0.95, 0.99))
          + f"  max {values.max():,.1f}x")

    n_flagged = int(df[flag].sum())
    print(f"  cutoff: {column} > {cutoff:,.1f}x  [{source}]")
    print(f"  n_{flag}: {n_flagged} / {len(values)} "
          f"({100 * n_flagged / len(values):.1f}%)")

    if not n_flagged:
        return

    print("\n  worst offenders:")
    print(f"    label   planes   used   area_min -> area_max   {column:>8}   z@{column}")
    for row in df.nlargest(min(n_worst, n_flagged), column).itertuples():
        print(f"    {int(row.label):>6}  {int(row.n_planes):>6} {int(row.n_planes_used):>6}   "
              f"{int(row.area_min):>8,} -> {int(row.area_max):<8,}  "
              f"{getattr(row, column):>8.1f}x  {int(getattr(row, z_col)):>7}")


AXIS_LABEL = {
    "jitter": "jitter = total up-and-down travel, as a factor (log scale)",
    "sway": "sway = worst change in growth rate, as a factor (log scale)",
}


def plot_area_sway(df, areas, cutoff, source, path=PLOT_PATH, column="sway",
                   n_bins=45, n_profiles=10, n_background=80, seed=0):
    """Left: the distribution, log-binned, with the flagged region shaded — 1.0x
    is a profile that never turns around, whatever its slope.  Right: area
    through z for the worst offenders over a grey sample of the rest, with the
    flagged plane ringed, so the thing being measured is the thing you look at.

    **This right-hand panel is how a profile metric gets judged.**  Drawn for
    ``sway`` before the sliver fix, every worst offender was a near-vertical
    onset ramp with the ringed plane at the top of it -- which is what the metric
    was really ranking.  Draw it for whatever you change and look at the red
    lines: they have to be the shape you meant to catch.
    """
    flag, z_col = f"large_{column}", f"z_at_{column}"
    rows = df[df[column].notna()]
    if rows.empty:
        print("\nNo mask has three consecutive planes to plot")
        return None

    values = rows[column].to_numpy(dtype=float)
    flagged = rows[flag].to_numpy(dtype=bool)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # --- left: the distribution ---------------------------------------------
    ax = axes[0]
    top = max(values.max(), cutoff * 1.2 if np.isfinite(cutoff) else 0.0, 1.1)
    bins = np.logspace(0, np.log10(top), n_bins + 1)  # both are >= 1 by construction

    ax.hist(np.clip(values, 1.0, top), bins=bins,
            weights=np.full(values.size, 1.0 / values.size),
            color="0.5", edgecolor="white", zorder=2)
    if np.isfinite(cutoff):
        ax.axvspan(cutoff, top, color="tab:red", alpha=0.07, linewidth=0, zorder=1)
        ax.axvline(cutoff, color="tab:red", linestyle="--",
                   label=f"cutoff = {cutoff:,.1f}x  ({source})")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)

    ax.set_xscale("log")
    ax.set_xlim(1.0, top)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _pos: f"{v:g}x"))
    ax.xaxis.grid(True, which="major", color="0.9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(AXIS_LABEL.get(column, f"{column} (log scale)"))
    ax.set_ylabel("fraction of masks")
    ax.set_title(f"Per-mask {column}  ({int(flagged.sum())} flagged)")
    ax.legend(loc="upper right", fontsize=9)

    # --- right: area through z ----------------------------------------------
    ax = axes[1]
    rng = np.random.default_rng(seed)
    z_axis = np.arange(areas.shape[0])

    def profile(label):
        values_z = areas[:, int(label)].astype(float)
        values_z[values_z == 0] = np.nan
        return values_z

    quiet = rows.loc[~flagged, "label"].to_numpy()
    sample = rng.choice(quiet, size=min(n_background, quiet.size), replace=False)
    for label in sample:
        ax.plot(z_axis, profile(label), color="0.6", alpha=0.35, linewidth=0.9, zorder=1)

    worst = rows.loc[flagged].nlargest(min(n_profiles, int(flagged.sum())), column)
    for row in worst.itertuples():
        line = profile(row.label)
        z_flag = int(getattr(row, z_col))
        ax.plot(z_axis, line, color="tab:red", alpha=0.85, linewidth=1.4,
                marker="o", markersize=3, zorder=3)
        if z_flag >= 0:
            ax.plot(z_flag, line[z_flag], marker="o", markersize=9,
                    markerfacecolor="none", markeredgecolor="tab:red", zorder=4)

    ax.plot([], [], color="0.6", linewidth=0.9, label=f"unflagged sample (n={sample.size})")
    ax.plot([], [], color="tab:red", linewidth=1.4, marker="o", markersize=3,
            label=f"worst {len(worst)} flagged ({column} plane ringed)")

    ax.set_yscale("log")
    ax.set_xlim(-0.5, areas.shape[0] - 0.5)
    ax.set_xticks(z_axis)
    ax.set_xlabel("z plane")
    ax.set_ylabel("area (pixels, log scale)")
    ax.set_title("Area through z")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()

    if path is not None:
        fig.savefig(path, dpi=200)
        print(f"\nSaved plot to {path}")
    else:
        plt.show()  # blocks until closed, napari launches afterwards

    return fig


def launch_viewer(dapi, masks, sway_vol, cutoff):
    import napari  # pulls in Qt; kept out of module scope so headless runs work

    viewer = napari.Viewer()
    viewer.add_image(
        dapi, name="DAPI (deconvolved)", colormap="gray",
        contrast_limits=[float(dapi.min()), float(np.percentile(dapi, 99.5))],
    )

    # Give each mask a random color
    viewer.add_labels(masks, name="segmentation masks", opacity=0.4)
    viewer.add_labels(sway_vol, name=f"sway > {cutoff:,.1f}x",
                      opacity=0.6, blending="additive")

    print("\nLaunching napari viewer")
    napari.run()
    return viewer


def analyse_masks(masks, cutoff=SWAY_CUTOFF, quantile=CUTOFF_QUANTILE,
                  area_floor=AREA_FLOOR, sliver_frac=SLIVER_FRAC,
                  jitter_cutoff=JITTER_CUTOFF, column="sway"):
    """Returns ``(df, areas, cutoff, source)`` — one row per label, plus the
    per-plane areas the profile panel draws from.

    ``column="jitter"`` measures how much the profile wanders up and down;
    ``"sway"`` measures its sharpest single corner.  They catch different
    defects — a flat/jump/flat over-merge is monotone, so it scores a clean 1.0
    jitter — so a full look at a FOV runs both.
    """
    areas = measure_slice_areas(masks)
    if column == "jitter":
        df, cutoff, source = flag_jitter(
            check_area_jitter(areas, area_floor, sliver_frac), jitter_cutoff, quantile)
    else:
        df, cutoff, source = flag_sway(
            check_area_sway(areas, area_floor, sliver_frac), cutoff, quantile)
    return df, areas, cutoff, source


def main():
    dapi, masks, files = load_data(DAPI_GLOB, MASKS_PATH)

    for column in ("jitter", "sway"):
        df, areas, cutoff, source = analyse_masks(masks, column=column)
        give_sway_summary(df, cutoff, source, column=column)

        if MAKE_PLOT:
            path = None if PLOT_PATH is None else PLOT_PATH.with_name(
                f"{PLOT_PATH.stem}_{column}{PLOT_PATH.suffix}")
            plot_area_sway(df, areas, cutoff, source, path=path, column=column)

        if LAUNCH_NAPARI and column == "jitter":
            launch_viewer(dapi, masks, build_sway_layer(masks, df, column), cutoff)


# Might want to change DAPI colormap to magma, reduce opacity to 0.7, and gamma to 1.5

if __name__ == "__main__":
    main()
