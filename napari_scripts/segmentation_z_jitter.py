"""Flag masks whose area-through-z profile *wanders*, rather than ones that grow.

Drawn as area against z, a healthy nucleus is a smooth line: it grows to a
mid-plane and shrinks away from it.  A mask that goes up, down, up, down is not
tracing an object at all — there is no nucleus whose cross-section does that —
and it is usually a per-plane segmenter changing its mind about where the
boundary is.

The metric is how far the profile travels *back on itself*::

    d[k]     = log10(a[k+1]) - log10(a[k])       per-step log growth
    reversal = sign(d[k]) != sign(d[k+1])        the profile turned around
    w[k]     = min(|d[k]|, |d[k+1]|)             how deep that V goes
    jitter   = 10 ** sum(w over reversals)       a factor, >= 1.0

Two choices carry it.  Weighting a reversal by the **smaller** of its two limbs
makes a ramp free: a huge rise followed by a tiny dip is worth the tiny dip, so
the steep onset every nucleus has cannot dominate the score.  And **summing**
rather than taking the max lets many small wobbles outscore one big excursion —
under a max, ten alternating 1.5x wobbles score 1.5x while a single monotone
jump scores 50x, which is exactly backwards.

Being a ratio of ratios it is dimensionless, so a small nucleus and a large one
are held to the same standard and a fixed cutoff means the same thing in every
FOV.

Reference points: a monotone ramp of any steepness scores **1.00**, a flat run
with one 10x jump and another flat run **1.00** (it never reverses — that shape
is what ``sway`` in ``segmentation_z_area_sway.py`` is for), a healthy
grow-peak-shrink arc **1.16**, +-10% noise on a plateau **1.48**, and a profile
that goes up-down-up-down **53**.

A healthy nucleus is unimodal, so it contains exactly one reversal at its peak
and lands a little above 1.0 — typically 1.1-1.3.  That peak is *not* exempt,
which is why ``JITTER_CUTOFF`` has to sit above the ordinary bulk rather than at
1.0.  The shipped 1.75 is provisional, from worked profiles rather than from
data: the distribution this script writes is how you set it, by looking for the
knee where the ordinary bulk ends and the tail begins.

A mask needs three consecutive surviving planes for a reversal to exist; one
that never gets them has ``jitter = NaN`` and is never flagged.

The measurement itself lives in ``segmentation_helpers_v2.py`` and is called from
here, so this script and the notebook's scorecard cannot drift apart.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# the helpers sit one directory up.  Running this as
# `python napari_scripts/segmentation_z_jitter.py` puts only napari_scripts/ on sys.path, so the
# repo root goes on it too -- `pip install -e .` makes the same imports work
# without this, and the line is harmless when it has.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import segmentation_helpers_v2 as h  # noqa: E402

# CONFIG
# ----------------------------------------------------------------------------
MASKS_DIR = Path("../data/segmentations_3d_true/cpdino/decon/VePo/region_UWA-7648/fov_07")
DAPI_DIR = Path("../data/patches/VePo/region_UWA-7648/fov_07")

MASKS_PATH = MASKS_DIR / "masks.tif"
DAPI_GLOB = str(DAPI_DIR / "DAPI_decon_z*.tif")  # Each z index

RUN_LABEL = None           # legend text; None -> "<region> · <fov>" read off MASKS_DIR

JITTER_CUTOFF = 1.75       # a factor pins the cutoff and overrides CUTOFF_QUANTILE.
                           # Jitter is dimensionless, so an absolute number means the
                           # same thing in every FOV -- where a quantile flags the top
                           # 10% of every FOV by construction, which is a shortlist to
                           # look at and a useless number to compare runs on
CUTOFF_QUANTILE = 0.90     # None cutoff -> flag the top 10% of this FOV's jitters
AREA_FLOOR = 20            # px; ratios between handfuls of pixels are noise
SLIVER_FRAC = 0.05         # a plane holding less than this share of the mask's own peak
                           # is a partial-volume sliver, not a cross-section, and is
                           # left out of the profile entirely.  Every nucleus enters
                           # and leaves the stack through one
DROP_SINGLE_SLICE = True   # a mask on exactly one z plane is an imaging artifact.  It
                           # could never be measured anyway, so this changes the
                           # denominator of the flagged rate, not the numerator

N_WORST = 8                # how many offenders the summary lists

MAKE_PLOT = True
PLOT_PATH = Path("z_jitter.png")   # None -> show interactively

LAUNCH_NAPARI = True
Z_SCALE = None             # e.g. 15 -> stretch z so the 3D view is roughly isotropic;
                           # the right number is the z step divided by the pixel size
# ----------------------------------------------------------------------------


def load_data(dapi_glob, masks_path):
    """The DAPI stack and the mask volume that was segmented from it."""
    dapi, files = h.load_dapi(dapi_glob)
    masks = h.load_masks(masks_path)
    if masks.shape != dapi.shape:
        raise ValueError(f"masks shape {masks.shape} != dapi shape {dapi.shape}")

    print(f"Loaded {len(files)} DAPI planes from {Path(files[0]).parent}")
    print(f"Volume shape: {dapi.shape}  |  {int(masks.max())} labeled objects")
    return dapi, masks, files


def run_label(masks_dir=MASKS_DIR):
    """``<region> · <fov>`` off the mask path, for the legend and the title."""
    parts = Path(masks_dir).parts
    return " · ".join(parts[-2:]) if len(parts) >= 2 else str(masks_dir)


def analyse_masks(masks, cutoff=JITTER_CUTOFF, quantile=CUTOFF_QUANTILE,
                  area_floor=AREA_FLOOR, sliver_frac=SLIVER_FRAC,
                  drop_single_slice=DROP_SINGLE_SLICE):
    """``(df, kept, dropped, cutoff, source, report)`` — one row per surviving mask.

    The single-slice filter runs first, so the artifacts are gone before the
    profile is measured and before the flagged rate gets its denominator.
    """
    areas = h.measure_slice_areas(masks)
    kept, dropped, areas, report = h.filter_single_slice(
        masks, areas, drop=drop_single_slice)

    df = h.check_area_jitter(areas, area_floor=area_floor, sliver_frac=sliver_frac)
    df, cutoff, source = h.flag_jitter(df, cutoff=cutoff, quantile=quantile)
    return df, kept, dropped, cutoff, source, report


def build_jitter_layer(masks, df):
    """A copy of the volume with everything but the flagged labels zeroed."""
    flagged = df.loc[df["large_jitter"], "label"].to_numpy()
    return np.where(np.isin(masks, flagged), masks, 0).astype(masks.dtype)


def give_jitter_summary(df, cutoff, source, report=None, n_worst=N_WORST):
    """The distribution, the cutoff, and the masks that fail it."""
    print("\n=== AREA-JITTER SUMMARY ===")
    if report is not None and report["n_single_slice"]:
        print(f"  single-slice masks: {report['n_single_slice']} of {report['n_total']} "
              f"({report['pct_dropped']:.1f}%)"
              f"{' -- dropped before measuring' if report['n_dropped'] else ' -- kept'}")

    values = df["jitter"].dropna()
    if values.empty:
        print("  no mask has three consecutive planes; nothing to measure")
        return

    print(f"  n_objects: {len(df)}  |  with a measurable jitter: {len(values)}")
    print("  jitter quantiles: "
          + "  ".join(f"p{100 * q:g} {values.quantile(q):,.2f}x"
                      for q in (0.5, 0.75, 0.9, 0.95, 0.99))
          + f"  max {values.max():,.1f}x")

    n_flagged = int(df["large_jitter"].sum())
    print(f"  cutoff: jitter > {cutoff:,.2f}x  [{source}]")
    print(f"  n_large_jitter: {n_flagged} / {len(values)} "
          f"({100 * n_flagged / len(values):.1f}%)")

    if not n_flagged:
        return

    # n_reversals says how many times the profile turned around; z_at_jitter is the
    # middle plane of the deepest one -- the plane to scroll to in the viewer
    print("\n  worst offenders:")
    print("    label   planes   used   rev   area_min -> area_max     jitter   z@jitter")
    for row in df.nlargest(min(n_worst, n_flagged), "jitter").itertuples():
        print(f"    {int(row.label):>6}  {int(row.n_planes):>6} {int(row.n_planes_used):>6} "
              f"{int(row.n_reversals):>5}   "
              f"{int(row.area_min):>8,} -> {int(row.area_max):<8,}  "
              f"{row.jitter:>9.2f}x  {int(row.z_at_jitter):>8}")


def plot_jitter(df, cutoff, path=PLOT_PATH, label=None):
    """The notebook's distribution figure: log-binned per-mask jitter, cutoff shaded.

    This is how the cutoff gets set — look for the knee where the ordinary bulk
    ends and the tail begins, and put ``JITTER_CUTOFF`` there.
    """
    label = label or run_label()
    fig = h.compare_profile_distribution(h.tag(df, label), "jitter", cutoff=cutoff)

    if path is None:
        plt.show()
    else:
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"\nWrote {path}")
        plt.close(fig)
    return fig


def launch_viewer(dapi, masks, df, dropped=None, cutoff=JITTER_CUTOFF, z_scale=Z_SCALE):
    """DAPI, every mask, and the flagged ones as a layer to toggle over them."""
    layers = {f"jitter > {cutoff:,.2f}x": build_jitter_layer(masks, df)}
    if dropped is not None and int(dropped.max()) > 0:
        layers["dropped: single-slice artifacts"] = dropped

    return h.launch_viewer(dapi, masks, layers, title=run_label(), z_scale=z_scale)


def main():
    # every CONFIG value is passed explicitly rather than left to the signature
    # defaults, which bind at import: this way setting e.g. ``JITTER_CUTOFF`` on
    # the imported module takes effect, instead of being silently ignored
    h.apply_theme()
    dapi, masks, files = load_data(DAPI_GLOB, MASKS_PATH)

    df, kept, dropped, cutoff, source, report = analyse_masks(
        masks, cutoff=JITTER_CUTOFF, quantile=CUTOFF_QUANTILE,
        area_floor=AREA_FLOOR, sliver_frac=SLIVER_FRAC,
        drop_single_slice=DROP_SINGLE_SLICE)
    give_jitter_summary(df, cutoff, source, report, n_worst=N_WORST)

    if MAKE_PLOT:
        plot_jitter(df, cutoff, path=PLOT_PATH, label=RUN_LABEL)

    if LAUNCH_NAPARI:
        launch_viewer(dapi, kept, df, dropped, cutoff, z_scale=Z_SCALE)


if __name__ == "__main__":
    main()
