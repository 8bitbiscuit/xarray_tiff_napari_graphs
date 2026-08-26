"""The two-panel brightness figure with an absolute cutoff drawn across it.

``segmentation_brightness_helpers.compare_brightness_distributions`` is the
figure the comparison notebook's section 4 draws: two panels -- voxels a
technique put in a mask, voxels it did not -- with both techniques overlaid
inside each, on bins built once from the DAPI stack they both segmented. This
script draws *that* figure, unchanged, and adds one thing to it: a vertical rule
at ``BRIGHTNESS_CUTOFF``, unlabelled, with the title saying what it is set to.
What sits either side of that rule comes back as two
tables rather than as annotations on the curves -- where the distributions fall
relative to the line is what the figure is for, and the shares either side read
better as numbers than as text laid over the thing they describe.

The figure is not re-implemented here. It is produced by the same call the
notebook makes, and :func:`draw_cutoff` annotates the axes afterwards, so the
curves, bins, colours, medians and clipping note stay exactly what section 4
shows and the rule is the only addition.

Why a fixed cutoff at all
-------------------------
Every threshold the brightness code already has is a *quantile* of a
distribution -- the overlap layers use the median of the other group, the x
axis stops at the 0.999 quantile of the pooled data. A quantile moves with the
data, so "above the threshold" means something different in every FOV and in
every technique. An absolute cutoff does not: 7,000 counts is 7,000 counts in
both arms and in every field, which is what makes the two sides of it
comparable *across* techniques and not only within one.

What the tables answer
----------------------
Everything in them is about the rule. Nothing counts voxels, and no column
holds a statistic the cutoff plays no part in.

``auc_below`` / ``auc_above`` (within a technique)
    The area under that curve on each side of the rule, taken literally. The
    curve is a histogram normalised within (technique, group), so its whole area
    is 1 and the two sides sum to it.

``pct_below`` / ``pct_above``
    Those same two areas as percentages. The same numbers times 100 -- a share
    reads faster than a fraction, and the fraction is what "area under the
    curve" names, so both are carried.

``auc_below`` / ``auc_above`` (across techniques)
    Not an area: in the across table these are the rank statistic, P(a voxel
    from technique ``a``'s group is brighter than one from technique ``b``'s
    same group), on one side of the rule at a time. 0.5 says the two techniques'
    distributions on that side are interchangeable; away from 0.5 says one is
    systematically brighter, which is the comparison the panels are drawn for.
    The name is shared with the within table's columns and the quantity is not,
    so read each table by its own heading.

``d_pct_above``
    ``a`` minus ``b``: the difference in area on the bright side of the rule.
    The dim side's difference is its exact negative, so only one is carried.

The within table splits by (technique, group), since a share is a property of
one curve; the across table pairs the techniques off (``a`` minus ``b``, in
``TECHNIQUE_ROOTS`` order) inside each group.

napari
------
Two boolean volumes per technique, both defined by the same rule the figure
draws: masked voxels *below* the cutoff (background the technique pulled in) and
unmasked voxels *at or above* it (signal it left behind). Those are the two
disagreement corners of the cutoff; the agreement corners are the mask volume
and everything else, and neither needs a layer of its own.

napari is imported inside :func:`launch_viewers` only, so the analysis and the
figure run on a machine without Qt.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# the helpers sit one directory up.  Running this as
# `python napari_scripts/segmentation_z_brightness_bimodal.py` puts only napari_scripts/ on sys.path, so the
# repo root goes on it too -- `pip install -e .` makes the same imports work
# without this, and the line is harmless when it has.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import segmentation_brightness_helpers as bh  # noqa: E402
import segmentation_helpers_v2 as v2  # noqa: E402

# CONFIG
# ----------------------------------------------------------------------------
DATA_ROOT = Path("../data")

FAMILY = "cpdino"
PREPROCESSING = "decon"
MODEL = "VePo"

TECHNIQUE_ROOTS = {
    "3D true":     DATA_ROOT / "segmentations_3d_true"     / FAMILY / PREPROCESSING / MODEL,
    "3D stitched": DATA_ROOT / "segmentations_3d_stitched" / FAMILY / PREPROCESSING / MODEL,
}
DAPI_PATCHES = DATA_ROOT / "patches" / MODEL

REGION = "region_UWA-7648"
FOV = "fov_07"

# The rule this whole script is about. Absolute, in the units of the image, so
# it means the same thing in both techniques and in every field.
BRIGHTNESS_CUTOFF = 7000

# The notebook's filter, kept because the figure is the notebook's figure: a
# single-slice mask is an artifact, and dropping it moves its voxels into the
# unmasked group rather than out of the volume.
DROP_SINGLE_SLICE = True
DROP_SINGLE_SLICE_IN = None   # None -> both techniques; ("3D stitched",) -> that one only

# Distribution plot -- the same knobs compare_brightness_distributions takes.
N_DISPLAY_BINS = 60
CLIP_QUANTILE = 0.999   # x axis stops here, on the pooled distribution
SHARE_Y = False         # the unmasked group is far more concentrated; False gives
                        # each panel its own height, as the notebook draws it
LOG_Y = False
MAKE_PLOT = True
PLOT_PATH = Path("z_brightness_cutoff.png")   # None -> show interactively

# Tables. CSV_PATH writes two files, "<stem>_within" and "<stem>_across".
CSV_PATH = None         # e.g. Path("z_brightness_cutoff.csv")

LAUNCH_NAPARI = True
Z_SCALE = None          # e.g. 15 -> stretch z for the 3D view on launch
# ----------------------------------------------------------------------------

# Row 1 of the (2, n_z, n_bins) counts is the masked group, row 0 the unmasked
# one -- accumulate_histograms' convention, and the order the two panels are in.
GROUP_ROWS = {"masked": 1, "unmasked": 0}
GROUPS = tuple(GROUP_ROWS)   # "Voxels in a mask", then "Voxels not in a mask"


# --- loading -----------------------------------------------------------------

def load_fov(technique_roots=TECHNIQUE_ROOTS, dapi_patches=DAPI_PATCHES,
             region=REGION, fov=FOV, drop=DROP_SINGLE_SLICE,
             drop_in=DROP_SINGLE_SLICE_IN, verbose=True):
    """One FOV under every technique: the image, the masks, and shared bins.

    Both techniques segmented the same DAPI stack, so it is read once and the
    bins are built from it once -- two distributions binned differently cannot
    be laid over each other, which is the whole reason
    ``accumulate_histograms`` takes its bins as an argument.

    Returns ``(dapi, masks_by_method, edges, counts_by_method, reports)``, where
    each ``counts`` is the ``(2, n_z, n_bins)`` array of per-plane histograms and
    each ``masks`` is the volume *after* the single-slice filter, so the volume
    the viewer shows is the one the histograms were taken from.
    """
    found = v2.find_region_fovs(technique_roots, region)
    rows = found[found["fov"] == fov]
    if rows.empty:
        raise FileNotFoundError(
            f"{fov} is not a paired FOV in {region}; "
            f"have: {', '.join(sorted(found['fov'].unique()))}")

    dapi, files = v2.load_dapi(v2.fov_dapi_glob(Path(dapi_patches) / region, fov))
    edges, exact = v2.build_bin_edges(dapi)
    if verbose:
        print(f"{region} / {fov}: {len(files)} DAPI planes, shape {dapi.shape}, "
              f"dtype {dapi.dtype}")

    masks_by_method, counts_by_method, reports = {}, {}, {}
    for row in rows.itertuples():
        method = str(row.method)
        volume = v2.load_masks(row.path)
        if volume.shape != dapi.shape:
            raise ValueError(f"{region} / {fov} / {method}: masks shape "
                             f"{volume.shape} != dapi shape {dapi.shape}")

        kept, _dropped, _areas, report = v2.filter_single_slice(
            volume, drop=v2.drops_here(method, drop_in, drop))

        brightness, present = v2.vectorise_pixels(dapi, kept)
        counts_by_method[method] = v2.accumulate_histograms(
            brightness, present, edges, exact=exact)
        masks_by_method[method] = kept
        reports[method] = report
        del present

        if verbose:
            n_masked = int(counts_by_method[method][1].sum())
            print(f"  {method:<12} {report['n_total']:>5} masks in -> "
                  f"{report['n_total'] - report['n_dropped']:>5} kept "
                  f"({report['n_dropped']} single-slice dropped)   "
                  f"{n_masked:>12,} masked voxels "
                  f"({100 * n_masked / dapi.size:5.2f}%)")

    return dapi, masks_by_method, edges, counts_by_method, reports


# --- the cutoff, read off the histograms -------------------------------------

def exact_bins(edges) -> bool:
    """Is this the one-bin-per-intensity-value grid ``build_bin_edges`` emits?

    Integer volumes get unit-width bins centred on the values, i.e. edges on the
    half-integers. Float volumes get uniform bins of arbitrary width, and the
    test has to fail for those.
    """
    edges = np.asarray(edges, dtype=float)
    return bool(np.allclose(np.diff(edges), 1.0)
                and np.isclose(edges[0] % 1.0, 0.5))


def cut_point(edges, cutoff) -> float:
    """Where to cut a binned distribution so "below" means *strictly* below.

    On the exact grid an integer cutoff lands dead centre of the bin holding
    that intensity value, so splitting the bin proportionally would put half the
    voxels of value 7,000 on each side of a rule they are not below. Snapping to
    the edge underneath puts them where the rule says they belong: at or above.

    Off that grid a bin covers a range of values, so there is nothing to snap to
    and interpolating inside the straddled bin is the best available estimate --
    the boundary is returned unchanged.
    """
    if exact_bins(edges) and float(cutoff).is_integer():
        return float(cutoff) - 0.5
    return float(cutoff)


def below_weights(edges, cutoff=BRIGHTNESS_CUTOFF) -> np.ndarray:
    """Per-bin share of a histogram that lies below the cutoff.

    The one place the rule is applied. Every number in both tables is read off
    the two histograms this splits a distribution into, so the shares and the
    per-side statistics can never disagree about where the line is.

    A bin the cut falls inside is divided by the share of its width on each side
    -- the same proportional rule ``v2.hist_cdf_at`` interpolates with. On the
    exact grid the cut lands on a bin edge, so every weight comes out 0 or 1 and
    nothing is divided at all.
    """
    edges = np.asarray(edges, dtype=float)
    widths = np.diff(edges)
    share = (cut_point(edges, cutoff) - edges[:-1]) / np.where(widths > 0, widths, 1.0)
    return np.clip(share, 0.0, 1.0)


def split_counts(counts, weights):
    """``(below, above)`` -- one histogram cut in two, as float counts.

    Float rather than integer because a straddled bin is divided between the
    sides. Nothing downstream needs whole voxels: the shares are ratios and
    ``hist_auc`` takes weights.
    """
    counts = np.asarray(counts, dtype=float)
    return counts * weights, counts * (1.0 - weights)


def area_either_side(counts, weights):
    """``(auc_below, auc_above)`` -- the area under one curve either side of the rule.

    The plotted curve is this histogram normalised by its own total, so its whole
    area is 1 and these two sum to it. Nothing rank-based here: it is the area
    under the curve, read off the same split every other number comes from.
    """
    below, above = split_counts(counts, weights)
    total = below.sum() + above.sum()
    if total <= 0:
        return float("nan"), float("nan")
    return float(below.sum() / total), float(above.sum() / total)


def pooled_groups(counts) -> dict:
    """``{"masked": (n_bins,), "unmasked": (n_bins,)}`` -- the z axis collapsed."""
    return {name: counts[row].sum(axis=0) for name, row in GROUP_ROWS.items()}


WITHIN_COLUMNS = ["pct_below", "pct_above", "auc_below", "auc_above"]


def cutoff_table(counts_by_method, edges, cutoff=BRIGHTNESS_CUTOFF) -> pd.DataFrame:
    """One row per (technique, group): the curve's area either side of the cutoff.

    ``auc_below`` and ``auc_above`` are that area, taken literally -- the curve
    is normalised within (technique, group), so its whole area is 1 and the two
    sides sum to it. ``pct_below`` and ``pct_above`` are the same two numbers
    times 100.

    Every row is one curve in one panel of the figure, and its two areas are
    the figure's own answer to where that distribution sits relative to the
    rule. Nothing here is rank-based; the rank statistic lives in the across
    table, which compares two curves rather than splitting one.
    """
    weights = below_weights(edges, cutoff)
    rows = []

    for method, counts in counts_by_method.items():
        groups = pooled_groups(counts)
        for name in GROUPS:
            auc_below, auc_above = area_either_side(groups[name], weights)
            rows.append({
                "method": method,
                "group": name,
                "pct_below": 100.0 * auc_below,
                "pct_above": 100.0 * auc_above,
                "auc_below": auc_below,
                "auc_above": auc_above,
            })

    table = pd.DataFrame(rows, columns=["method", "group"] + WITHIN_COLUMNS)
    return table.set_index(["method", "group"])


ACROSS_COLUMNS = ["pct_above_a", "pct_above_b", "d_pct_above",
                  "auc_below", "auc_above"]


def cross_cutoff_table(counts_by_method, edges,
                       cutoff=BRIGHTNESS_CUTOFF) -> pd.DataFrame:
    """The same questions between techniques: one row per (group, pair).

    ``pct_above_a`` and ``pct_above_b`` are each technique's share of that group
    on the bright side of the rule, and ``d_pct_above`` is ``a`` minus ``b``.
    The dim side is not carried: each technique's two shares sum to 100, so the
    difference below the rule is exactly the negative of the one above it.

    ``auc_below`` and ``auc_above`` are P(a random voxel from ``a``'s group is
    brighter than a random one from ``b``'s same group), on one side of the rule
    at a time -- the rank statistic, *not* the area the within table's columns of
    the same name carry. 0.5 says the two techniques' distributions on that side
    are interchangeable, above 0.5 says ``a``'s are systematically brighter.

    The pair of them is the useful part. Two techniques can hold the same share
    above a cutoff while the voxels they hold there sit at quite different
    brightnesses, and a share cannot show that -- ``auc_above`` can.
    """
    methods = list(counts_by_method)
    weights = below_weights(edges, cutoff)
    pooled = {m: pooled_groups(c) for m, c in counts_by_method.items()}
    sides = {(m, g): split_counts(pooled[m][g], weights)
             for m in methods for g in GROUPS}
    shares = {(m, g): area_either_side(pooled[m][g], weights)
              for m in methods for g in GROUPS}

    rows = []
    for name in GROUPS:
        for method_a, method_b in combinations(methods, 2):
            below_a, above_a = sides[method_a, name]
            below_b, above_b = sides[method_b, name]
            pct_above_a = 100.0 * shares[method_a, name][1]
            pct_above_b = 100.0 * shares[method_b, name][1]

            rows.append({
                "group": name,
                "method_a": method_a,
                "method_b": method_b,
                "pct_above_a": pct_above_a,
                "pct_above_b": pct_above_b,
                "d_pct_above": pct_above_a - pct_above_b,
                "auc_below": v2.hist_auc(below_a, below_b),
                "auc_above": v2.hist_auc(above_a, above_b),
            })

    table = pd.DataFrame(rows, columns=["group", "method_a", "method_b"] + ACROSS_COLUMNS)
    return table.set_index(["group", "method_a", "method_b"])


def cutoff_report(counts_by_method, edges, cutoff=BRIGHTNESS_CUTOFF) -> dict:
    """Both tables in one call: ``{"within": ..., "across": ..., "cutoff": ...}``."""
    return {"within": cutoff_table(counts_by_method, edges, cutoff),
            "across": cross_cutoff_table(counts_by_method, edges, cutoff),
            "cutoff": cutoff}


def print_tables(within, across) -> None:
    """The two tables, and nothing else.

    A prose read-out of the same numbers used to sit above these; it said what
    the columns already say, so the tables stand on their own now.
    """
    print("\n--- within each technique " + "-" * 40)
    print(within.round(4).to_string())
    print("\n--- across techniques " + "-" * 44)
    print(across.round(4).to_string())


def write_tables(within, across, path=CSV_PATH) -> None:
    """``<stem>_within.csv`` and ``<stem>_across.csv`` beside ``path``."""
    if path is None:
        return
    path = Path(path)
    for name, table in (("within", within), ("across", across)):
        out = path.with_name(f"{path.stem}_{name}{path.suffix or '.csv'}")
        table.to_csv(out)
        print(f"Saved {name} table to {out}")


# --- the figure --------------------------------------------------------------

def draw_cutoff(fig, cutoff=BRIGHTNESS_CUTOFF, shade=True) -> None:
    """Add the cutoff rule to a ``compare_brightness_distributions`` figure.

    Drawn afterwards rather than inside that function, so the figure this script
    produces is the notebook's figure with one thing added and nothing changed.

    The rule and a tint on the dim side, and nothing else: unlabelled, because
    the title carries the number and the panels do not need to repeat it twice
    each. The shares either side are in the tables, where four numbers per panel
    read better than they would laid over the curves they describe. Where the
    distributions fall relative to the line is what the figure is for, and that
    needs only the line.
    """
    axes = list(fig.axes)
    lo, hi = axes[0].get_xlim()
    if not lo <= cutoff <= hi:
        print(f"\ncutoff {cutoff:,} is outside the plotted range "
              f"({lo:,.0f}-{hi:,.0f}); no rule drawn")
        return

    for ax in axes:
        if shade:
            # the dim side, tinted -- "either side of the cutoff" should be
            # visible as two regions and not just as a line
            ax.axvspan(lo, cutoff, color=v2.INK_SECONDARY, alpha=0.05,
                       linewidth=0, zorder=0)
        ax.axvline(cutoff, color=v2.INK, linestyle=(0, (6, 3)), linewidth=1.6,
                   zorder=5)


def plot_cutoff_distributions(counts_by_method, edges,
                              cutoff=BRIGHTNESS_CUTOFF, title=None,
                              path=PLOT_PATH, n_bins=N_DISPLAY_BINS,
                              clip_quantile=CLIP_QUANTILE, share_y=SHARE_Y,
                              log_y=LOG_Y, **kwargs):
    """The notebook's brightness figure, with the cutoff drawn across both panels."""
    if title is None:
        title = f"{REGION}  ·  {FOV}  ·  brightness by mask presence, cutoff {cutoff:,}"

    fig = bh.compare_brightness_distributions(
        counts_by_method, edges, title=title, n_bins=n_bins,
        clip_quantile=clip_quantile, share_y=share_y, log_y=log_y, **kwargs)
    if fig is None:
        return None

    draw_cutoff(fig, cutoff=cutoff)

    if path is not None:
        fig.savefig(path, dpi=200)
        print(f"\nSaved plot to {path}")
    else:
        plt.show()   # blocks until closed, napari launches afterwards

    return fig


# --- napari ------------------------------------------------------------------

def build_cutoff_layers(dapi, masks_by_method, cutoff=BRIGHTNESS_CUTOFF,
                        verbose=True) -> dict:
    """Per technique, the two corners where the mask and the cutoff disagree.

    ``dim, masked``    inside a mask yet below the cutoff -- background the
                       technique pulled in.
    ``bright, unmasked`` outside every mask yet at or above it -- signal it left
                       behind.

    Boolean volumes rather than labels: "voxel outside every mask and over 7,000
    counts" is not an object. Two ``uint8`` volumes per technique, each the size
    of the stack, so this is the memory-expensive step -- the agreement corners
    are deliberately not built, since the mask volume already shows one and the
    rest of the image is the other.

    The threshold is the same number in both techniques, which is the point: the
    layers can be compared between viewers, which quantile-defined ones cannot.
    """
    layers = {}
    for method, masks in masks_by_method.items():
        present = masks > 0
        dim_masked = (present & (dapi < cutoff)).astype(np.uint8)
        bright_unmasked = (~present & (dapi >= cutoff)).astype(np.uint8)
        layers[method] = {
            f"dim, masked (< {cutoff:,})": dim_masked,
            f"bright, unmasked (>= {cutoff:,})": bright_unmasked,
        }
        if verbose:
            print(f"  {method:<12} {int(dim_masked.sum()):>12,} dim-but-masked   "
                  f"{int(bright_unmasked.sum()):>12,} bright-but-unmasked")
        del present

    return layers


def launch_viewers(dapi, masks_by_method, layers=None, title="", z_scale=Z_SCALE):
    """One viewer per technique over the same DAPI. Blocks until all are closed.

    Every viewer is built before ``napari.run()`` is called once, so the two
    windows are open together and the same nucleus can be found under each --
    calling ``run()`` per viewer would show them one after the other.
    """
    import napari  # Qt is the one dependency a headless machine usually lacks

    viewers = []
    for method, masks in masks_by_method.items():
        viewer = napari.Viewer(title=f"{title} {method}".strip())
        viewer.add_image(
            dapi, name="DAPI (deconvolved)", colormap="gray",
            contrast_limits=[float(dapi.min()), float(np.percentile(dapi, 99.5))],
        )
        viewer.add_labels(masks, name=f"{method} masks", opacity=0.4)

        for name, volume in (layers or {}).get(method, {}).items():
            viewer.add_labels(volume, name=name, opacity=0.6, visible=False,
                              blending="additive")

        if z_scale is not None:
            v2.make_3d(viewer, z_scale)
        viewers.append(viewer)

    print("\nLaunching napari viewer(s)")
    napari.run()
    return viewers


def main():
    """The whole thing on one FOV: tables, figure, viewers.

    Every value from the CONFIG block is read here and passed down explicitly
    rather than left to the functions' defaults, so the module can be imported
    and re-pointed -- ``cut.FOV = "fov_09"; cut.main()`` -- and not just edited
    in place. A default argument is bound at import, a global is read at call.

    Returns ``{"within", "across", "cutoff", "figure"}``: the two tables, the
    rule they were computed on, and the figure it was drawn on.
    """
    v2.apply_theme()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    cutoff = BRIGHTNESS_CUTOFF
    dapi, masks_by_method, edges, counts_by_method, _reports = load_fov(
        technique_roots=TECHNIQUE_ROOTS, dapi_patches=DAPI_PATCHES,
        region=REGION, fov=FOV,
        drop=DROP_SINGLE_SLICE, drop_in=DROP_SINGLE_SLICE_IN)

    report = cutoff_report(counts_by_method, edges, cutoff)
    within, across = report["within"], report["across"]
    print_tables(within, across)
    write_tables(within, across, CSV_PATH)

    report["figure"] = None
    if MAKE_PLOT:
        report["figure"] = plot_cutoff_distributions(
            counts_by_method, edges, cutoff=cutoff,
            title=f"{REGION}  ·  {FOV}  ·  brightness by mask presence, "
                  f"cutoff {cutoff:,}",
            path=PLOT_PATH, n_bins=N_DISPLAY_BINS, clip_quantile=CLIP_QUANTILE,
            share_y=SHARE_Y, log_y=LOG_Y)

    if LAUNCH_NAPARI:
        print()
        layers = build_cutoff_layers(dapi, masks_by_method, cutoff)
        launch_viewers(dapi, masks_by_method, layers,
                       title=f"{REGION} {FOV} — cutoff {cutoff:,}",
                       z_scale=Z_SCALE)

    return report


if __name__ == "__main__":
    main()
