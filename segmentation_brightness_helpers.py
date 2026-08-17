"""Helpers for ``brightness_method_comparison.ipynb``.

The question is the one section 7 of ``segmentation_method_comparison.ipynb``
asks, against a different measurement: **run every FOV in a region under both
techniques and compare them**, where the measurement is *how bright the voxels a
technique claims are, against the ones it leaves behind*.

Nothing here re-implements the measurement. The histogram machinery --
``vectorise_pixels``, ``build_bin_edges``, ``accumulate_histograms``, and the
statistics that read off the counts -- is imported from
``segmentation_z_brightness_claude.py``, and the region walk (``find_region_fovs``)
and palette come from ``segmentation_z_helpers.py``. This module is only the
part neither of those has: **two mask volumes over the same image, on the same
bins, in the same panel.**

That last point is why ``accumulate_histograms`` takes its bins as an argument.
Both techniques segment the *same* DAPI stack, so the bins are built once per
FOV from that stack and both mask volumes are histogrammed onto them. Two
distributions binned differently cannot be laid over each other.

The panels split the opposite way from the single-run figure. There, the two
panels were masked vs unmasked for one technique; here each panel holds one of
those groups and the *techniques* are overlaid inside it, because the comparison
being made is between techniques.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.figure import Figure

import segmentation_z_brightness_claude as b
import segmentation_z_helpers as h

# CONFIG DEFAULTS
# ----------------------------------------------------------------------------
DAPI_PATTERN = "DAPI_decon_z*.tif"   # one file per z index, inside the FOV directory
N_DISPLAY_BINS = 60
CLIP_QUANTILE = 0.999                # x-axis stops here, on the pooled distribution
# ----------------------------------------------------------------------------


def load_dapi(dapi_glob):
    """The DAPI half of ``h.load_data``, without a mask volume to check against.

    Both techniques segment the same image, so the stack is read once per FOV
    and reused; ``load_data`` would re-read it per technique. Same glob, same
    numeric z ordering -- ``z10`` sorts after ``z9``, which a plain sort of the
    filenames gets wrong.
    """
    files = glob.glob(dapi_glob)

    def z_index(f):
        m = re.search(r"z(\d+)", Path(f).stem)
        return int(m.group(1)) if m else 0
    files = sorted(files, key=z_index)

    if not files:
        raise FileNotFoundError(f"No DAPI files matched pattern: {dapi_glob}")
    return np.stack([tifffile.imread(f) for f in files], axis=0), files


def fov_dapi_glob(dapi_root, fov, pattern=DAPI_PATTERN):
    """``<dapi_root>/<fov>/DAPI_decon_z*.tif`` -- the image both techniques saw."""
    return str(Path(dapi_root) / fov / pattern)


def scan_region_brightness(found, dapi_root, pattern=DAPI_PATTERN, verbose=True):
    """Histogram every FOV in ``found`` under every technique, one FOV at a time.

    ``found`` is what ``h.find_region_fovs`` returns: one row per (method, FOV)
    with the mask path. Returns ``{fov: {"edges": ..., "counts": {method: ...}}}``
    where each ``counts`` is the ``(2, n_z, n_bins)`` array
    ``accumulate_histograms`` produces -- index 0 unmasked, index 1 masked.

    Only the histograms are kept. They are a few hundred KB per run against
    hundreds of MB for the volumes, so a whole region fits in memory while no
    more than one DAPI stack and one mask volume are resident at a time.
    """
    runs = {}

    for fov, rows in found.groupby("fov", sort=True, observed=True):
        dapi, files = load_dapi(fov_dapi_glob(dapi_root, fov, pattern))
        edges, exact = b.build_bin_edges(dapi)
        counts = {}

        for row in rows.itertuples():
            masks = tifffile.imread(row.path)
            if masks.ndim == 2:
                masks = masks[np.newaxis]
            if masks.shape != dapi.shape:
                raise ValueError(f"{fov} / {row.method}: masks shape {masks.shape} "
                                 f"!= dapi shape {dapi.shape}")

            brightness, present = b.vectorise_pixels(dapi, masks)
            counts[str(row.method)] = b.accumulate_histograms(
                brightness, present, edges, exact=exact)
            del masks, brightness, present

            if verbose:
                n_masked = int(counts[str(row.method)][1].sum())
                print(f"  {fov:<10} {str(row.method):<12} "
                      f"{n_masked:>10,} masked voxels "
                      f"({100 * n_masked / dapi.size:5.2f}%)")

        runs[fov] = {"edges": edges, "counts": counts, "n_planes": len(files),
                     "shape": dapi.shape}
        del dapi

    return runs


def load_fov(technique_roots, dapi_patches, region, fov, pattern=DAPI_PATTERN):
    """Everything one FOV needs, without scanning its region first.

    ``dapi_patches`` is the directory *above* the region, the same way
    ``technique_roots`` is — the region name is appended here so a caller names
    it once. Returns ``(dapi, masks_by_method, edges, counts_by_method)``.

    This exists so the napari cell can name its own region and FOV instead of
    inheriting whichever region block ran last. The DAPI stack is read once and
    both mask volumes are histogrammed onto bins built from it, exactly as
    :func:`scan_region_brightness` does.
    """
    found = find_region_fovs_cached(technique_roots, region)
    rows = found[found["fov"] == fov]
    if rows.empty:
        raise FileNotFoundError(
            f"{fov} is not a paired FOV in {region}; "
            f"have: {', '.join(sorted(found['fov'].unique()))}")

    dapi, _ = load_dapi(fov_dapi_glob(Path(dapi_patches) / region, fov, pattern))
    edges, exact = b.build_bin_edges(dapi)

    masks, counts = {}, {}
    for row in rows.itertuples():
        volume = tifffile.imread(row.path)
        if volume.ndim == 2:
            volume = volume[np.newaxis]
        if volume.shape != dapi.shape:
            raise ValueError(f"{region} / {fov} / {row.method}: masks shape "
                             f"{volume.shape} != dapi shape {dapi.shape}")

        brightness, present = b.vectorise_pixels(dapi, volume)
        masks[str(row.method)] = volume
        counts[str(row.method)] = b.accumulate_histograms(
            brightness, present, edges, exact=exact)

    return dapi, masks, edges, counts


def find_region_fovs_cached(technique_roots, region, **kwargs):
    """``h.find_region_fovs`` without re-printing the skipped-FOV notice.

    The region blocks already reported which FOVs were dropped; repeating it
    every time the viewer cell runs is noise.
    """
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        return h.find_region_fovs(technique_roots, region, **kwargs)


def brightness_metrics(runs) -> pd.DataFrame:
    """One row per (FOV, technique) -- the numbers behind each figure.

    ``auc`` is P(a random masked voxel is brighter than a random unmasked one)
    within that run: 0.5 says brightness carries no information about what the
    technique masked, 1.0 says every masked voxel outshines every unmasked one.
    """
    rows = []
    for fov, run in runs.items():
        for method, counts in run["counts"].items():
            stats = b.summarise_pooled(counts, run["edges"])
            rows.append({"fov": fov, "method": method, **stats.to_dict()})

    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics
    return metrics.set_index(["fov", "method"])


def method_colors(methods) -> dict:
    """Hue per technique, fixed slot order -- the same assignment ``h`` uses.

    A technique keeps its colour across every figure in both notebooks, so a
    colour never changes meaning between them.
    """
    return {name: h.CATEGORICAL[i] for i, name in enumerate(methods)}


def display_range(runs, clip_quantile=CLIP_QUANTILE):
    """One (lo, hi) brightness span covering every FOV in the scan.

    Pass it to every figure and the whole stack of them shares an x axis, so
    FOVs can be read against each other and not just techniques within a FOV.
    Each FOV is binned on its own DAPI range, so without this the axis moves
    from figure to figure.
    """
    los, his = [], []
    for run in runs.values():
        # (2, n_z, n_bins) -> (n_bins,): both groups, every plane
        pooled = sum(c.sum(axis=(0, 1)) for c in run["counts"].values())
        nonzero = np.flatnonzero(pooled)
        if not nonzero.size:
            continue
        los.append(float(run["edges"][nonzero[0]]))
        his.append(b.hist_quantile(pooled, run["edges"], clip_quantile))

    if not los:
        return None
    return min(los), max(his)


def pool_runs(runs, n_bins=4096):
    """Sum every FOV's histograms onto one set of bins.

    Each FOV is binned on its own DAPI range, so the counts cannot simply be
    added — this re-bins them onto a common grid spanning the whole region
    first. Returns ``(edges, {method: (2, 1, n_bins)})``, the shape
    :func:`compare_brightness_distributions` expects, with the z axis already
    collapsed since a pooled figure has no single z to speak of.

    Re-binning costs a little resolution against the per-FOV figures, whose
    integer volumes get one bin per intensity value. It is a display aggregate;
    read the per-FOV numbers from ``brightness_metrics`` rather than from this.
    """
    if not runs:
        return None, {}

    lo = min(float(r["edges"][0]) for r in runs.values())
    hi = max(float(r["edges"][-1]) for r in runs.values())
    if hi <= lo:
        hi = lo + 1.0
    edges = np.linspace(lo, hi, n_bins + 1)

    methods = list(next(iter(runs.values()))["counts"])
    pooled = {m: np.zeros((2, 1, n_bins), dtype=np.int64) for m in methods}

    for run in runs.values():
        for method, counts in run["counts"].items():
            for group in (0, 1):
                pooled[method][group, 0] += b.rebin(
                    counts[group].sum(axis=0), run["edges"], edges).astype(np.int64)

    return edges, pooled


def compare_brightness_distributions(
        counts_by_method, edges, title="Pixel brightness by mask presence",
        n_bins=N_DISPLAY_BINS, clip_quantile=CLIP_QUANTILE, xlim=None,
        log_y=False, share_y=True) -> Figure:
    """Two panels, techniques overlaid inside each.

    Left: voxels the technique put in a mask. Right: voxels it did not. One
    curve per technique in each, normalised within (technique, panel) so runs
    with very different mask counts compare directly -- the masked group is a
    few percent of the volume and the unmasked group is the rest, and raw counts
    would only ever show that.

    Both panels share the x axis always and the y axis by default, so the two
    groups can be read against each other and not only within themselves. The
    unmasked group is far more concentrated, so it sets the shared height and
    the masked panel uses less of its own — pass ``share_y=False`` to give each
    panel its own scale when that comparison is the one you want.

    The masked curve sitting to the right of the unmasked one is the technique
    agreeing with the image; the gap between two techniques *within* a panel is
    where they disagree about the same voxels.
    """
    methods = list(counts_by_method)
    colors = method_colors(methods)
    styles = ("-", "--", "-.", ":")   # identity does not rest on hue alone

    pooled = {m: {"masked": c[1].sum(axis=0), "unmasked": c[0].sum(axis=0)}
              for m, c in counts_by_method.items()}
    if all(p["masked"].sum() == 0 for p in pooled.values()):
        print(f"{title}: no masked voxels in any technique; nothing to plot")
        return None

    # --- common bins for every curve in the figure ---------------------------
    if xlim is not None:
        lo, top = float(xlim[0]), float(xlim[1])
    else:
        combined = sum(p["masked"] + p["unmasked"] for p in pooled.values())
        nonzero = np.flatnonzero(combined)
        lo = float(edges[nonzero[0]])
        top = b.hist_quantile(combined, edges, clip_quantile)
    if top <= lo:
        top = lo + 1.0
    new_edges = np.linspace(lo, top, n_bins + 1)
    centers = b.bin_centers(edges)
    draw_at = b.bin_centers(new_edges)

    frac, totals, over = {}, {}, {}
    for method, groups in pooled.items():
        for group, counts in groups.items():
            total = int(counts.sum())
            totals[method, group] = total
            frac[method, group] = (b.rebin(counts, edges, new_edges) / total
                                   if total else np.zeros(n_bins))
            over[method, group] = int(counts[centers > top].sum())

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharex=True, sharey=share_y)
    panels = (("masked", "Voxels in a mask"), ("unmasked", "Voxels not in a mask"))

    for (group, panel_title), ax in zip(panels, axes):
        for i, method in enumerate(methods):
            if not totals[method, group]:
                continue
            values = frac[method, group]
            ax.fill_between(draw_at, values, step="mid",
                            color=colors[method], alpha=0.16, linewidth=0)
            ax.step(draw_at, values, where="mid", color=colors[method],
                    linestyle=styles[i % len(styles)], linewidth=2, zorder=3,
                    label=f"{method}  (n={totals[method, group]:,})")

            median = b.hist_quantile(pooled[method][group], edges, 0.5)
            ax.axvline(median, color=colors[method],
                       linestyle=styles[i % len(styles)], linewidth=1.2,
                       alpha=0.55, zorder=2)

        clipped = [f"{method}: {over[method, group]:,}"
                   for method in methods if over[method, group]]
        if clipped:
            ax.annotate(f"above {top:,.0f}, clipped into the last bin\n"
                        + "   ".join(clipped),
                        xy=(0.98, 0.72), xycoords="axes fraction",
                        ha="right", va="top", fontsize=8.5, color=h.INK_SECONDARY)

        ax.set_title(panel_title, pad=10, loc="left")
        ax.set_xlabel("pixel brightness")
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)
        ax.spines["bottom"].set_color(h.GRID)
        ax.spines["left"].set_visible(False)
        ax.legend(loc="upper right")

        if not share_y and not log_y:
            own = max(frac[m, group].max() for m in methods)
            ax.set_ylim(0, own * 1.18)

    axes[0].set_ylabel("fraction of voxels in group")
    axes[0].set_xlim(lo, top)
    if log_y:
        axes[0].set_yscale("log")
    elif share_y:
        axes[0].set_ylim(0, max(v.max() for v in frac.values()) * 1.18)

    fig.suptitle(title, x=0.005, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_region_brightness(runs, metrics=None, share_x=True, **kwargs):
    """One figure per FOV, in FOV order.

    With ``share_x`` every figure gets the same brightness axis, so the stack
    reads down as well as across.
    """
    xlim = kwargs.pop("xlim", None)
    if share_x and xlim is None:
        xlim = display_range(runs, clip_quantile=kwargs.get("clip_quantile",
                                                            CLIP_QUANTILE))

    figures = {}
    for fov, run in runs.items():
        subtitle = ""
        if metrics is not None and fov in metrics.index.get_level_values("fov"):
            aucs = metrics.loc[fov, "auc"]
            subtitle = "   ·   " + "   ".join(
                f"{m} AUC {v:.3f}" for m, v in aucs.items())

        figures[fov] = compare_brightness_distributions(
            run["counts"], run["edges"], title=f"{fov}{subtitle}",
            xlim=xlim, **kwargs)

    return figures


def build_overlap_layers(dapi, masks_by_method, edges, counts_by_method,
                         quantile=0.5):
    """Per technique, the two tails of the masked/unmasked overlap.

    ``bright_unmasked`` is outside every mask yet brighter than ``quantile`` of
    that technique's masked voxels -- signal it left behind. ``dim_masked`` is
    inside a mask yet dimmer than ``quantile`` of its unmasked voxels --
    background it pulled in. Thresholds are per technique, since each one's own
    distribution is what defines bright and dim for it.
    """
    layers = {}
    for method, masks in masks_by_method.items():
        counts = counts_by_method[method]
        present = masks > 0
        bright_thr = b.hist_quantile(counts[1].sum(axis=0), edges, quantile)
        dim_thr = b.hist_quantile(counts[0].sum(axis=0), edges, quantile)

        layers[method] = {
            "bright_unmasked": (~present & (dapi >= bright_thr)).astype(np.uint8),
            "dim_masked": (present & (dapi <= dim_thr)).astype(np.uint8),
            "bright_thr": bright_thr,
            "dim_thr": dim_thr,
        }
    return layers


def launch_viewer(dapi, masks_by_method, layers=None, title=""):
    """One napari viewer per technique over the same DAPI. Blocks until closed."""
    import napari  # Qt is the one dependency a headless kernel usually lacks

    viewers = []
    for method, masks in masks_by_method.items():
        viewer = napari.Viewer(title=f"{title} {method}".strip())
        viewer.add_image(
            dapi, name="DAPI (deconvolved)", colormap="gray",
            contrast_limits=[float(dapi.min()), float(np.percentile(dapi, 99.5))],
        )
        viewer.add_labels(masks, name=f"{method} masks", opacity=0.4)

        if layers is not None and method in layers:
            entry = layers[method]
            viewer.add_labels(entry["bright_unmasked"],
                              name=f"bright, unmasked (>{entry['bright_thr']:,.0f})",
                              opacity=0.6, visible=False, blending="additive")
            viewer.add_labels(entry["dim_masked"],
                              name=f"dim, masked (<{entry['dim_thr']:,.0f})",
                              opacity=0.6, visible=False, blending="additive")
        viewers.append(viewer)

    napari.run()
    return viewers
