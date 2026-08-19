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
MASKS_DIR = Path("data/segmentations_3d_true/cpdino/decon/VePo/region_UWA-7648/fov_09")
DAPI_DIR = Path("data/patches/VePo/region_UWA-7648/fov_09")

MASKS_PATH = MASKS_DIR / "masks.tif"
DAPI_GLOB = str(DAPI_DIR / "DAPI_decon_z*.tif")  # Each z index

SWAY_CUTOFF = None         # a factor pins the cutoff and overrides CUTOFF_QUANTILE.
                           # A sway is dimensionless, so an absolute number means the
                           # same thing in every FOV: 3.0 sits above the sharpest
                           # healthy arc and well below a stitching step
CUTOFF_QUANTILE = 0.90     # None cutoff -> flag the top 10% of this FOV's sways
AREA_FLOOR = 20            # px; area differences below this are noise, not shape

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


SWAY_COLUMNS = ["label", "n_planes", "area_min", "area_max", "sway", "z_at_sway"]


def check_area_sway(areas, area_floor=AREA_FLOOR):
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

    log_area = np.log10(np.maximum(areas, area_floor))

    if areas.shape[0] >= 3:
        triple = present[:-2] & present[1:-1] & present[2:]
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
        "area_min": absent_high.min(axis=0)[labels],
        "area_max": areas.max(axis=0)[labels],
        "sway": sway[labels],
        "z_at_sway": z_at_sway[labels],
    })


def flag_sway(sways, cutoff=SWAY_CUTOFF, quantile=CUTOFF_QUANTILE):
    """Add ``large_sway``.  Returns ``(sways, cutoff, source)``.

    ``cutoff=None`` takes the quantile of the sways that exist — a rank
    statistic, so nothing is assumed about the distribution's shape, and masks
    too short to have a sway can't drag it down.
    """
    sways = sways.copy()
    if sways.empty:
        sways["large_sway"] = pd.Series(dtype=bool)
        return sways, float("inf"), "no masks"

    if cutoff is None:
        values = sways["sway"].dropna()
        cutoff = float(values.quantile(quantile)) if len(values) else float("inf")
        source = f"p{100 * quantile:g} of this FOV's sways"
    else:
        cutoff, source = float(cutoff), "set in CONFIG"

    sways["large_sway"] = sways["sway"] > cutoff  # NaN compares False
    return sways, cutoff, source


def build_sway_layer(masks, df):
    flagged = df.loc[df["large_sway"], "label"].to_numpy()
    return np.where(np.isin(masks, flagged), masks, 0).astype(masks.dtype)


def give_sway_summary(df, cutoff, source, n_worst=8):
    print("\n=== AREA-SWAY SUMMARY ===")
    values = df["sway"].dropna()
    if values.empty:
        print("  no mask has three consecutive planes; nothing to measure")
        return

    print(f"  n_objects: {len(df)}  |  with a measurable sway: {len(values)}")
    print("  sway quantiles: "
          + "  ".join(f"p{100 * q:g} {values.quantile(q):,.1f}x"
                      for q in (0.5, 0.75, 0.9, 0.95, 0.99))
          + f"  max {values.max():,.1f}x")

    n_flagged = int(df["large_sway"].sum())
    print(f"  cutoff: sway > {cutoff:,.1f}x  [{source}]")
    print(f"  n_large_sway: {n_flagged} / {len(values)} "
          f"({100 * n_flagged / len(values):.1f}%)")

    if not n_flagged:
        return

    print("\n  worst offenders:")
    print("    label   planes   area_min -> area_max      sway   z@sway")
    for row in df.nlargest(min(n_worst, n_flagged), "sway").itertuples():
        print(f"    {int(row.label):>6}  {int(row.n_planes):>6}   "
              f"{int(row.area_min):>8,} -> {int(row.area_max):<8,}  {row.sway:>7.1f}x  "
              f"{int(row.z_at_sway):>7}")


def plot_area_sway(df, areas, cutoff, source, path=PLOT_PATH,
                   n_bins=45, n_profiles=10, n_background=80, seed=0):
    """Left: the sway distribution, log-binned, with the flagged region shaded —
    1.0x is a perfectly smooth profile, whatever its slope.  Right: area through
    z for the worst offenders over a grey sample of the rest, with the sway plane
    ringed, so the corner being measured is the thing you look at.
    """
    rows = df[df["sway"].notna()]
    if rows.empty:
        print("\nNo mask has three consecutive planes to plot")
        return None

    values = rows["sway"].to_numpy(dtype=float)
    flagged = rows["large_sway"].to_numpy(dtype=bool)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # --- left: the distribution ---------------------------------------------
    ax = axes[0]
    top = max(values.max(), cutoff * 1.2 if np.isfinite(cutoff) else 0.0, 1.1)
    bins = np.logspace(0, np.log10(top), n_bins + 1)  # a sway is >= 1 by construction

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
    ax.set_xlabel("sway = worst change in growth rate, as a factor (log scale)")
    ax.set_ylabel("fraction of masks")
    ax.set_title(f"Per-mask sway  ({int(flagged.sum())} flagged)")
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

    worst = rows.loc[flagged].nlargest(min(n_profiles, int(flagged.sum())), "sway")
    for row in worst.itertuples():
        line = profile(row.label)
        ax.plot(z_axis, line, color="tab:red", alpha=0.85, linewidth=1.4,
                marker="o", markersize=3, zorder=3)
        ax.plot(row.z_at_sway, line[int(row.z_at_sway)], marker="o", markersize=9,
                markerfacecolor="none", markeredgecolor="tab:red", zorder=4)

    ax.plot([], [], color="0.6", linewidth=0.9, label=f"unflagged sample (n={sample.size})")
    ax.plot([], [], color="tab:red", linewidth=1.4, marker="o", markersize=3,
            label=f"worst {len(worst)} flagged (sway ringed)")

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
                  area_floor=AREA_FLOOR):
    """Returns ``(df, areas, cutoff, source)`` — one row per label, plus the
    per-plane areas the profile panel draws from."""
    areas = measure_slice_areas(masks)
    df, cutoff, source = flag_sway(check_area_sway(areas, area_floor), cutoff, quantile)
    return df, areas, cutoff, source


def main():
    dapi, masks, files = load_data(DAPI_GLOB, MASKS_PATH)
    df, areas, cutoff, source = analyse_masks(masks)

    give_sway_summary(df, cutoff, source)

    if MAKE_PLOT:
        plot_area_sway(df, areas, cutoff, source, path=PLOT_PATH)

    if LAUNCH_NAPARI:
        launch_viewer(dapi, masks, build_sway_layer(masks, df), cutoff)


# Might want to change DAPI colormap to magma, reduce opacity to 0.7, and gamma to 1.5

if __name__ == "__main__":
    main()
