import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

# CONFIG
# ----------------------------------------------------------------------------
MASKS_DIR = Path("data/segmentations_3d_true/cpdino/raw/VePo/region_UCI-5224/fov_07")
DAPI_DIR = Path("data/patches/VePo/region_UCI-5224/fov_07")

MASKS_PATH = MASKS_DIR / "masks.tif"
DAPI_GLOB = str(DAPI_DIR / "DAPI_decon_z*.tif") # Each z index

LAYER_SPAN_CUTOFF = 1

# Area-change analysis
AREA_CHANGE_CUTOFF = 3        # flag objects changing > this many x the mean change
CHANGE_REFERENCE = "layer"      # "layer" = mean of that z-transition, "global" = mean of all transitions
OBJECT_STATISTIC = "max"        # collapse each object's transitions with "max" or "mean"
MAKE_PLOT = True
PLOT_PATH = Path("z_area_change.png")                # e.g. Path("z_area_change.png"); None -> show interactively

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
    n_objects = int(masks.max())

    print(f"Volume shape: {dapi.shape}  |  {n_objects} labeled objects")
    return dapi, masks, files


def check_z_span(mask_volume, layer_span_cutoff=1):

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


def build_span_layers(masks, df):
    normal_cutoff = df.loc[df["z_span"] == LAYER_SPAN_CUTOFF, "label"].to_numpy()
    next_cutoff = df.loc[df["z_span"] == LAYER_SPAN_CUTOFF + 1, "label"].to_numpy()

    normal_cutoff_layer = np.where(np.isin(masks, normal_cutoff), masks, 0).astype(masks.dtype)
    next_cutoff_layer = np.where(np.isin(masks, next_cutoff), masks, 0).astype(masks.dtype)

    return normal_cutoff_layer, next_cutoff_layer


def build_change_layer(masks, df):
    flagged = df.loc[df["large_area_change"], "label"].to_numpy()
    return np.where(np.isin(masks, flagged), masks, 0).astype(masks.dtype)


def give_output_summary(df):
    n_single = int((df["z_span"] == 1).sum())
    n_two = int((df["z_span"] == 2).sum())
    n_multi = int(df["spans_multiple_layers"].sum())

    print("\n=== Z-SPAN SUMMARY ===")
    print(f"  n_objects: {len(df)}")
    print(f"  n_single_slice (z_span==1): {n_single}")
    print(f"  n_two_slice (z_span==2): {n_two}")
    print(f"  n_spans_multiple_layers (z_span>{LAYER_SPAN_CUTOFF}): {n_multi}")


def give_change_summary(df, changes, layer_stats):
    print("\n=== AREA-CHANGE SUMMARY ===")
    if changes.empty:
        print("  no objects span more than one z slice; nothing to compare")
        return

    print(f"  n_transitions measured: {len(changes)}")
    print(f"  mean |Δarea|/area: {changes['abs_frac_change'].mean():.3f}")
    print(f"  median |Δarea|/area: {changes['abs_frac_change'].median():.3f}")
    print(f"  reference: {CHANGE_REFERENCE}  |  object statistic: {OBJECT_STATISTIC}")
    print(f"  n_large_area_change (> {AREA_CHANGE_CUTOFF}x mean): "
          f"{int(df['large_area_change'].sum())}")

    print("\n  per-layer mean |Δarea|/area:")
    for row in layer_stats.itertuples():
        std = 0.0 if pd.isna(row.std_change) else row.std_change
        print(f"    z {row.z_from}->{row.z_to}: {row.mean_change:.3f} "
              f"± {std:.3f}  (n={row.n_objects})")


def plot_area_change(changes, layer_stats, cutoff=AREA_CHANGE_CUTOFF, path=PLOT_PATH,
                     hist_xmax=4.0, n_bins=50, hist_ymax=0.60, layer_ymax=2.5,
                     spread="iqr"):
    
    if changes.empty:
        print("\nNo z-transitions to plot")
        return None

    vals = changes["abs_frac_change"].to_numpy()
    global_mean = vals.mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- left panel: distribution of per-transition area change --------------
    ax = axes[0]
    top = float(hist_xmax) if hist_xmax is not None else float(vals.max())
    bins = np.linspace(0, top, n_bins + 1)
    n_over = int((vals > top).sum())

    ax.hist(np.clip(vals, None, top), bins=bins,
            weights=np.full(vals.size, 1.0 / vals.size),
            color="0.5", edgecolor="white")
    ax.axvline(global_mean, color="tab:blue",
               label=f"mean = {global_mean:.3f}")
    ax.axvline(cutoff * global_mean, color="tab:red", linestyle="--",
               label=f"{cutoff}x mean = {cutoff * global_mean:.3f}")

    if n_over:
        ax.annotate(f"{n_over} / {vals.size} "
                    f"({100 * n_over / vals.size:.1f}%) > {top:g}\n"
                    f"clipped into last bin",
                    xy=(0.97, 0.70), xycoords="axes fraction",
                    ha="right", va="top", fontsize=9, color="0.3")

    ax.set_xlim(0, top)
    if hist_ymax is not None:
        ax.set_ylim(0, hist_ymax)
    ax.set_xlabel("|Δ area| / area  (per z transition)")
    ax.set_ylabel("fraction of object-transitions")
    ax.set_title("Distribution of between-layer area change")
    ax.legend(loc="upper right")

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


def launch_viewer(dapi, masks, single_layer_vol, two_layer_vol, change_vol=None):

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
        viewer.add_labels(change_vol, name=F">{AREA_CHANGE_CUTOFF}x mean area change",
                          opacity=0.6, visible=False, blending='additive')

    print("\nLaunching napari viewer")
    napari.run()
    return viewer


def main():
    dapi, masks, files = load_data(DAPI_GLOB, MASKS_PATH)
    df = check_z_span(masks, layer_span_cutoff=LAYER_SPAN_CUTOFF)

    changes = check_area_change(masks)
    layer_stats = summarise_layer_change(changes)
    per_object = flag_area_outliers(changes, layer_stats,
                                    cutoff=AREA_CHANGE_CUTOFF,
                                    reference=CHANGE_REFERENCE,
                                    statistic=OBJECT_STATISTIC)

    df = df.merge(per_object, on="label", how="left")
    df["large_area_change"] = df["large_area_change"].fillna(False).astype(bool)

    single_layer_vol, two_layer_vol = build_span_layers(masks, df)
    change_vol = build_change_layer(masks, df)

    give_output_summary(df)
    give_change_summary(df, changes, layer_stats)

    if MAKE_PLOT:
        plot_area_change(changes, layer_stats, cutoff=AREA_CHANGE_CUTOFF, path=PLOT_PATH)

    if LAUNCH_NAPARI:
        launch_viewer(dapi, masks, single_layer_vol, two_layer_vol, change_vol)


# Might want to change DAPI colormap to magma, reduce opacity to 0.7, and gamma to 1.5

if __name__ == "__main__":
    main()