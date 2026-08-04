import glob
import re
from pathlib import Path

import napari
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

# CONFIG
# ----------------------------------------------------------------------------
MASKS_DIR = Path("data/segmentations_3d_stitched/cpdino/decon_kde/CBDN/region_UCI-2424/fov_07")
DAPI_DIR = Path("data/patches/VePo/region_UCI-2424/fov_07")

MASKS_PATH = MASKS_DIR / "masks.tif"
DAPI_GLOB = str(DAPI_DIR / "DAPI_decon_z*.tif") # Each z index

LAYER_SPAN_CUTOFF = 1
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


def build_span_layers(masks, df):
    normal_cutoff = df.loc[df["z_span"] == LAYER_SPAN_CUTOFF, "label"].to_numpy()
    next_cutoff = df.loc[df["z_span"] == LAYER_SPAN_CUTOFF + 1, "label"].to_numpy()

    normal_cutoff_layer = np.where(np.isin(masks, normal_cutoff), masks, 0).astype(masks.dtype)
    next_cutoff_layer = np.where(np.isin(masks, next_cutoff), masks, 0).astype(masks.dtype)

    return normal_cutoff_layer, next_cutoff_layer


def give_output_summary(df):
    n_single = int((df["z_span"] == 1).sum())
    n_two = int((df["z_span"] == 2).sum())
    n_multi = int(df["spans_multiple_layers"].sum())

    print("\n=== Z-SPAN SUMMARY ===")
    print(f"  n_objects: {len(df)}")
    print(f"  n_single_slice (z_span==1): {n_single}")
    print(f"  n_two_slice (z_span==2): {n_two}")
    print(f"  n_spans_multiple_layers (z_span>{LAYER_SPAN_CUTOFF}): {n_multi}")


def launch_viewer(dapi, masks, single_layer_vol, two_layer_vol):

    viewer = napari.Viewer()
    viewer.add_image(
        dapi, name="DAPI (deconvolved)", colormap="gray",
        contrast_limits=[float(dapi.min()), float(np.percentile(dapi, 99.5))],
    )

    # Give each mask a random color
    viewer.add_labels(masks, name="segmentation masks", opacity=0.4)
    viewer.add_labels(single_layer_vol, name=F"{LAYER_SPAN_CUTOFF}-slice nuclei", opacity=0.6, visible=False)
    viewer.add_labels(two_layer_vol, name=F"{LAYER_SPAN_CUTOFF + 1}-slice nuclei", opacity=0.6, visible=False)

    print("\nLaunching napari viewer")
    napari.run()
    return viewer


def main():
    dapi, masks, files = load_data(DAPI_GLOB, MASKS_PATH)
    df = check_z_span(masks, layer_span_cutoff=LAYER_SPAN_CUTOFF)
    single_layer_vol, two_layer_vol = build_span_layers(masks, df)
    give_output_summary(df)
    if LAUNCH_NAPARI:
        launch_viewer(dapi, masks, single_layer_vol, two_layer_vol)


# Might want to change DAPI colormap to magma, reduce opacity to 0.7, and gamma to 1.5

if __name__ == "__main__":
    main()