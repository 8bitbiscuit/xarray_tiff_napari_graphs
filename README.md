# zspan — comparing segmentations by how far masks span in z

Cellpose-style 2D segmenters run per z-plane, so one nucleus visible on several
planes can come back as a single label stretched through z instead of one label
per plane. **The share of masks spanning more than one z-layer** is a cheap,
sensitive proxy for how much a run is over-merging in z.

This repo scores a directory of segmentation runs on that metric and compares
them visually — without ever loading a volume into memory.

```python
from zspan import scan_segmentations, add_variant_column, plot_variant_summary

result = scan_segmentations(
    "data/segmentations/cpdino",
    pattern="**/masks.tif",
    level_names=("preprocessing", "model", "region", "fov"),
)
summary = add_variant_column(result.summary, ["preprocessing", "model"])
plot_variant_summary(summary, group="variant", value="pct_multi_layer")
```

The worked example lives in
[`z_span_analysis_v4.ipynb`](z_span_analysis_v4.ipynb).
[`segmentation_method_comparison.ipynb`](segmentation_method_comparison.ipynb)
comes at the same question from the other end — one field at a time, volumes
loaded, images alongside — see
[below](#comparing-the-two-techniques-on-one-field).

## Install

```bash
pip install -e ".[xarray,dev]"
pytest
```

The tests build their own miniature TIFF trees in `tmp_path`, so they need no
data on disk. For the notebooks, point `TECHNIQUE_ROOTS` at your own output —
`segmentation_method_comparison.ipynb` also wants `DATA_ROOT` and its `METHODS`
dict, and reads volumes with `tifffile` (the `dev` extra). napari is imported
only if you ask for a viewer, so a headless kernel runs everything else.

## How it stays in bounds

Each multi-page TIFF is opened as a lazy zarr array whose chunks are the TIFF's
own tiles. Nothing is copied or converted; indexing fetches only the tiles the
window covers.

Two backends do that, chosen with `reader=`:

| | `"tifffile"` | `"virtual"` |
|---|---|---|
| how | `tifffile`'s `aszarr` store | `virtualizarr` + `virtual_tiff` IFD manifest |
| open, small tiled file | ~1 ms | 17 ms |
| open, 7×4000×4000 striped (1750 chunks) | ~1 ms | 325 ms |
| read, that same 448 MB volume @ 8 MiB blocks | 1.07 s | 1.35 s |
| peak memory | identical | identical |
| reads `s3://` via `obstore` | no | **yes** |
| manifest persists (Icechunk/Kerchunk) | no | **yes** |

`reader="auto"` (the default) routes on the path: a plain local file goes to
`tifffile`, anything with a non-`file` URL scheme — or any call given an
`ObjectStoreRegistry` — goes to `virtual`. So the same code moves to object
storage without an edit, and pays nothing for the manifest until it does.

The two are interchangeable by construction: same `chunks`, so the same block
sizes and the same tuning; identical statistics, asserted against the same
`find_objects` oracle under both. `reader="tifffile"` *raises* rather than
silently ignores a registry, a remote path, or an `ifd_layout` — a switch that
quietly reads the wrong thing is worse than no switch.

The metric is then computed by streaming. The observation that makes it work:

> a label's z bounding box is just `min`/`max` over the planes it appears on,
> and min/max are associative.

So the extent accumulates plane by plane, tile by tile, in any order — peak
memory is one tile plus two label-indexed arrays, i.e. `O(max_label)` and
**independent of image size**. On the demo data that is 0.5 MB streamed against
7.5 MB loaded; the gap widens linearly as volumes grow.

This is not an approximation. `tests/test_zspan.py` keeps the original
`scipy.ndimage.find_objects` implementation as an oracle and asserts the
streaming version reproduces it exactly — across random volumes, z-gaps, empty
planes, sparse and non-contiguous labels, and every block shape.

## Layout

The scanner maps directory levels to columns, so any nesting works:

```
data/segmentations/cpdino/<preprocessing>/<model>/<region>/<fov>/masks.tif
                          └── level_names=("preprocessing","model","region","fov")
```

`level_names` aligns to the **trailing** directories — naming just
`("region", "fov")` leaves the levels above as `level_0`, `level_1`.

To compare *segmentation techniques*, scan one root per technique and tag each
scan, so the technique becomes a column rather than a directory level — which is
what `z_span_analysis_v4.ipynb` does:

```
data/segmentations_3d_stitched/<model>/<region>/<fov>/masks.tif
data/segmentations_3d_true/<model>/<region>/<fov>/masks.tif
                           └── level_names=("model","region","fov")
```

`relpath` is relative to its own root, so both trees reuse the same values —
join across them on `technique` + `relpath`, never `relpath` alone.

## What you get back

`scan_segmentations` returns a `ScanResult`:

| | |
|---|---|
| `.summary` | one row per volume — level columns plus `n_labels`, `pct_multi_layer`, `mean_z_span`, `max_z_span`, `n_with_z_gaps` |
| `.labels` | one row per label — `z_start`, `z_end`, `z_span`, `n_planes`, `n_voxels`, `spans_multiple_layers` |
| `.failures` | `(path, exception)` for volumes that could not be read; one bad file never stops a scan |

`z_span` is the inclusive bounding-box depth; `n_planes` is how many planes the
label actually occupies. They differ only when a label has a hole in z, so
**`n_planes < z_span` flags a probable merge of two objects stacked in z** — a
harder error than plain over-merging, and one `find_objects` alone won't show
you.

### Per-plane areas

`check_z_span` collapses a label to its extent. `label_plane_areas` stops one
step earlier and keeps the label's **area on each plane**, which is what a
change in size between layers is computed from:

```python
from zspan import label_plane_areas, size_change_between_layers

areas = label_plane_areas(volume, n_sections=4)   # label, z, section, area
steps = size_change_between_layers(areas)         # ... delta, pct_change per z step
```

Same accumulation, same guarantee: one chunk-aligned block at a time, per-plane
results merged across blocks so a label straddling a tile boundary is counted
once, memory independent of image size. The areas are exact counts, and the
tests hold them against an in-memory oracle under every block shape.

`n_sections` splits each plane into that many equal horizontal bands
(`section_bounds` gives the row ranges) and reports an area per band, so one
scan answers both *what does the whole field do* and *does the top of it behave
like the bottom*. A mask crossing a divider is counted in every band it covers,
so summing `area` over `section` reproduces the whole-plane area **exactly** —
the whole-field reading is not an approximation of the banded one.

`size_change_between_layers` differences those along z within each mask. It
counts only consecutive observations of the same mask: a first plane has nothing
to change from, and steps that jump a hole in z are dropped by default
(`adjacent_only=False` keeps them, flagged by `z_gap > 1`) because a mask that
vanishes and returns is two objects, not one that resized. `group_cols` defaults
to every column except `z` and `area`, so a concatenated scan carrying
`technique`/`relpath`/`section` columns differences within each of them without
being told to.

## The plots

| Function | Question | Encoding |
|---|---|---|
| `plot_variant_summary` | *which variant* wins? | one bar per variant in a single hue, each volume overlaid as a dot |
| `plot_span_distribution` | *where* does the difference live? | share of labels at each z-span, categorical hues in fixed slot order |

`CATEGORICAL`, `SEQUENTIAL_BLUE` and `BLUES` are exported too, so notebook-side
figures stay consistent with these. `z_span_analysis_v4.ipynb` builds all three
of its views inline on top of them — see below — and
`segmentation_method_comparison.ipynb` copies the same values so its figures
match without importing the package.

Colours come from a validated palette — adjacent categorical pairs clear
colour-vision-deficiency separation thresholds. Three of the categorical hues
sit below 3:1 on a light surface, which is why both notebooks ship a summary
table next to the charts rather than as an extra.

### Choosing a model, across techniques

`z_span_analysis_v4.ipynb` targets a different question from the package defaults:
**which model produces fewest throwaway masks, and does that answer survive the
segmentation technique?** It scans one tree per technique — 2D-per-plane
`stitch_threshold` output against native `do_3D` output — and scores every run
on the *thin rate*: the share of masks occupying ≤ 2 z-slices, i.e. exactly what
a cellpose `min_z = 3` filter discards. Lower is better.

Technique owns the hue channel throughout, so a colour never changes meaning
between figures; model identity travels on row position and marker shape. Three
views:

1. **Which model wins, under which technique** — ranked rows, one bar per
   technique inside each model, one dot per FOV, because with a handful of FOVs
   per model a gap smaller than the within-model spread is not a result. Pairing
   the bars is what makes a *flip* — a model that wins under one technique and
   loses under the other — visible at all.
2. **Is the win real** — yield against thin rate, one point per FOV, with an
   arrow from each model's centroid under one technique to its centroid under
   the other. A model *or* a technique can post a low thin rate purely by not
   detecting faint cells; this separates *cleaner* from *more conservative*.
3. **Where the difference lives** — 100% stacked composition of mask thickness,
   one panel per technique on shared axes, rows in the ranking from view 1. Read
   down for the model comparison, across for the technique effect with the model
   held fixed. Single hue light→dark because thickness is *ordinal*.

The notebook prints a coverage table (FOVs scored per model per technique)
before any figure, so a comparison that is ragged rather than paired cannot pass
unnoticed.

Note the polarity: the package's `pct_multi_layer` treats spanning many slices
as the defect (over-merging), while the thin rate treats the opposite end as the
defect. Both come off the same `n_planes` column; pick whichever matches the
failure you are chasing.

## Comparing the two techniques on one field

[`segmentation_method_comparison.ipynb`](segmentation_method_comparison.ipynb)
puts the same two techniques — 2D-per-plane `stitch_threshold` against native
`do_3D` — on the same field, then scores every FOV in a region.

It runs on a different footing from the rest of the repo. The scanner above
streams and never holds a volume; this notebook loads them, through
[`segmentation_z_helpers.py`](segmentation_z_helpers.py). That is a choice, not
an oversight: the notebook hands the same arrays straight to napari to look at
the masks it just flagged, and per-plane areas for a single FOV are cheap once
the volume is already resident. It takes nothing from the package but the
palette in `zspan/plotting.py`, so figures from both tracks still sit together.

`segmentation_z_helpers.py` is `segmentation_z_claude.py` moved into a module —
same function bodies, same defaults, same printed wording. What changed is
listed in its docstring: the config globals became keyword arguments (the
notebook calls each function once per technique, and a global would silently
apply one technique's settings to the other), `plot_area_change` renders inline
by default, `import napari` moved inside `launch_viewer`, and `main()` became
`analyse_masks()` so the comparison loop can re-run the analysis without
re-narrating it.

### Two defects, two measurements

| | catches | how |
|---|---|---|
| **z-span** | masks stitching never joined up (yield a `min_z` filter discards) and masks reaching the whole stack (two nuclei read as one) | `check_z_span` — the inclusive z bounding box of every label |
| **area change** | the segmenter changing its mind between adjacent planes | `check_area_change` — `abs_frac_change` = \|Δ area\| / area, one row per (object, z-transition) |

Consecutive planes are a few hundred nanometres apart, so a nucleus's
cross-section should grow to a mid-plane and shrink away from it smoothly. A
mask whose area jumps between adjacent layers is usually the segmenter being
inconsistent, not a cell behaving oddly — and per-plane stitching has a
per-plane decision to be inconsistent about, where a native `do_3D` run does
not.

Only transitions where a label is present in **both** planes are counted, so a
mask appearing or disappearing never registers as a 100% change.
`flag_area_outliers` compares each transition against a reference —
`CHANGE_REFERENCE="layer"` uses the mean of that z-transition, which controls
for systematic drift through the stack, and `"global"` the mean over all of them
— then flags an object whose `OBJECT_STATISTIC` (`max` or `mean`) of that ratio
clears `AREA_CHANGE_CUTOFF`.

### What it draws

1. **`plot_area_change`**, per technique — the script's two-panel figure, with
   the left panel log-binned on a log x axis. `log_hist_range` runs the span
   from the decade holding the smallest *positive* change, floored
   `HIST_DECADES` below the largest, up to the largest observed: on a log axis
   the right tail costs almost no width, so there is no reason to truncate it.
   Log bins cannot hold zero, so both tails fold into the edge bins and are
   *counted in the annotation* rather than dropped — the exactly-stable
   transitions are usually a substantial share of real data. The log axis is a
   display transform and nothing more; the mean and the `cutoff × mean`
   threshold are still plain arithmetic means, so the flagged objects and the
   napari layer are exactly the linear version's. Drawing both techniques, pin
   `hist_xmin`/`hist_xmax` to the pooled range — a log axis that moves under you
   is not a comparison.
2. **`compare_span_distribution`** — share of masks at each z-span, grouped
   bars, normalised within technique so runs with different mask counts compare
   directly. The bar at 1 is the yield a `min_z` filter throws away; the `6+`
   bar is masks reaching most of the way through the stack.
3. **`compare_layer_change`** — mean |Δ area|/area per z-transition with its
   IQR band, both techniques on one axes, flag threshold dropped because two
   means and two thresholds is four lines to read where two carry the
   comparison. A technique sitting higher *everywhere* is less stable overall; a
   technique that tracks the other but spikes at one transition is usually one
   bad plane in the acquisition — and if it is, it spikes in both.

Technique owns the hue channel, and marker shape follows it too, so identity
never rests on colour alone.

### Scoring a whole region

Sections 1–6 walk a single field. Section 7 scans a region:

```python
found = h.find_region_fovs(TECHNIQUE_ROOTS, "region_UCI-2424")
region_labels, region_changes = h.scan_region(found, ...)
metrics = h.fov_metrics(region_labels, region_changes)
h.plot_fov_table(metrics, title=...)
```

`find_region_fovs` drops any FOV only one technique produced and prints which —
a ragged comparison is worse than a missing one — and pins technique order to
`TECHNIQUE_ROOTS` rather than the alphabet, so rows pair up the same way in
every region. `scan_region` reads only the mask volume, one at a time: the DAPI
stack is what `load_data` checks against and neither metric needs it, so peak
memory is set by the largest FOV rather than by how many there are.

`plot_fov_table` sets the raw per-FOV numbers beside two group scores:

| | |
|---|---|
| `Thinness` | `1 − (pct_thin + pct_single_slice) / 100` |
| `Stability` | `1 − pct_flagged / 100` |

Both start at 1.00 and pay a point per percent of defect. A single-plane mask
lands in both columns of the thinness score, so it costs twice what a two-plane
one does — one plane is a failed mask outright, two is only what a `min_z = 3`
filter happens to discard. The scores are **absolute, not ranked**: 0.82 means
the same thing in every region, and adding a FOV to the table moves nobody.
Enough defect drives a score below zero, and only the drawn bar clips at 0 — the
printed number does not, because a 0.00 and a −0.40 are not the same run.

### The same region, scored on brightness instead

[`brightness_method_comparison.ipynb`](brightness_method_comparison.ipynb) walks
the same paired FOVs — same `find_region_fovs`, same technique hues — and asks
what the mask geometry cannot: **do the voxels a technique claims actually look
like signal?** Every voxel contributes one `(brightness, is_masked)` pair, so a
technique that over-grows its masks drags background in and its masked
distribution slides left toward the unmasked one.

One figure per FOV, two panels — voxels in a mask, voxels not in one — with the
*techniques* overlaid inside each panel, which is the opposite split from the
single-run figure in
[`segmentation_z_brightness_claude.py`](segmentation_z_brightness_claude.py).
Both techniques segment the same DAPI stack, so the bins are built from it once
and both mask volumes are histogrammed onto them; two distributions binned
differently cannot be laid over each other. `auc` — P(a random masked voxel is
brighter than a random unmasked one) — is the one number that survives the two
groups being wildly different sizes, so it compares techniques and FOVs
directly. Read it next to `frac_masked`: a technique can post a high AUC by
masking only the brightest voxels and skipping most of the nuclei.

### Looking at the flagged masks

`build_span_layers` and `build_change_layer` are label-volume filters: each
returns a copy of the mask volume with everything but the selected labels
zeroed, so napari toggles them as separate label layers over the DAPI. One
viewer per technique over the same DAPI, so the same nucleus can be found under
each. The viewer blocks the kernel until its window closes, which is why that
cell is off by default.

## Scaling further

- **Read size is the main lever.** Every `page[...]` call crosses zarr's
  sync-to-async bridge, costing roughly a millisecond of thread handoff no
  matter how little data comes back. Reading one native chunk at a time makes
  that overhead the whole runtime: a 4000×4000 striped TIFF has 250 sixteen-row
  strips per plane, so a 7-plane volume spends 1,750 round-trips waiting rather
  than reading. The default batches whole chunks up to `target_bytes` (8 MiB),
  which cut that same volume from 1,750 reads to 56 and 2.9 s to 1.3 s. It never
  changes the result, only the speed.

  Measured on 7×4000×4000 uint32 volumes, the curve is a **plateau with cliffs
  on both sides**, not a peak:

  | read size | reads/volume | s/volume | |
  |---|---|---|---|
  | 0.2 MiB (native strip) | 1,750 | 2.93 | sync overhead dominates |
  | 1 MiB | 441 | 1.77 | |
  | 2–32 MiB | 224 → 14 | **1.30–1.45** | flat, all within noise |
  | 64 MiB (whole plane) | 7 | 1.68 | allocation + cache pressure |

  Inside the plateau the differences are smaller than run-to-run variance, so
  there is no single magic number to find — anything from 2 to 32 MiB is fine,
  and only the extremes cost you.

- **Tune it on your own hardware** if you want to confirm. The optimum moves
  with strip height, dtype, CPU cache, and whether bytes come off NVMe or an
  object store:
  ```python
  from zspan import tune_read_size, best_read_size

  table = tune_read_size("region_X/fov_01/masks.tif")   # timings per candidate
  scan_segmentations(root, target_bytes=best_read_size(table))
  ```
  `best_read_size` returns the *middle* of the fast plateau rather than the
  argmin — the argmin wanders between 4, 8 and 16 MiB across identical runs,
  while the plateau midpoint is stable and has margin from both cliffs. Tune
  with the `reader` you will scan with: the block sizes come out the same, but
  the per-read overhead the curve measures does not.
- **Don't raise `target_bytes` past the plateau.** Time is flat from 2 to 32 MiB
  but peak memory tracks the read size linearly — 21 MB at 2 MiB, 29 at 8, 90 at
  32, 165 at 64 — so anything above ~8 MiB spends the one thing streaming buys
  and gets nothing back.
- **Remote data** needs a different store, not different code. This is what
  `reader="virtual"` exists for, and `"auto"` selects it as soon as the path is
  a URL or a registry is supplied:
  ```python
  from obstore.store import S3Store
  from obspec_utils.registry import ObjectStoreRegistry

  registry = ObjectStoreRegistry({"s3://bucket/segs": S3Store("bucket")})
  volume = open_mask_volume("s3://bucket/segs/.../masks.tif", registry)
  ```
- **Persist the manifests** with `virtualizarr`'s Icechunk/Kerchunk writers if
  you rescan often, and later passes skip TIFF header parsing entirely.
- **Don't count on threads.** `max_workers` avoids pickling volumes between
  workers, but it does not scale linearly: zarr funnels every chunk request
  through a single background event loop, so workers contend on it. Measured
  ~1.1× going from 1 to 4 threads on striped TIFFs, against 2.1× from
  `target_bytes` alone. Tune the read size first.

## napari

`to_xarray()` returns a lazy dask-backed `DataArray`, so napari pulls only the
planes it displays:

```python
import napari
viewer = napari.Viewer()
viewer.add_labels(open_mask_volume(path).to_xarray().data)
```
