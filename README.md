# Scoring segmentation runs: z-span, area profile, brightness

Cellpose-style segmenters can work a stack two ways — natively in 3D (`do_3D`),
or per z-plane with the 2D masks stitched afterwards (`stitch_threshold`). This
repo scores both on the same field and puts the answer on one card, reading the
TIFFs lazily from a local disk or straight out of S3.

Three measurements, all read off the same volumes:

| | question | what a bad answer looks like |
|---|---|---|
| **z-span** | how far does each mask reach through z? | a pile of masks on one or two planes (yield a `min_z` filter discards), or a tail reaching through the whole stack (two nuclei read as one) |
| **area profile** | does the area-through-z profile wander, or jump? | a profile that turns around plane after plane, or a corner: a flat run, a jump, another flat run |
| **brightness** | do the voxels a technique claims look like signal? | a masked brightness distribution sliding left toward the unmasked one |

```python
import segmentation_helpers_v2 as v2

scan = v2.scan_regions(TECHNIQUE_ROOTS, DAPI_PATCHES)
metrics = v2.fov_metrics(scan["labels"], scan["brightness"], scan["filters"])
v2.plot_scorecard(metrics.droplevel("region"))
```

Two worked notebooks, same code, different source of bytes:

| | |
|---|---|
| [`segmentation_comparison_v2.ipynb`](segmentation_comparison_v2.ipynb) | reads a local `../data` tree |
| [`segmentation_comparison_s3.ipynb`](segmentation_comparison_s3.ipynb) | reads an S3 bucket through VirtualiZarr |

They should print identical numbers from the same data — that is the check that
the loading layer changed nothing.

## Install

```bash
pip install -e .            # helpers + the virtual-zarr read path
pip install -e ".[dev]"     # + ipykernel, to run the notebooks
pip install -e ".[viewer]"  # + napari, only for the viewers
```

The editable install is what lets `import segmentation_helpers_v2` work from a
kernel started anywhere; without it, run from the repo root. **napari is never
imported at module scope** except in `napari_scripts/segmentation_z_kanai_only.py`,
so a headless kernel with no Qt runs everything else — every viewer function
imports it inside the call.

Point `TECHNIQUE_ROOTS` and `DAPI_PATCHES` at your own output. Nothing here
ships data.

## What is in the repo

Two kinds of file, and the line between them is the one structural rule:
**a module never imports a script; a script calls into a module.**

### Modules — the library

| | |
|---|---|
| [`segmentation_loading.py`](segmentation_loading.py) | lazy, chunk-aware TIFF access, local or `s3://`. Imports nothing else here |
| [`segmentation_helpers_v2.py`](segmentation_helpers_v2.py) | every measurement, the region walk, the scorecard, the paired comparison, the napari layers. Imports only `segmentation_loading` |
| [`segmentation_brightness_helpers.py`](segmentation_brightness_helpers.py) | two mask volumes over the same image, on the same bins, in the same panel |

`segmentation_helpers_v2.py` is where the measurements live —
`check_area_jitter`, `check_area_sway` and their flags in Part 2, the histogram
machinery (`vectorise_pixels`, `build_bin_edges`, `accumulate_histograms` and
the statistics read off the counts) in Part 3.

### Scripts — one question, one field, a viewer at the end

Each has a CONFIG block at the top, a `main()`, and a napari launch it can be
told to skip. Set `LAUNCH_NAPARI = False` and they run headless. Each puts the
repo root on `sys.path` itself, so `python napari_scripts/whichever.py` works
from a fresh shell whether or not the package is installed.

| | |
|---|---|
| [`segmentation_z_kanai_only.py`](napari_scripts/segmentation_z_kanai_only.py) | z-span, the original. `import napari` at module scope, so it needs Qt to import at all |
| [`segmentation_z_jitter.py`](napari_scripts/segmentation_z_jitter.py) | does the area profile wander up and down |
| [`segmentation_z_area_sway.py`](napari_scripts/segmentation_z_area_sway.py) | both profile metrics, with the worst drawn over a sample of quiet ones |
| [`segmentation_z_brightness.py`](napari_scripts/segmentation_z_brightness.py) | brightness inside vs outside the masks, for one run, pooled and per z |
| [`segmentation_z_log_cutoff.py`](napari_scripts/segmentation_z_log_cutoff.py) | z-span plus per-transition \|Δ area\| / area, flagged by an absolute cutoff on the log axis |
| [`segmentation_z_brightness_bimodal.py`](napari_scripts/segmentation_z_brightness_bimodal.py) | the notebook's two-technique brightness figure with an absolute cutoff drawn across it |

**The scripts do not carry their own copies of the measurements.** Every one of
them calls into `segmentation_helpers_v2`, so a script and the notebook's
scorecard cannot drift apart. The dependency runs one way only: a script may
import the library, the library never imports a script — a library that imported
a script would inherit that script's CONFIG globals, and editing a CONFIG block
would silently move the notebook's numbers.

Where a script and the module ship *different* defaults, the script wraps the
call rather than re-exporting it. `flag_sway` is the case: the module's
`SWAY_CUTOFF` is an absolute 3.0, the sway script's is `None`, meaning "flag the
top decile of this FOV". A bare re-export would have swapped one for the other
without a word.

## Reading the data

`segmentation_loading` opens each multi-page TIFF as a lazy zarr array whose
chunks are the file's own tiles. Nothing is copied or converted; indexing
fetches only the tiles the window covers.

Two backends do that, chosen with `reader=`:

| | `"tifffile"` | `"virtual"` |
|---|---|---|
| how | `tifffile`'s `aszarr` store | `virtualizarr` + `virtual_tiff` IFD manifest |
| open, 7×4000×4000 striped | ~1 ms | 325 ms |
| read, that volume @ 8 MiB blocks | 1.07 s | 1.35 s |
| peak memory | 29 MB | 29 MB |
| reads `s3://` via `obstore` | no | **yes** |
| manifest persists (Icechunk/Kerchunk) | no | **yes** |

`reader="auto"` (the default) routes on the path: a plain local file goes to
`tifffile`, anything with a non-`file` URL scheme — or any call given a registry
— goes to `virtual`. So the same code moves to object storage without an edit,
and pays nothing for the manifest until it does. The two agree exactly: same
`chunks`, same blocks, identical statistics.

One registry backs a whole scan; build it once and pass it as `registry=`:

```python
import segmentation_loading as sl

registry, DATA_ROOT = sl.make_registry("s3://bucket/prefix", region="us-west-2")
scan = v2.scan_regions(TECHNIQUE_ROOTS, DAPI_PATCHES, registry=registry)
```

`sl.local_registry(path)` does the same for a local tree, which is how the S3
notebook is exercised without a bucket. Join URL pieces with `sl.join_url` and
never with `Path`: `Path("s3://b") / "x"` collapses the double slash into
`s3:/b/x` and the URL stops pointing at the bucket.

**Read size is the main lever**, and it matters most over a network, where every
chunk is its own HTTP request and a striped plane can cost hundreds of round
trips. `sl.DEFAULT_READ_BYTES` batches whole strips up to a budget; 4–32 MiB all
measure the same and 8 MiB is the low-variance middle. It never changes a value,
only the wait.

## Data layout

```
<root>/segmentations_3d_true/<family>/<preprocessing>/<model>/<region>/<fov>/masks.tif
<root>/segmentations_3d_stitched/<family>/<preprocessing>/<model>/<region>/<fov>/masks.tif
<root>/patches/<model>/<region>/<fov>/DAPI_decon_z*.tif
```

Masks are one multi-page TIFF per FOV, one IFD per z-plane; DAPI is one
single-page TIFF *per* plane, so a directory of them is stacked.
`TECHNIQUE_ROOTS` maps a technique name to the directory *above* the region, and
`DAPI_PATCHES` is the matching patches directory — the region name is appended
wherever it is needed, so it is named once.

`find_region_fovs` drops any FOV only one technique produced and prints which — a
ragged comparison is worse than a missing one — and pins technique order to
`TECHNIQUE_ROOTS` rather than the alphabet, so rows pair up the same way in every
region. One DAPI stack is read per FOV and both mask volumes are histogrammed
onto bins built from it: two distributions binned differently cannot be laid over
each other. Only one stack and one mask volume are resident at a time, so peak
memory is set by the largest FOV rather than by how many there are.

## The measurements

### Single-slice masks are removed before anything is measured

A mask living on exactly one z plane is an imaging artifact, not a nucleus. It is
dropped from the volume **first**, so it stops counting as a mask *and* its
voxels go back into the unmasked brightness pool — the filter has to happen
upstream of every metric or the brightness figures would still be scoring it.

Per-plane stitching is what produces them, so `DROP_SINGLE_SLICE_IN` can filter
only that arm; the default filters **both**. Removing masks from one arm and not
the other moves every rate in that arm's favour, because the masks that survive
are the thicker ones — its thin rate, its median z-span and its jitter rate all
improve for free, and none of that improvement is a segmenter doing better. The
counts removed are printed either way and carried as `n_dropped` / `pct_dropped`:
reported, never scored, because an artifact of the imaging is not a defect of the
segmenter.

### z-span

`z_span` is the inclusive depth of a label's z bounding box; `n_planes` is how
many planes it actually occupies. They differ only when the label has a hole in
z, so **`n_planes < z_span` flags a probable merge of two objects stacked in z** —
a harder error than plain over-merging. Both are read off the per-plane areas
rather than from a second `find_objects` pass, since the areas are what every
other measurement needs anyway.

### Jitter and sway — two defects, two statistics

Both are dimensionless factors starting at 1.0, so an absolute cutoff means the
same thing in every FOV.

```
jitter = 10 ** Σ min(|d[k]|, |d[k+1]|)  over reversals of  d[k] = log10 a[k+1] − log10 a[k]
sway   = 10 ** max |log10 a[z+1] − 2 log10 a[z] + log10 a[z−1]|
```

`jitter` asks **how often, and how far, does the profile turn around** — a
monotone profile scores exactly 1.0 at any steepness, because each reversal is
weighted by the *smaller* of its two limbs, and summing rather than maxing lets
many small wobbles outscore one big excursion. `sway` asks **how sharp is the
worst corner**: the flat-run/jump/flat-run of an over-merge is monotone, so it
reverses nowhere and scores a clean 1.00 jitter, which is why both are measured.

A plane holding under `AREA_FLOOR` px, or under `SLIVER_FRAC` of the mask's own
peak area, is a partial-volume sliver and is left out of the profile — every
nucleus enters and leaves the stack through one, and a sliver against a mid-plane
is a ratio of hundreds to one that says where the nucleus met the edge of the
volume, not how it was segmented. Both metrics need three consecutive surviving
planes and are `NaN` without them, so a mask too short to measure is never
flagged and never counted in the denominator.

**`JITTER_CUTOFF = 1.75` and `SWAY_CUTOFF = 3.0` are provisional** — worked
profiles, not real data. `compare_profile_distribution` draws the pooled
distribution; set them from the knee where the ordinary bulk ends.

### Brightness

Every voxel contributes one `(brightness, is_masked)` pair, histogrammed per z
plane so peak memory is one slice. `separability` — P(a random masked voxel is
brighter than a random unmasked one) — is the one number that survives the two
groups being wildly different sizes. 0.5 says brightness carries no information
about what the technique masked. Read it next to `frac_masked`: a technique can
post a high separability by masking only the brightest voxels and skipping most
of the nuclei.

### Nothing assumes a normal distribution

A jitter is a spike at 1.0x with a sparse tail past 100, z-span is a small
integer count, and voxel brightness is bimodal by construction. So centre and
spread are the median and the IQR; the cutoffs are absolute rather than quantiles
of the FOV they are applied to (a quantile cutoff flags the top 10% of every FOV
*by construction*, which cannot compare two techniques); and comparing two
techniques is paired **Wilcoxon signed-rank** with the Hodges–Lehmann shift, the
matched-pairs rank-biserial correlation and a percentile bootstrap CI. The means
the scripts print are still computed and still shown, and nothing is scored on
them.

## The scorecard

`score_fovs` turns the per-FOV statistics into four absolute subscores and the
weighted total they earn:

| score | | pays for |
|---|---|---|
| `Span` | `1 − pct_thin / 100` | masks too short to keep |
| `Jitter` | `1 − pct_jittery / 100` | masks whose area profile wanders up and down |
| `Sway` | `1 − pct_swayed / 100` | masks whose profile jumps once and holds |
| `Signal` | `separability` | masked voxels that do not outshine unmasked ones |
| `Total` | the four, weighted | out of 1.00 |

The weights live in one constant, `SCORE_WEIGHTS`, and sum to 1, so a clean run
earns 1.00 and every point lost is traceable to the line that lost it. They are
unequal on purpose: a corner in an area profile is a segmenter gluing two objects
together, a wrong answer nothing downstream repairs, while a thin mask is deleted
by a `min_z` filter in one line. An equal split would have been a weighting too —
this one just says what it is. Change them in that constant, keep them summing to
1.00, and say so when you do.

Every subscore is **absolute**: 0.82 means the same thing in every region, and
adding a FOV to the table moves nobody. Enough defect drives a rate score below
zero, and only the drawn bar clips at 0 — a 0.00 and a −0.40 are not the same
run. `Signal` is the odd one: an AUC's no-information point is 0.5, not 0, so it
banks half its weight for any technique whose masks are not actively
anti-correlated with brightness, and the range worth reading on it is about 0.8
to 1.0. Read a `Total` against other `Total`s rather than as a percentage earned.

A missing line renormalises over the weights that remain — a scan without DAPI
has no `Signal`, so its `Total` is the other three over the weight they carry
between them. It is still a number out of 1.00, but it is a shorter rubric, so do
not read it against a card that has all four.

## Colour

`CATEGORICAL`, `SEQUENTIAL_BLUE` and the theme live in
`segmentation_helpers_v2.py` and are what every figure in the repo uses.
Technique owns the hue channel throughout, so a colour never changes meaning
between figures, and identity also travels on line style and marker shape rather
than resting on hue alone. Slots are assigned in fixed order and never generated:
adjacent categorical pairs clear colour-vision-deficiency separation thresholds,
which only holds if the order is kept.

## Looking at the flagged masks

`build_flag_layers` returns one label volume per defect — a copy of the mask
volume with everything but the selected labels zeroed — so napari toggles them
over the DAPI without recomputing anything. The single-slice masks the filter
removed come back as their own layer, worth a look at least once to confirm they
really are artifacts in your data. The two brightness layers are boolean volumes
instead: "outside every mask yet brighter than the median masked voxel" is not a
label.

```python
layers = v2.build_flag_layers(run["masks"], run["labels"], run["dropped"],
                              dapi=dapi, counts=run["counts"], edges=run["edges"])
v2.launch_viewer(dapi, run["masks"], layers, title="region · fov — 3D true")
```

One viewer per technique over the same DAPI, so the same nucleus can be found
under each. The viewer blocks the kernel until its window closes, which is why
the notebooks keep it in a cell of its own and off by default.
