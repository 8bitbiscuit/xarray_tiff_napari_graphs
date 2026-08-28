"""The whole rubric on one FOV: the scorecard, and a napari layer per line.

Every other script in this directory measures one thing.
``segmentation_z_kanai_only.py`` asks how far a mask reaches through z,
``segmentation_z_jitter.py`` how much its area profile wanders,
``segmentation_z_area_sway.py`` how sharp its worst corner is,
``segmentation_z_brightness.py`` whether the voxels inside a mask outshine the
ones outside it — each with its own figure, its own printed summary, and a
viewer whose extra layers are the masks *that* script flagged.

``segmentation_helpers_v2.py`` turns those same measurements into a rubric::

    Span    1 - pct_thin / 100      masks a min_z filter would discard
    Jitter  1 - pct_jittery / 100   profiles that wander up and down
    Sway    1 - pct_swayed / 100    profiles that jump once and hold (an over-merge)
    Signal  separability            P(a masked voxel outshines an unmasked one)
    Total   the four, weighted, out of 1.00   (h.SCORE_WEIGHTS)

but it scores a *scan*: ``scan_region`` walks every FOV of every technique and
``plot_scorecard`` draws them as a table, which is the comparison notebook's
job and not a thing you can point at one field.  This script is that card for a
single FOV, in the shape of its neighbours — a CONFIG block, ``load_data``,
``analyse_masks``, a summary, a plot, ``launch_viewer``.

Nothing is measured here.  ``h.analyse_fov`` does the span, profile and
brightness work in one pass, ``h.fov_metrics`` and ``h.score_fovs`` turn it into
the card, and ``h.build_flag_layers`` builds the volumes — so this script and
the notebook's scorecard cannot drift apart, which is the same bargain
``segmentation_z_jitter.py`` makes.  What this file adds is the single-FOV
framing: the tidy one-row tables those functions expect, a printed card that
carries the count behind every line, and two layers the notebook has no use for
(below).

What it loads into napari
-------------------------
``h.build_flag_layers`` already returns one label volume per defect; each is
prefixed here with the rubric line that charges for it, so the layer list reads
as the card::

    Span: thin (z-span <= 2)                     the Span line
    Jitter: flagged: jitter (...)                the Jitter line
    Sway: flagged: sway (...)                    the Sway line
    Signal: bright, unmasked (>...)              the two corners of the
    Signal: dim, masked (<...)                   brightness overlap
    deep (z-span >= 6)                           measured, scored by nothing
    z-gapped (hole in z)                         measured, scored by nothing
    dropped: single-slice artifacts              removed before anything is measured

plus two built here: ``unscored``, the masks too short for a profile metric to
exist — they are out of the Jitter and Sway *denominators*, so they are neither
flagged nor clean — and ``clean``, the masks no scored line flags at all.

Between them those layers cover every mask in the field, but they do not
partition it and the overlaps are the point: a thin mask usually has no
measurable profile either, and a mask can be jittery and swayed both.  Which
masks two lines agree on is the question the card cannot answer and the viewer
can, so nothing is flattened into one layer with a flag channel to make the
counts add up.

``import napari`` stays inside ``h.launch_viewer``, so the card and the figure
run on a machine without Qt.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# the helpers sit one directory up.  Running this as
# `python napari_scripts/segmentation_z_scorecard.py` puts only napari_scripts/ on sys.path, so
# the repo root goes on it too -- `pip install -e .` makes the same imports work
# without this, and the line is harmless when it has.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import segmentation_helpers_v2 as h  # noqa: E402

# CONFIG
# ----------------------------------------------------------------------------
MASKS_DIR = Path("../data/additional_annotated_datasets/other/Platynereis-Nuclei-CBG/train/masks")
DAPI_DIR = Path("../data/additional_annotated_datasets/other/Platynereis-Nuclei-CBG/train/images")

MASKS_PATH = MASKS_DIR / "dataset_hdf5_050_0.tif"
DAPI_GLOB = str(DAPI_DIR / "dataset_hdf5_050_0.tif") # Each z index

RUN_LABEL = None           # the card's row label; None -> "<region> · <fov>" off MASKS_DIR

# The rubric's cutoffs.  Absolute, never quantiles of the field they are applied
# to: a score is only comparable between FOVs if the line it is measured against
# means the same thing in both.  The two profile cutoffs are provisional -- set
# them from the knee in the distributions segmentation_z_jitter.py and
# segmentation_z_area_sway.py draw, not from here.
THIN_MAX = h.THIN_MAX          # z_span <= this is what a min_z = 3 filter discards
DEEP_MIN = h.DEEP_MIN          # z_span >= this reaches most of the stack; measured, unscored
JITTER_CUTOFF = h.JITTER_CUTOFF   # total reversal travel, as a factor
SWAY_CUTOFF = h.SWAY_CUTOFF       # worst corner, as a factor
AREA_FLOOR = h.AREA_FLOOR      # px; ratios between handfuls of pixels are noise
SLIVER_FRAC = h.SLIVER_FRAC    # a plane under this share of the mask's own peak is a
                               # partial-volume sliver, left out of the profile
DROP_SINGLE_SLICE = True       # a mask on one z plane is an imaging artifact.  Dropping
                               # it changes the denominator of every rate, not the
                               # numerator, and returns its voxels to the unmasked pool
WEIGHTS = None                 # None -> the rubric's own h.SCORE_WEIGHTS

OVERLAP_QUANTILE = h.OVERLAP_QUANTILE   # the Signal layers' bright/dim thresholds

MAKE_PLOT = True
PLOT_PATH = Path("z_scorecard.png")   # None -> show interactively
CSV_PATH = None            # e.g. Path("z_scorecard.csv") -> "<stem>_metrics.csv" (the
                           # card's own row) and "<stem>_masks.csv" (one row per mask)

LAUNCH_NAPARI = True
Z_SCALE = None             # e.g. 15 -> stretch z so the 3D view is roughly isotropic;
                           # the right number is the z step divided by the pixel size
# ----------------------------------------------------------------------------

# (head of the helper's layer name, the rubric line that charges for it, what to
# call it after the prefix).  Matched on the head because the rest of the name
# carries a threshold; ``None`` keeps the helper's own name, which is where the
# thresholds are.  A layer that is not in here is one nothing scores -- deep
# masks and z-gaps are measured and reported, and score_fovs charges for
# neither, so they keep their names and stay in the viewer.
LINE_OF = (
    ("thin", "Span", None),
    ("flagged: jitter", "Jitter", "wanders up and down"),
    ("flagged: sway", "Sway", "one big jump"),
    ("bright, unmasked", "Signal", None),
    ("dim, masked", "Signal", None),
)


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


def fov_tags(masks_dir=MASKS_DIR, label=RUN_LABEL):
    """The group keys ``fov_metrics`` indexes one row by.

    ``scan_region`` tags every frame with ``region``/``fov``/``method`` and both
    ``fov_metrics`` and ``plot_scorecard`` read the row label off that index, so
    one FOV is tagged the same way rather than given a card of its own shape.
    ``RUN_LABEL`` collapses it to a single level, which is what to use when the
    directory names are not the label you want on the card.
    """
    if label is not None:
        return {"fov": label}
    parts = Path(masks_dir).parts
    return {"region": parts[-2], "fov": parts[-1]} if len(parts) >= 2 else {
        "fov": str(masks_dir)}


def analyse_masks(masks, dapi=None, tags=None, weights=WEIGHTS,
                  drop_single_slice=DROP_SINGLE_SLICE, thin_max=THIN_MAX,
                  deep_min=DEEP_MIN, jitter_cutoff=JITTER_CUTOFF,
                  sway_cutoff=SWAY_CUTOFF, area_floor=AREA_FLOOR,
                  sliver_frac=SLIVER_FRAC):
    """Every measurement, then the card: ``result`` plus ``metrics`` and ``scores``.

    ``h.analyse_fov`` runs the single-slice filter first and hands back one row
    per surviving mask with its span, jitter and sway side by side, the
    brightness statistics, and the filtered and removed volumes.  The three
    tidy frames built here are the one-FOV case of what ``scan_region``
    assembles for a whole scan, so ``fov_metrics`` and ``score_fovs`` see
    exactly the shape they see in the notebook.
    """
    tags = fov_tags() if tags is None else tags
    index = list(tags)

    result = h.analyse_fov(
        masks, dapi=dapi, drop=drop_single_slice,
        jitter_cutoff=jitter_cutoff, sway_cutoff=sway_cutoff,
        area_floor=area_floor, sliver_frac=sliver_frac,
        thin_max=thin_max, deep_min=deep_min)

    labels = result["labels"].assign(**tags)
    filters = pd.DataFrame([{**tags, **result["report"]}]).set_index(index)
    brightness = (pd.DataFrame([{**tags, **result["brightness"].to_dict()}]).set_index(index)
                  if result["brightness"] is not None else None)

    metrics = h.fov_metrics(labels, brightness, filters, by=index)
    scores = h.score_fovs(metrics, weights)

    result["labels"] = labels
    return result, metrics, scores


def give_scorecard_summary(result, metrics, scores, weights=WEIGHTS,
                           thin_max=THIN_MAX, jitter_cutoff=JITTER_CUTOFF,
                           sway_cutoff=SWAY_CUTOFF):
    """The card as text: every line, what it earned, and the count behind it.

    The point of the last column is that a score is a share, and a share says
    nothing about how many masks are behind it — 0.94 off eleven measurable
    masks and 0.94 off fifteen hundred are not the same evidence, and only the
    counts show which one you have.
    """
    labels, report = result["labels"], result["report"]
    row, card = metrics.iloc[0], scores.iloc[0]
    w = pd.Series(h.SCORE_WEIGHTS if weights is None else weights, dtype=float)
    lines = [name for name in h.SCORE_COLUMNS if name != h.TOTAL_COLUMN]

    n_measurable = int(labels["jitter"].notna().sum())
    detail = {
        "Span": (int(labels["thin"].sum()), len(labels),
                 f"masks thin (z-span <= {thin_max:g})"),
        "Jitter": (int(labels["large_jitter"].sum()), n_measurable,
                   f"measurable profiles wander (> {jitter_cutoff:g}x)"),
        "Sway": (int(labels["large_sway"].sum()), n_measurable,
                 f"measurable profiles have a corner (> {sway_cutoff:g}x)"),
    }

    print("\n=== SCORECARD ===")
    if report["n_single_slice"]:
        print(f"  single-slice masks: {report['n_single_slice']} of {report['n_total']} "
              f"({report['pct_dropped']:.1f}%)"
              f"{' -- dropped before measuring' if report['n_dropped'] else ' -- kept'}")
    print(f"  n_objects: {len(labels)}  |  with a measurable profile: {n_measurable}")

    # the weight of the lines that exist, which is what each one's share of the
    # Total is taken over: a FOV scanned without its DAPI has no Signal, and the
    # other three still have to add up to 1.00
    scored = [name for name in lines if name in scores.columns and np.isfinite(card[name])]
    available = float(w[scored].sum()) if scored else float("nan")

    print(f"\n  {'line':<8} {'score':>6} {'weight':>7} {'earned':>7}   behind it")
    for name in lines:
        value = card[name] if name in scores.columns else np.nan
        if name in detail:
            n_flagged, n_of, what = detail[name]
            behind = f"{n_flagged:,} / {n_of:,} {what}" if n_of else f"no masks — {what}"
        else:
            behind = "P(a masked voxel outshines an unmasked one)"

        if np.isfinite(value):
            print(f"  {name:<8} {value:>6.2f} {w[name]:>7.2f} "
                  f"{value * w[name] / available:>7.3f}   {behind}")
        else:
            print(f"  {name:<8} {'–':>6} {w[name]:>7.2f} {'–':>7}   not measured")

    print(f"  {'Total':<8} {card[h.TOTAL_COLUMN]:>6.2f} {available:>7.2f} "
          f"{card[h.TOTAL_COLUMN]:>7.3f}   out of 1.00")

    lost = sorted(((name, (1.0 - card[name]) * w[name] / available) for name in scored),
                  key=lambda pair: -pair[1])
    if lost:
        print("\n  points lost: "
              + ",  ".join(f"{name} {value:.3f}" for name, value in lost))

    # the wide table the card is the narrow reading of -- frac_masked in
    # particular is what says whether a high separability was earned or bought
    # by masking only the brightest voxels
    print("\n  every statistic behind it:")
    print(row.to_frame(name="value").to_string(
        float_format=lambda v: f"{v:,.3f}".rstrip("0").rstrip(".")))


def add_headroom(fig, inches=0.6):
    """Grow the top margin without moving the axes, in inches.

    ``plot_scorecard`` sizes a card as ``0.42 * n_rows + 2.3`` inches and puts
    its title at ``y=0.98``, while the group headers sit a fixed 34 points above
    the axes.  On a scan that is fine; on a **one-row** card the figure is at its
    shortest and those two land on each other.  The axes box is kept where it is,
    in inches, and only the margin above it grows -- so the title clears the
    headers and nothing inside the card moves or rescales.
    """
    width, height = fig.get_size_inches()
    grown = height + inches
    pars = fig.subplotpars
    fig.set_size_inches(width, grown)
    fig.subplots_adjust(top=pars.top * height / grown,
                        bottom=pars.bottom * height / grown)
    return fig


def plot_card(metrics, scores, path=PLOT_PATH, label=None):
    """The notebook's scorecard figure, one row wide.

    ``h.plot_scorecard`` draws the individual statistics on the left and the
    rubric on the right, with each line's weight on its header.  Drawn by the
    same function the notebook calls, so a card for one field and a card for a
    whole scan are the same object read at different widths -- the only thing
    done to it here is :func:`add_headroom`, which is a margin and not a number.
    """
    label = label or run_label()
    fig = h.plot_scorecard(metrics, title=f"Segmentation scorecard — {label}",
                           scores=scores)
    add_headroom(fig)

    if path is None:
        plt.show()   # blocks until closed, napari launches afterwards
    else:
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"\nWrote {path}")
        plt.close(fig)
    return fig


def build_layers(result, quantile=OVERLAP_QUANTILE, thin_max=THIN_MAX,
                 deep_min=DEEP_MIN, verbose=True):
    """``h.build_flag_layers``, keyed by rubric line, plus ``unscored`` and ``clean``.

    The volumes are the helper's — this only renames them, so a layer in the
    viewer is the same array the notebook would have shown, and adds the two the
    notebook has no use for: the masks too short for a profile metric to exist
    (out of the Jitter and Sway denominators, so neither flagged nor clean) and
    the masks no scored line flags.  ``clean`` means no *scored* line: a deep or
    z-gapped mask can be in it, because ``score_fovs`` charges for neither.
    """
    labels, masks = result["labels"], result["masks"]

    layers = h.build_flag_layers(
        masks, labels, dropped=result.get("dropped"), dapi=result.get("dapi"),
        counts=result["counts"], edges=result["edges"], quantile=quantile,
        thin_max=thin_max, deep_min=deep_min)

    def renamed(name):
        for head, line, short in LINE_OF:
            if name.startswith(head):
                return f"{line}: {short or name}"
        return name          # measured but scored by nothing; keep its own name

    named = {renamed(name): volume for name, volume in layers.items()}

    def subset(condition):
        flagged = labels.loc[condition, "label"].to_numpy()
        return np.where(np.isin(masks, flagged), masks, 0).astype(masks.dtype)

    unmeasurable = labels["jitter"].isna()
    flagged_by_a_line = labels["thin"] | labels["large_jitter"] | labels["large_sway"]
    named["unscored: profile too short to measure"] = subset(unmeasurable)
    named["clean: no scored line flags it"] = subset(~flagged_by_a_line & ~unmeasurable)

    if verbose:
        # voxels rather than masks, because two of these are boolean volumes and
        # not objects at all; the per-line mask counts are in the summary above
        print("\n  napari layers:")
        for name, volume in named.items():
            print(f"  {name:<46} {int(np.count_nonzero(volume)):>12,} voxels")

    return named


def launch_viewer(result, layers=None, label=None, z_scale=Z_SCALE):
    """DAPI, the scored masks, and every line's layer on top of them, all hidden.

    The volume the viewer shows is ``result["masks"]`` — the one the card was
    computed from, with the single-slice artifacts already out of it — so what
    you look at is what was scored.
    """
    return h.launch_viewer(result["dapi"], result["masks"], layers,
                           title=label or run_label(), z_scale=z_scale)


def write_tables(metrics, labels, path=CSV_PATH):
    """``<stem>_metrics.csv`` (the card's row) and ``<stem>_masks.csv`` (one per mask)."""
    if path is None:
        return
    path = Path(path)
    suffix = path.suffix or ".csv"

    for name, table, index in (("metrics", metrics, True), ("masks", labels, False)):
        out = path.with_name(f"{path.stem}_{name}{suffix}")
        table.to_csv(out, index=index)
        print(f"Saved {name} table to {out}")


def main():
    # every CONFIG value is passed explicitly rather than left to the signature
    # defaults, which bind at import: this way setting e.g. ``JITTER_CUTOFF`` on
    # the imported module takes effect, instead of being silently ignored
    h.apply_theme()
    pd.set_option("display.width", 200)

    dapi, masks, files = load_data(DAPI_GLOB, MASKS_PATH)
    tags = fov_tags(MASKS_DIR, RUN_LABEL)
    label = RUN_LABEL or run_label(MASKS_DIR)

    result, metrics, scores = analyse_masks(
        masks, dapi=dapi, tags=tags, weights=WEIGHTS,
        drop_single_slice=DROP_SINGLE_SLICE, thin_max=THIN_MAX, deep_min=DEEP_MIN,
        jitter_cutoff=JITTER_CUTOFF, sway_cutoff=SWAY_CUTOFF,
        area_floor=AREA_FLOOR, sliver_frac=SLIVER_FRAC)
    result["dapi"] = dapi

    give_scorecard_summary(result, metrics, scores, weights=WEIGHTS,
                           thin_max=THIN_MAX, jitter_cutoff=JITTER_CUTOFF,
                           sway_cutoff=SWAY_CUTOFF)
    write_tables(metrics, result["labels"], CSV_PATH)

    if MAKE_PLOT:
        plot_card(metrics, scores, path=PLOT_PATH, label=label)

    if LAUNCH_NAPARI:
        layers = build_layers(result, quantile=OVERLAP_QUANTILE,
                              thin_max=THIN_MAX, deep_min=DEEP_MIN)
        launch_viewer(result, layers, label=label, z_scale=Z_SCALE)

    return {"result": result, "metrics": metrics, "scores": scores}


if __name__ == "__main__":
    main()
