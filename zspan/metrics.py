"""Per-label z-extent statistics, computed one tile at a time.

``scipy.ndimage.find_objects`` needs the whole label volume resident, which is
the one thing that does not scale.  The observation that removes the constraint:
a label's z bounding box is just ``min``/``max`` of the planes it appears on,
and min/max are associative.  So the extent can be accumulated plane by plane,
tile by tile, in any order, with peak memory of one tile plus a couple of arrays
indexed by label.

Results are identical to ``find_objects`` for the ``z_start``/``z_end``/``z_span``
columns -- see ``tests/test_metrics.py``, which asserts it against the original
implementation.

:func:`label_plane_areas` keeps the same shape of accumulation but stops one
step earlier, holding each label's area on each plane rather than collapsing it
to an extent -- which is what :func:`size_change_between_layers` differences to
say how a mask grows or shrinks from one layer to the next.  Both stream; the
areas are exact, not sampled.
"""

from __future__ import annotations

from typing import Any, Iterator, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from .loading import DEFAULT_READ_BYTES, MaskVolume

__all__ = [
    "AREA_COLUMNS",
    "SIZE_CHANGE_COLUMNS",
    "Z_SPAN_COLUMNS",
    "check_z_span",
    "label_plane_areas",
    "section_bounds",
    "size_change_between_layers",
    "summarise_z_spans",
]

Z_SPAN_COLUMNS = (
    "label",
    "z_start",
    "z_end",
    "z_span",
    "n_planes",
    "n_voxels",
    "spans_multiple_layers",
)

AREA_COLUMNS = ("label", "z", "section", "area")

SIZE_CHANGE_COLUMNS = (
    "z_from",
    "z_to",
    "z_gap",
    "area_from",
    "area_to",
    "delta",
    "pct_change",
)


# --------------------------------------------------------------------------- #
# plane sources -- adapt whatever the caller passes to "give me tiles of plane z"
# --------------------------------------------------------------------------- #


@runtime_checkable
class PlaneSource(Protocol):
    """Minimal interface the accumulator needs from an input volume."""

    @property
    def n_planes(self) -> int: ...

    def iter_blocks(
        self, z: int, block_shape: tuple[int, int] | None
    ) -> Iterator[np.ndarray]: ...


class _VolumePlaneSource:
    """Chunk-aligned blocks straight from a lazy :class:`MaskVolume`."""

    def __init__(self, volume: MaskVolume, target_bytes: int = DEFAULT_READ_BYTES) -> None:
        self._volume = volume
        self._target_bytes = target_bytes

    @property
    def n_planes(self) -> int:
        return self._volume.shape[0]

    @property
    def plane_shape(self) -> tuple[int, int]:
        return self._volume.shape[1:]

    def iter_windows(
        self, z: int, block_shape: tuple[int, int] | None
    ) -> Iterator[tuple[tuple[int, int, int, int], np.ndarray]]:
        return self._volume.iter_plane_windows(
            z, block_shape, target_bytes=self._target_bytes
        )

    def iter_blocks(
        self, z: int, block_shape: tuple[int, int] | None
    ) -> Iterator[np.ndarray]:
        for _, block in self.iter_windows(z, block_shape):
            yield block


class _ArrayPlaneSource:
    """Tiles from anything sliceable: numpy, dask, or ``xarray.DataArray``.

    Each block is materialised with ``np.asarray``, so a dask-backed array
    computes one block at a time and stays lazy overall.
    """

    def __init__(self, array: Any) -> None:
        # Unwrap xarray only.  A bare ndarray also has ``.data`` -- a memoryview
        # that does not support tuple indexing -- so test for ``dims`` too.
        data = array.data if hasattr(array, "dims") and hasattr(array, "data") else array
        if getattr(data, "ndim", None) != 3:
            raise ValueError(
                f"expected a 3D (z, y, x) label volume, got ndim="
                f"{getattr(data, 'ndim', None)!r}"
            )
        self._array = data

    @property
    def n_planes(self) -> int:
        return int(self._array.shape[0])

    @property
    def plane_shape(self) -> tuple[int, int]:
        return (int(self._array.shape[1]), int(self._array.shape[2]))

    def iter_windows(
        self, z: int, block_shape: tuple[int, int] | None
    ) -> Iterator[tuple[tuple[int, int, int, int], np.ndarray]]:
        ny, nx = self.plane_shape
        by, bx = block_shape if block_shape is not None else (ny, nx)
        for y0 in range(0, ny, by):
            y1 = min(y0 + by, ny)
            for x0 in range(0, nx, bx):
                x1 = min(x0 + bx, nx)
                yield (y0, y1, x0, x1), np.asarray(self._array[z, y0:y1, x0:x1])

    def iter_blocks(
        self, z: int, block_shape: tuple[int, int] | None
    ) -> Iterator[np.ndarray]:
        for _, block in self.iter_windows(z, block_shape):
            yield block


def _as_plane_source(masks: Any, target_bytes: int = DEFAULT_READ_BYTES) -> PlaneSource:
    if isinstance(masks, MaskVolume):
        return _VolumePlaneSource(masks, target_bytes)
    if isinstance(masks, PlaneSource):
        return masks
    return _ArrayPlaneSource(masks)


# --------------------------------------------------------------------------- #
# streaming accumulator
# --------------------------------------------------------------------------- #


def _block_labels(block: np.ndarray, background: int) -> tuple[np.ndarray, np.ndarray]:
    """Unique labels and voxel counts in one tile, background dropped."""
    labels, counts = np.unique(block, return_counts=True)
    if labels.size and background is not None:
        keep = labels != background
        if not keep.all():
            labels, counts = labels[keep], counts[keep]
    return labels, counts


def _merge_block_labels(
    parts: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Combine per-tile results into one deduplicated set for the plane.

    A label straddling a tile boundary shows up in several tiles; merging here
    keeps ``n_planes`` an honest count of distinct planes rather than tiles.
    """
    if not parts:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    if len(parts) == 1:
        labels, counts = parts[0]
        return labels.astype(np.int64, copy=False), counts.astype(np.int64, copy=False)

    labels = np.concatenate([p[0] for p in parts]).astype(np.int64, copy=False)
    counts = np.concatenate([p[1] for p in parts]).astype(np.int64, copy=False)
    unique, inverse = np.unique(labels, return_inverse=True)
    return unique, np.bincount(inverse, weights=counts).astype(np.int64)


class _ZExtentAccumulator:
    """Label-indexed arrays holding the running z extent.

    Memory is ``O(max_label)``, not ``O(voxels)`` -- a few tens of MB even for
    millions of labels, and independent of image size.
    """

    __slots__ = ("_first", "_last", "_planes", "_voxels")

    def __init__(self, initial_capacity: int = 4096) -> None:
        self._first = np.full(initial_capacity, -1, np.int64)
        self._last = np.full(initial_capacity, -1, np.int64)
        self._planes = np.zeros(initial_capacity, np.int64)
        self._voxels = np.zeros(initial_capacity, np.int64)

    def _grow(self, max_label: int) -> None:
        size = self._first.size
        if max_label < size:
            return
        new_size = max(max_label + 1, size * 2)
        pad = new_size - size
        self._first = np.concatenate([self._first, np.full(pad, -1, np.int64)])
        self._last = np.concatenate([self._last, np.full(pad, -1, np.int64)])
        self._planes = np.concatenate([self._planes, np.zeros(pad, np.int64)])
        self._voxels = np.concatenate([self._voxels, np.zeros(pad, np.int64)])

    def update_plane(self, z: int, labels: np.ndarray, counts: np.ndarray) -> None:
        """Fold one plane's labels in.  Planes must arrive in increasing z."""
        if labels.size == 0:
            return
        if labels.min() < 0:
            raise ValueError("negative label values are not supported")
        self._grow(int(labels.max()))

        # labels are unique here, so fancy-index assignment has no collisions
        unseen = labels[self._first[labels] < 0]
        self._first[unseen] = z
        self._last[labels] = z  # monotonic z => last write wins
        self._planes[labels] += 1
        self._voxels[labels] += counts

    def to_frame(self, layer_span_cutoff: int) -> pd.DataFrame:
        present = np.flatnonzero(self._first >= 0)
        z_start = self._first[present]
        z_end = self._last[present]
        z_span = z_end - z_start + 1  # inclusive, matches find_objects' stop-start
        return pd.DataFrame(
            {
                "label": present.astype(np.int64),
                "z_start": z_start,
                "z_end": z_end,
                "z_span": z_span,
                "n_planes": self._planes[present],
                "n_voxels": self._voxels[present],
                "spans_multiple_layers": z_span > layer_span_cutoff,
            }
        )


def check_z_span(
    mask_volume: Any,
    layer_span_cutoff: int = 1,
    *,
    background: int | None = 0,
    block_shape: tuple[int, int] | None = None,
    target_bytes: int = DEFAULT_READ_BYTES,
) -> pd.DataFrame:
    """Per-label z extent for a 3D label volume.

    Parameters
    ----------
    mask_volume
        A :class:`~zspan.loading.MaskVolume` (streamed from disk, recommended),
        or any 3D ``(z, y, x)`` array: numpy, dask, or ``xarray.DataArray``.
    layer_span_cutoff
        A label counts as spanning multiple layers when ``z_span`` exceeds this.
    background
        Label treated as background and excluded.  ``None`` keeps everything.
    block_shape
        ``(y, x)`` region to read at a time.  Defaults to a chunk-aligned block
        of about ``target_bytes`` for a :class:`MaskVolume`, and to the full
        plane otherwise.  Reading one native chunk at a time is far slower --
        see :meth:`~zspan.loading.MaskVolume.read_block_shape`.
    target_bytes
        Byte budget for a single read, when ``block_shape`` is not given.

    Returns
    -------
    pandas.DataFrame
        One row per label present in the volume, with columns
        ``label, z_start, z_end, z_span, n_planes, n_voxels,
        spans_multiple_layers``.

        ``z_span`` is the inclusive bounding-box depth; ``n_planes`` is how many
        planes the label actually occupies.  They differ only when a label has a
        z gap, so ``n_planes < z_span`` flags a probable merge of two objects
        stacked in z.
    """
    source = _as_plane_source(mask_volume, target_bytes)
    accumulator = _ZExtentAccumulator()

    for z in range(source.n_planes):
        parts = [
            _block_labels(block, background)
            for block in source.iter_blocks(z, block_shape)
        ]
        labels, counts = _merge_block_labels(parts)
        accumulator.update_plane(z, labels, counts)

    return accumulator.to_frame(layer_span_cutoff)


# --------------------------------------------------------------------------- #
# per-plane area, and how it changes from one layer to the next
# --------------------------------------------------------------------------- #


def section_bounds(ny: int, n_sections: int) -> list[tuple[int, int]]:
    """Split ``ny`` rows into ``n_sections`` horizontal bands, top first.

    ``n_sections=4`` is three equally spaced dividers across the image.  The
    bands are half-open ``[y0, y1)`` and cover the plane exactly; when ``ny``
    does not divide evenly, the leftover rows are spread across the bands rather
    than dumped on the last one, so no band is more than a row off the rest.
    """
    if n_sections < 1:
        raise ValueError(f"n_sections must be at least 1, got {n_sections}")
    if n_sections > ny:
        raise ValueError(
            f"cannot split {ny} rows into {n_sections} sections -- some would be empty"
        )
    edges = np.linspace(0, ny, n_sections + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])]


def label_plane_areas(
    mask_volume: Any,
    *,
    background: int | None = 0,
    n_sections: int = 1,
    block_shape: tuple[int, int] | None = None,
    target_bytes: int = DEFAULT_READ_BYTES,
) -> pd.DataFrame:
    """Pixel area of every label on every z-plane it appears on.

    This is the per-plane counterpart to :func:`check_z_span`: instead of
    collapsing a label to its z extent, it keeps one number per plane, which is
    what a *change in size between layers* is computed from (see
    :func:`size_change_between_layers`).

    It streams the same way -- one chunk-aligned block at a time, per-plane
    results merged across blocks so a label straddling a tile boundary is
    counted once.  Peak memory is a block plus the labels present on one plane,
    independent of image size.

    Parameters
    ----------
    mask_volume
        A :class:`~zspan.loading.MaskVolume`, or any 3D ``(z, y, x)`` array:
        numpy, dask, or ``xarray.DataArray``.
    background
        Label treated as background and excluded.  ``None`` keeps everything.
    n_sections
        Split each plane into this many horizontal bands (see
        :func:`section_bounds`) and report an area per band, so the profile can
        be read for one part of the field rather than the whole of it.  A label
        crossing a band boundary contributes its area to each band it covers, so
        summing ``area`` over ``section`` reproduces the whole-plane area
        exactly -- one scan serves both readings.
    block_shape, target_bytes
        Read geometry, as in :func:`check_z_span`.

    Returns
    -------
    pandas.DataFrame
        Columns ``label, z, section, area`` -- one row per label per plane per
        band it is present on, so a label missing from a plane simply has no row
        there.  ``section`` is 0 for the topmost band and runs down the image;
        it is all zeros when ``n_sections=1``.
    """
    source = _as_plane_source(mask_volume, target_bytes)

    if n_sections > 1:
        if not hasattr(source, "plane_shape") or not hasattr(source, "iter_windows"):
            raise TypeError(
                "n_sections > 1 needs a MaskVolume or an array-like volume; the "
                f"plane source {type(source).__name__} reports no geometry."
            )
        bounds = section_bounds(source.plane_shape[0], n_sections)
    else:
        bounds = None  # whole plane: every block is its own section-0 slice

    labels_out: list[np.ndarray] = []
    counts_out: list[np.ndarray] = []
    z_out: list[np.ndarray] = []
    section_out: list[np.ndarray] = []

    for z in range(source.n_planes):
        parts: list[list[tuple[np.ndarray, np.ndarray]]] = [
            [] for _ in range(n_sections)
        ]
        if bounds is None:
            for block in source.iter_blocks(z, block_shape):
                parts[0].append(_block_labels(block, background))
        else:
            for (y0, y1, _, _), block in source.iter_windows(z, block_shape):
                for section, (s0, s1) in enumerate(bounds):
                    top, bottom = max(y0, s0), min(y1, s1)
                    if top >= bottom:
                        continue
                    # A block is a contiguous row range, so its intersection
                    # with a band is one slice -- no copy when it is the whole
                    # block, which is the common case for small enough bands.
                    sub = (
                        block
                        if (top == y0 and bottom == y1)
                        else block[top - y0 : bottom - y0]
                    )
                    parts[section].append(_block_labels(sub, background))

        for section, section_parts in enumerate(parts):
            labels, counts = _merge_block_labels(section_parts)
            if labels.size == 0:
                continue
            labels_out.append(labels)
            counts_out.append(counts)
            z_out.append(np.full(labels.size, z, np.int64))
            section_out.append(np.full(labels.size, section, np.int16))

    if not labels_out:
        return pd.DataFrame(
            {
                "label": np.empty(0, np.int64),
                "z": np.empty(0, np.int64),
                "section": np.empty(0, np.int16),
                "area": np.empty(0, np.int64),
            }
        )

    return pd.DataFrame(
        {
            "label": np.concatenate(labels_out),
            "z": np.concatenate(z_out),
            "section": np.concatenate(section_out),
            "area": np.concatenate(counts_out),
        }
    )


def size_change_between_layers(
    areas: pd.DataFrame,
    *,
    group_cols: Sequence[str] | None = None,
    adjacent_only: bool = True,
) -> pd.DataFrame:
    """How much each mask's area changes from one z-layer to the next.

    Takes the output of :func:`label_plane_areas` -- possibly with extra columns
    identifying which volume each row came from -- and differences it along z
    within each mask.

    Only *consecutive observations of the same mask* produce a row: a mask's
    first plane has nothing to change from, and the plane after its last is not
    a shrink to zero but an absence.  Appearing and disappearing would
    contribute equal and opposite deltas to every mask anyway, so counting them
    adds noise and no signal; what is left measures how a mask's cross-section
    evolves through its own body.

    Parameters
    ----------
    areas
        Frame with ``z`` and ``area`` columns and one row per group per plane.
    group_cols
        What identifies one mask.  Defaults to every column except ``z`` and
        ``area``, which is what makes a concatenated scan work unchanged -- the
        volume and section columns come along on their own.  Must include
        ``label``.
    adjacent_only
        Keep only steps between neighbouring planes.  A mask with a hole in z
        (``n_planes < z_span`` in :func:`check_z_span`) is usually two objects
        stitched into one, and the jump across the hole is not a size change of
        anything; ``False`` keeps those steps, flagged by ``z_gap > 1``.

    Returns
    -------
    pandas.DataFrame
        The group columns plus ``z_from, z_to, z_gap, area_from, area_to,
        delta, pct_change``.  ``delta`` is signed, in pixels; ``pct_change`` is
        relative to ``area_from``, so it is bounded below by -100% but unbounded
        above -- prefer its median, or use ``delta``, when averaging.
    """
    if group_cols is None:
        group_cols = [c for c in areas.columns if c not in ("z", "area")]
    group_cols = list(group_cols)
    if "label" not in group_cols:
        raise ValueError(f"group_cols must include 'label', got {group_cols}")
    missing = [c for c in [*group_cols, "z", "area"] if c not in areas.columns]
    if missing:
        raise KeyError(f"columns missing from the areas frame: {missing}")

    out_columns = [*group_cols, *SIZE_CHANGE_COLUMNS]
    if areas.empty:
        empty = areas.iloc[:0][group_cols].copy()
        for column in SIZE_CHANGE_COLUMNS:
            empty[column] = np.empty(0, np.float64 if column == "pct_change" else np.int64)
        return empty[out_columns]

    # Reset rather than sort in place: a concatenated scan can carry duplicate
    # index labels, and the shift below is aligned on the index.
    frame = areas.sort_values([*group_cols, "z"], kind="stable").reset_index(drop=True)
    previous = frame.groupby(group_cols, sort=False, observed=True)[["z", "area"]].shift(1)

    frame = frame.assign(z_from=previous["z"], area_from=previous["area"])
    frame = frame[frame["z_from"].notna()].rename(columns={"z": "z_to", "area": "area_to"})
    # The shift makes these float to carry the NaN; the rows that survive it are
    # whole numbers, and downstream groupbys read better on integers.
    frame["z_from"] = frame["z_from"].astype(np.int64)
    frame["area_from"] = frame["area_from"].astype(np.int64)

    frame["z_gap"] = frame["z_to"] - frame["z_from"]
    frame["delta"] = frame["area_to"] - frame["area_from"]
    frame["pct_change"] = 100.0 * frame["delta"] / frame["area_from"]
    if adjacent_only:
        frame = frame[frame["z_gap"] == 1]

    return frame[out_columns].reset_index(drop=True)


def summarise_z_spans(
    z_table: pd.DataFrame,
    *,
    layer_span_cutoff: int = 1,
) -> dict[str, float]:
    """Collapse a per-label table into one row of headline numbers."""
    n_labels = len(z_table)
    if n_labels == 0:
        return {
            "n_labels": 0,
            "n_multi_layer": 0,
            "pct_multi_layer": float("nan"),
            "mean_z_span": float("nan"),
            "median_z_span": float("nan"),
            "max_z_span": float("nan"),
            "n_with_z_gaps": 0,
        }

    spans = z_table["z_span"].to_numpy()
    n_multi = int((spans > layer_span_cutoff).sum())
    return {
        "n_labels": n_labels,
        "n_multi_layer": n_multi,
        "pct_multi_layer": 100.0 * n_multi / n_labels,
        "mean_z_span": float(spans.mean()),
        "median_z_span": float(np.median(spans)),
        "max_z_span": int(spans.max()),
        "n_with_z_gaps": int((z_table["n_planes"].to_numpy() < spans).sum()),
    }
