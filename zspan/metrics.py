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
"""

from __future__ import annotations

from typing import Any, Iterator, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from .loading import DEFAULT_READ_BYTES, MaskVolume

__all__ = ["check_z_span", "summarise_z_spans", "Z_SPAN_COLUMNS"]

Z_SPAN_COLUMNS = (
    "label",
    "z_start",
    "z_end",
    "z_span",
    "n_planes",
    "n_voxels",
    "spans_multiple_layers",
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

    def iter_blocks(
        self, z: int, block_shape: tuple[int, int] | None
    ) -> Iterator[np.ndarray]:
        return self._volume.iter_plane_blocks(
            z, block_shape, target_bytes=self._target_bytes
        )


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

    def iter_blocks(
        self, z: int, block_shape: tuple[int, int] | None
    ) -> Iterator[np.ndarray]:
        _, ny, nx = self._array.shape
        by, bx = block_shape if block_shape is not None else (ny, nx)
        for y0 in range(0, ny, by):
            for x0 in range(0, nx, bx):
                yield np.asarray(
                    self._array[z, y0 : min(y0 + by, ny), x0 : min(x0 + bx, nx)]
                )


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
