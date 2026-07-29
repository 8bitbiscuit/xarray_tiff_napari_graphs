"""Walk a segmentation output tree and score every volume in it.

The layout this targets looks like::

    data/<segmentation_type>/<segmentation_model>/<preprocessing>/<model>/<region>/<fov>/masks.tif

but nothing here is tied to that depth -- the directory components between the
root and the file become columns, so any nesting works.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import pandas as pd

from .loading import DEFAULT_READ_BYTES, IFDLayout, local_registry, open_mask_volume
from .metrics import check_z_span, summarise_z_spans

__all__ = ["ScanResult", "find_mask_files", "scan_segmentations", "add_variant_column"]

DEFAULT_PATTERN = "*/*/masks.tif"


def find_mask_files(root: str | Path, pattern: str = DEFAULT_PATTERN) -> list[Path]:
    """Sorted list of mask files under ``root`` matching a glob ``pattern``.

    ``pattern`` is relative to ``root``; ``"**/masks.tif"`` recurses to any depth.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"segmentation root does not exist: {root}")
    return sorted(root.glob(pattern))


def _level_columns(
    relative: Path, level_names: Sequence[str] | None
) -> dict[str, str]:
    """Map a file's parent directories to named columns.

    ``level_names`` aligns to the *trailing* directories, so passing
    ``("region", "fov")`` works whether the root sits above the variant
    directories or inside them.  Unnamed leading levels get ``level_<i>``.
    """
    parts = relative.parts[:-1]
    if level_names is None:
        names: list[str] = [f"level_{i}" for i in range(len(parts))]
    else:
        level_names = list(level_names)
        if len(level_names) > len(parts):
            raise ValueError(
                f"{len(level_names)} level names given but {relative} only has "
                f"{len(parts)} directory level(s) below the root"
            )
        pad = len(parts) - len(level_names)
        names = [f"level_{i}" for i in range(pad)] + level_names
    return dict(zip(names, parts))


@dataclass
class ScanResult:
    """Output of :func:`scan_segmentations`.

    Attributes
    ----------
    summary
        One row per volume: the directory levels plus the headline statistics
        (``n_labels``, ``pct_multi_layer``, ``mean_z_span``, ...).
    labels
        One row per label across every volume, carrying the same level columns.
        Empty when the scan ran with ``keep_labels=False``.
    failures
        ``(path, exception)`` for volumes that could not be read.
    """

    summary: pd.DataFrame
    labels: pd.DataFrame
    failures: list[tuple[Path, Exception]] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"ScanResult(volumes={len(self.summary)}, labels={len(self.labels)}, "
            f"failures={len(self.failures)})"
        )


def _score_one(
    path: Path,
    root: Path,
    registry,
    *,
    layer_span_cutoff: int,
    background: int | None,
    block_shape: tuple[int, int] | None,
    target_bytes: int,
    ifd_layout: IFDLayout,
    sample: int,
    keep_labels: bool,
) -> tuple[dict, pd.DataFrame | None]:
    volume = open_mask_volume(path, registry, ifd_layout=ifd_layout, sample=sample)
    z_table = check_z_span(
        volume,
        layer_span_cutoff,
        background=background,
        block_shape=block_shape,
        target_bytes=target_bytes,
    )
    nz, ny, nx = volume.shape
    row = {
        "path": str(path),
        "relpath": str(path.relative_to(root)),
        "n_z": nz,
        "shape": f"{nz}x{ny}x{nx}",
        **summarise_z_spans(z_table, layer_span_cutoff=layer_span_cutoff),
    }
    return row, (z_table if keep_labels else None)


def scan_segmentations(
    root: str | Path,
    pattern: str = DEFAULT_PATTERN,
    *,
    level_names: Sequence[str] | None = ("region", "fov"),
    layer_span_cutoff: int = 1,
    background: int | None = 0,
    block_shape: tuple[int, int] | None = None,
    target_bytes: int = DEFAULT_READ_BYTES,
    ifd_layout: IFDLayout = "nested",
    sample: int = 0,
    keep_labels: bool = True,
    max_workers: int | None = None,
    progress: Callable[[int, int, Path], None] | None = None,
) -> ScanResult:
    """Score every mask volume under ``root`` on how far its labels span in z.

    Volumes are read lazily in chunk-aligned blocks (see
    :func:`~zspan.metrics.check_z_span`), so total memory is set by
    ``target_bytes`` and the worker count, not by the size of the images.

    Parameters
    ----------
    root
        Directory holding the segmentation outputs, e.g.
        ``data/segmentations/cpdino`` or ``.../cpdino/decon/CBDN``.
    pattern
        Glob relative to ``root``.  The default ``"*/*/masks.tif"`` matches the
        ``<region>/<fov>/masks.tif`` layout; use ``"**/masks.tif"`` to sweep
        every variant under a method root at once.
    level_names
        Names for the trailing directory levels; extra leading levels become
        ``level_0``, ``level_1``, ...
    layer_span_cutoff
        A label spans multiple layers when its ``z_span`` exceeds this.
    target_bytes
        Byte budget for a single read.  Peak memory is roughly this times
        ``max_workers``.
    max_workers
        Threads used to read volumes.  Expect only a modest speedup: zarr funnels
        every chunk request through one background event loop, so workers
        contend on it rather than scaling linearly.  Measured at ~1.1x going
        from 1 to 4 threads on striped TIFFs -- ``target_bytes`` is by far the
        bigger lever.  Defaults to ``min(8, cpu_count)``; pass ``1`` for a
        deterministic serial scan.
    progress
        Called as ``progress(done, total, path)`` after each volume.

    Returns
    -------
    ScanResult
        ``.summary`` (one row per volume) and ``.labels`` (one row per label).
    """
    root = Path(root).resolve()
    files = find_mask_files(root, pattern)
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} under {root}")

    # Validate the layout up front: a level-name mismatch is a configuration
    # error, and must not be mistaken for one unreadable file mid-scan.
    _level_columns(files[0].relative_to(root), level_names)

    registry, _ = local_registry(root)
    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 1)

    kwargs = dict(
        layer_span_cutoff=layer_span_cutoff,
        background=background,
        block_shape=block_shape,
        target_bytes=target_bytes,
        ifd_layout=ifd_layout,
        sample=sample,
        keep_labels=keep_labels,
    )

    rows: list[dict] = []
    label_frames: list[pd.DataFrame] = []
    failures: list[tuple[Path, Exception]] = []
    done = 0

    def collect(path: Path, result, error: Exception | None) -> None:
        nonlocal done
        done += 1
        if error is not None:
            failures.append((path, error))
        else:
            row, z_table = result
            levels = _level_columns(path.relative_to(root), level_names)
            rows.append({**levels, **row})
            if z_table is not None and len(z_table):
                label_frames.append(z_table.assign(**levels, relpath=row["relpath"]))
        if progress is not None:
            progress(done, len(files), path)

    # Only reading a volume is allowed to fail softly -- the bookkeeping in
    # ``collect`` stays outside the guard so genuine bugs still surface.
    if max_workers == 1:
        for path in files:
            try:
                scored, error = _score_one(path, root, registry, **kwargs), None
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the scan
                scored, error = None, exc
            collect(path, scored, error)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(_score_one, path, root, registry, **kwargs): path
                for path in files
            }
            for future in as_completed(futures):
                path = futures[future]
                try:
                    scored, error = future.result(), None
                except Exception as exc:  # noqa: BLE001
                    scored, error = None, exc
                collect(path, scored, error)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        sort_cols = [c for c in summary.columns if c not in {"path", "relpath"}]
        level_cols = [c for c in sort_cols if summary[c].dtype == object]
        summary = summary.sort_values(level_cols or ["relpath"]).reset_index(drop=True)

    labels = (
        pd.concat(label_frames, ignore_index=True)
        if label_frames
        else pd.DataFrame(columns=["label", "z_span", "spans_multiple_layers"])
    )
    return ScanResult(summary=summary, labels=labels, failures=failures)


def add_variant_column(
    frame: pd.DataFrame,
    levels: Iterable[str],
    *,
    name: str = "variant",
    sep: str = " / ",
) -> pd.DataFrame:
    """Join several level columns into one label for grouping and plotting.

    e.g. ``add_variant_column(df, ["level_0", "level_1"])`` turns
    ``decon`` + ``CBDN`` into a ``variant`` column reading ``decon / CBDN``.
    """
    levels = [c for c in levels if c in frame.columns]
    if not levels:
        raise ValueError("none of the requested level columns are present")
    joined = frame[levels[0]].astype(str)
    for column in levels[1:]:
        joined = joined + sep + frame[column].astype(str)
    return frame.assign(**{name: joined})
