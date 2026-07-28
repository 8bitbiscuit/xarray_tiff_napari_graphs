"""Measure the best read size for a given volume, machine and storage.

The default in :data:`~zspan.loading.DEFAULT_READ_BYTES` is a reasonable middle,
but the optimum moves with the TIFF's own layout (strip height, tile size,
dtype), the CPU's cache, and whether the bytes come off local NVMe or an object
store.  This measures it instead of assuming it.

The one trap worth knowing: the curve is *flat* over a wide middle range, and
run-to-run noise is easily 5-25%.  Taking the fastest single run overfits to
noise, so :func:`tune_read_size` reports the spread and
:func:`best_read_size` deliberately returns the **smallest** size that ties the
winner -- smaller reads mean less memory per worker, for no measurable time.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import pandas as pd

from .loading import DEFAULT_READ_BYTES, MaskVolume, open_mask_volume
from .metrics import check_z_span

__all__ = ["DEFAULT_TARGETS", "best_read_size", "tune_read_size"]

#: Candidate budgets spanning the useful range, 1 MiB to 64 MiB.
DEFAULT_TARGETS: tuple[int, ...] = tuple(mib << 20 for mib in (1, 2, 4, 8, 16, 32, 64))


def tune_read_size(
    volume: MaskVolume | str | Path,
    *,
    targets: Sequence[int] = DEFAULT_TARGETS,
    repeats: int = 5,
    layer_span_cutoff: int = 1,
    background: int | None = 0,
) -> pd.DataFrame:
    """Time :func:`~zspan.metrics.check_z_span` at each candidate read size.

    Parameters
    ----------
    volume
        A :class:`~zspan.loading.MaskVolume`, or a path to one mask TIFF.  Use a
        representative volume -- geometry matters more than content.
    targets
        Byte budgets to try.
    repeats
        Timed runs per candidate.  Three is enough to rank; five to trust the
        spread.

    Returns
    -------
    pandas.DataFrame
        One row per candidate, sorted by ``target_bytes``, with the resulting
        ``block_shape``, ``reads_per_volume``, timing (``min_s``, ``median_s``),
        ``spread_pct`` (max-to-min, the noise floor for that size), and
        ``vs_best`` (median relative to the fastest candidate).

    Notes
    -----
    The first run is discarded as warm-up, so the page cache state is consistent
    across candidates.  That makes this a *warm-cache* measurement: it isolates
    per-read overhead rather than disk throughput, which is the thing the read
    size actually controls.
    """
    if not isinstance(volume, MaskVolume):
        volume = open_mask_volume(volume)
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    n_planes, ny, _ = volume.shape
    rows = []
    for target in targets:
        block_shape = volume.read_block_shape(target)
        check_z_span(volume, block_shape=block_shape)  # warm-up, not timed

        timings = []
        for _ in range(repeats):
            started = time.perf_counter()
            check_z_span(
                volume,
                layer_span_cutoff,
                background=background,
                block_shape=block_shape,
            )
            timings.append(time.perf_counter() - started)

        low, high = min(timings), max(timings)
        rows.append(
            {
                "target_bytes": target,
                "target_mib": target / 2**20,
                "block_shape": block_shape,
                "block_mib": block_shape[0] * block_shape[1] * volume.dtype.itemsize / 2**20,
                "reads_per_volume": -(-ny // block_shape[0]) * n_planes,
                "min_s": low,
                "median_s": float(pd.Series(timings).median()),
                "spread_pct": 100.0 * (high - low) / low,
            }
        )

    table = pd.DataFrame(rows)
    table["vs_best"] = table["median_s"] / table["median_s"].min()
    return table


def best_read_size(
    table: pd.DataFrame | MaskVolume | str | Path,
    *,
    tolerance: float = 0.10,
    **kwargs,
) -> int:
    """Pick a read size from the middle of the fast plateau.

    Pass either a table from :func:`tune_read_size` or a volume to measure now.

    What this does *not* do is return the argmin.  The timing curve is flat
    across a wide middle range -- on 7x4000x4000 uint32 TIFFs, 2 MiB through
    32 MiB all land within noise of each other, and repeated runs moved the
    argmin between 4, 8 and 16 MiB with nothing changing.  Chasing that is
    fitting to interference.

    So: take every candidate within ``tolerance`` of the best as tied, then
    return the **middle** of that plateau.  The middle has margin from both
    failure modes -- per-read sync overhead off the small end, allocation and
    cache pressure off the large end -- which makes the answer stable across
    runs, and stable is the property that matters for a default.

    Ranking uses ``min_s`` rather than ``median_s``: interference only ever adds
    time, so the minimum is the least contaminated estimate of the true cost.
    """
    if not isinstance(table, pd.DataFrame):
        table = tune_read_size(table, **kwargs)
    if table.empty:
        return DEFAULT_READ_BYTES

    cutoff = table["min_s"].min() * (1.0 + tolerance)
    tied = table.loc[table["min_s"] <= cutoff].sort_values("target_bytes")
    return int(tied["target_bytes"].iloc[(len(tied) - 1) // 2])
