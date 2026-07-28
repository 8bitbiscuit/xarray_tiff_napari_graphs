"""Lazy, chunk-aware access to segmentation masks stored as multi-page TIFFs.

The TIFF is never read through ``tifffile.imread``.  Instead ``virtual_tiff``
parses the file's IFDs into a :class:`~virtualizarr.manifests.ManifestStore` --
a zarr store whose chunks point straight at the byte ranges of the TIFF tiles.
Opening that store with zarr gives arrays that only fetch the tiles you index,
so a volume far larger than RAM can be walked a tile at a time.

``virtual_tiff`` exposes each IFD (each z-plane, here) as its own 2D array, so
:class:`MaskVolume` stacks them back into a ``(z, y, x)`` view.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Literal

import numpy as np
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from virtual_tiff import VirtualTIFF

if TYPE_CHECKING:  # pragma: no cover - import cost only paid by type checkers
    import xarray as xr

__all__ = ["DEFAULT_READ_BYTES", "MaskVolume", "local_registry", "open_mask_volume"]

IFDLayout = Literal["flat", "nested"]

#: Byte budget for one read.  Multiply by the worker count for the real peak.
#:
#: Measured on 7x4000x4000 uint32 striped TIFFs: anything from 4 to 32 MiB ties
#: within run-to-run noise, while the extremes cost real time -- under 2 MiB the
#: per-read sync overhead dominates, and at 64 MiB allocation and cache pressure
#: both slow it down and make it erratic (25% spread against 5%).  8 MiB sits at
#: the low-variance end of the plateau and matches a typical per-core L2.
#:
#: The optimum is hardware- and layout-dependent; :func:`zspan.tune_read_size`
#: measures it on your own data.  See :meth:`MaskVolume.read_block_shape`.
DEFAULT_READ_BYTES = 8 << 20  # 8 MiB


def local_registry(root: str | Path) -> tuple[ObjectStoreRegistry, str]:
    """Build a registry serving every file under ``root``.

    One registry can back an entire scan -- there is no need to build a store
    per file, and reusing it keeps connection setup out of the hot loop.

    Returns the registry and the ``file://`` URL its store is rooted at.
    """
    root = Path(root).resolve()
    root_url = root.as_uri()
    return ObjectStoreRegistry({root_url: LocalStore(prefix=root)}), root_url


def _page_arrays(group: zarr.Group, sample: int) -> tuple[zarr.Array, ...]:
    """Return one 2D zarr array per IFD, ordered by IFD index.

    ``zarr`` lists keys lexicographically, which puts page 10 before page 2, so
    the keys are sorted numerically before stacking.
    """
    group_keys = sorted(group.group_keys(), key=int)
    if group_keys:  # ifd_layout="nested": /<ifd>/<sample>
        pages = []
        for key in group_keys:
            sub = group[key]
            sample_keys = sorted(sub.array_keys(), key=int)
            if sample >= len(sample_keys):
                raise IndexError(
                    f"IFD {key} has {len(sample_keys)} sample(s); sample={sample} requested"
                )
            pages.append(sub[sample_keys[sample]])
        return tuple(pages)

    array_keys = sorted(group.array_keys(), key=int)  # ifd_layout="flat": /<ifd>
    if not array_keys:
        raise ValueError("no arrays found in the parsed TIFF")
    return tuple(group[key] for key in array_keys)


@dataclass(frozen=True)
class MaskVolume:
    """A ``(z, y, x)`` label volume backed by one zarr array per z-plane.

    Nothing is held in memory beyond the tiles you ask for.  ``chunks`` reports
    the TIFF's own tile/strip geometry, and reads that follow it avoid decoding
    tiles you do not need.
    """

    path: Path
    url: str
    pages: tuple[zarr.Array, ...]

    @property
    def shape(self) -> tuple[int, int, int]:
        ny, nx = self.pages[0].shape
        return (len(self.pages), ny, nx)

    @property
    def chunks(self) -> tuple[int, int, int]:
        """Native chunking, ``(1, tile_y, tile_x)``.

        Each IFD is a separate array, so z is always chunked one plane at a time.
        """
        cy, cx = self.pages[0].chunks
        return (1, cy, cx)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.pages[0].dtype)

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * self.dtype.itemsize

    def __repr__(self) -> str:
        z, y, x = self.shape
        return (
            f"MaskVolume({self.path.name!r}, shape=({z}, {y}, {x}), "
            f"chunks={self.chunks}, dtype={self.dtype})"
        )

    def read_plane(
        self,
        z: int,
        y: slice = slice(None),
        x: slice = slice(None),
    ) -> np.ndarray:
        """Read one (region of one) z-plane, fetching only the tiles it covers."""
        return np.asarray(self.pages[z][y, x])

    def read_block_shape(self, target_bytes: int = DEFAULT_READ_BYTES) -> tuple[int, int]:
        """Pick a chunk-aligned ``(y, x)`` read size of roughly ``target_bytes``.

        Reading one chunk at a time is the memory optimum but a throughput
        disaster: every ``page[...]`` call crosses zarr's sync-to-async bridge,
        which costs ~1 ms of thread handoff regardless of how little data comes
        back.  A 4000x4000 striped TIFF has 250 sixteen-row strips per plane, so
        the native size spends most of its time waiting on that bridge rather
        than reading.  Batching whole chunks up to a byte budget amortises it.

        Blocks span the full plane width -- chunks are laid out row-major, so a
        full-width block is the cheapest shape per chunk fetched.
        """
        _, ny, nx = self.shape
        _, chunk_y, _ = self.chunks
        row_bytes = nx * self.dtype.itemsize
        if row_bytes >= target_bytes:  # a single row already blows the budget
            return (chunk_y, nx)
        whole_chunks = max(1, (target_bytes // row_bytes) // chunk_y)
        return (min(ny, whole_chunks * chunk_y), nx)

    def iter_plane_blocks(
        self,
        z: int,
        block_shape: tuple[int, int] | None = None,
        *,
        target_bytes: int = DEFAULT_READ_BYTES,
    ) -> Iterator[np.ndarray]:
        """Yield a z-plane as chunk-aligned blocks.

        ``block_shape`` defaults to :meth:`read_block_shape`, which batches
        whole chunks up to ``target_bytes``.  Blocks stay chunk-aligned either
        way, so no chunk is ever decoded twice.  Pass an explicit
        ``block_shape`` to override, or lower ``target_bytes`` to cut peak
        memory at the cost of more round-trips.
        """
        _, ny, nx = self.shape
        by, bx = block_shape if block_shape is not None else self.read_block_shape(target_bytes)
        page = self.pages[z]
        for y0 in range(0, ny, by):
            y1 = min(y0 + by, ny)
            for x0 in range(0, nx, bx):
                x1 = min(x0 + bx, nx)
                yield np.asarray(page[y0:y1, x0:x1])

    def to_dask(self, z_chunk: int = 1):
        """Stack the pages into a lazy ``(z, y, x)`` dask array."""
        import dask.array as da

        stacked = da.stack([da.from_zarr(page) for page in self.pages])
        return stacked.rechunk({0: z_chunk}) if z_chunk != 1 else stacked

    def to_xarray(self, name: str = "masks", z_chunk: int = 1) -> "xr.DataArray":
        """Lazy ``xarray.DataArray`` with named ``(z, y, x)`` dims.

        Handy for napari (``viewer.add_labels(vol.to_xarray().data)``) and for
        any xarray-native downstream work.
        """
        import xarray as xr

        return xr.DataArray(
            self.to_dask(z_chunk=z_chunk),
            dims=("z", "y", "x"),
            name=name,
            attrs={"source": str(self.path)},
        )


def open_mask_volume(
    path: str | Path,
    registry: ObjectStoreRegistry | None = None,
    *,
    ifd_layout: IFDLayout = "nested",
    sample: int = 0,
) -> MaskVolume:
    """Open a multi-page TIFF as a lazy :class:`MaskVolume`.

    Parameters
    ----------
    path
        Path to the ``.tif``.
    registry
        Reuse a registry from :func:`local_registry` when opening many files.
        A single-file local registry is built on demand when omitted.
    ifd_layout
        ``virtual_tiff`` layout.  ``"nested"`` groups each IFD's samples under
        the IFD; ``"flat"`` puts one array per IFD at the root.  Both are
        handled, and both give one 2D array per z-plane for single-sample masks.
    sample
        Which sample (channel) to take from each IFD, for ``"nested"`` files
        that carry more than one.
    """
    path = Path(path).resolve()
    url = path.as_uri()
    if registry is None:
        registry, _ = local_registry(path.parent)

    manifest_store = VirtualTIFF(ifd_layout=ifd_layout)(url, registry)
    group = zarr.open_group(store=manifest_store, mode="r")
    return MaskVolume(path=path, url=url, pages=_page_arrays(group, sample))
