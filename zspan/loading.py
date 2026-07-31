"""Lazy, chunk-aware access to segmentation masks stored as multi-page TIFFs.

Two backends produce the same lazy ``(z, y, x)`` handle, selected by ``reader``:

``"virtual"``
    ``virtual_tiff`` parses the file's IFDs into a
    :class:`~virtualizarr.manifests.ManifestStore` -- a zarr store whose chunks
    point straight at the byte ranges of the TIFF tiles.  The manifest is a
    portable artifact: it can be persisted (Icechunk/Kerchunk) and it can front
    an object store through ``obstore``, which is what makes ``s3://`` work.
``"tifffile"``
    ``tifffile``'s own ``aszarr`` store.  Same sub-plane laziness, same chunk
    geometry, no manifest -- so no manifest cost either.  Local files only.

``"auto"`` (the default) picks ``"virtual"`` for anything remote or backed by an
explicit registry, and ``"tifffile"`` otherwise.  The two agree exactly: same
``chunks``, same blocks, identical statistics -- ``tests/test_zspan.py`` asserts
it against the same oracle used for the streaming/eager equivalence.  The
manifest is what you are paying for, so pay for it when it buys something:

============================  ===================  ====================
7x4000x4000 uint32, striped   virtual              tifffile
============================  ===================  ====================
open (manifest construction)  325 ms               1 ms
read @ 8 MiB blocks           1.35 s               1.07 s
peak memory                   29 MB                29 MB
============================  ===================  ====================

``virtual_tiff`` exposes each IFD (each z-plane, here) as its own 2D array;
``tifffile`` presents the series as one 3D array.  :class:`MaskVolume` works in
terms of the former, so the latter is adapted by :class:`_TiffPlaneView` and
everything downstream stays on one code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Literal
from urllib.parse import urlsplit

import numpy as np
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from virtual_tiff import VirtualTIFF

if TYPE_CHECKING:  # pragma: no cover - import cost only paid by type checkers
    import xarray as xr

__all__ = [
    "DEFAULT_READ_BYTES",
    "MaskVolume",
    "Reader",
    "local_registry",
    "open_mask_volume",
    "resolve_reader",
]

IFDLayout = Literal["flat", "nested"]

#: Which backend fetches the bytes.  ``"auto"`` routes on the path -- see
#: :func:`resolve_reader`.
Reader = Literal["auto", "virtual", "tifffile"]

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


def _is_remote(path: str | Path) -> bool:
    """True for a URL that is not a local file.

    A bare path has no scheme; ``file://`` is local; ``C:\\...`` parses as scheme
    ``"c"``, hence the length guard.
    """
    scheme = urlsplit(str(path)).scheme
    return len(scheme) > 1 and scheme != "file"


def resolve_reader(
    reader: Reader,
    path: str | Path,
    registry: ObjectStoreRegistry | None = None,
) -> Literal["virtual", "tifffile"]:
    """Turn ``"auto"`` into a concrete backend; pass anything else through.

    ``"auto"`` picks ``"virtual"`` when the bytes are not a plain local file --
    a non-``file`` URL, or an explicit registry, which is how an object store is
    handed in.  Everything else gets ``"tifffile"``, which is faster locally and
    gives up nothing there.

    Routing on the path rather than on a global means the same notebook works
    unchanged when the data moves to ``s3://``.
    """
    if reader == "auto":
        return "virtual" if (registry is not None or _is_remote(path)) else "tifffile"
    if reader not in ("virtual", "tifffile"):
        raise ValueError(
            f"reader must be 'auto', 'virtual' or 'tifffile', got {reader!r}"
        )
    return reader


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


class _TiffPlaneView:
    """One z-plane of a 3D ``tifffile`` zarr array, shaped like a 2D zarr array.

    ``MaskVolume`` is written against ``virtual_tiff``'s one-array-per-IFD
    layout.  ``tifffile`` presents the series as a single ``(z, y, x)`` array,
    so this adapts it -- ``shape``, ``chunks``, ``dtype`` and ``[y, x]``
    indexing are the whole interface ``MaskVolume`` needs from a page.

    Slicing stays lazy: the underlying store fetches only the tiles the window
    covers, exactly as the virtual backend does.
    """

    __slots__ = ("_array", "_z")

    def __init__(self, array: zarr.Array, z: int) -> None:
        self._array = array
        self._z = z

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self._array.shape[1:])  # type: ignore[return-value]

    @property
    def chunks(self) -> tuple[int, int]:
        return tuple(self._array.chunks[1:])  # type: ignore[return-value]

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self._array.dtype)

    def __getitem__(self, key: Any) -> np.ndarray:
        y, x = key if isinstance(key, tuple) else (key, slice(None))
        return self._array[self._z, y, x]


def _open_virtual(
    path: Path,
    url: str,
    registry: ObjectStoreRegistry | None,
    ifd_layout: IFDLayout,
    sample: int,
) -> tuple[None, tuple[zarr.Array, ...]]:
    """Parse the IFDs into a manifest store, one array per IFD."""
    if registry is None:
        registry, _ = local_registry(path.parent)
    manifest_store = VirtualTIFF(ifd_layout=ifd_layout)(url, registry)
    group = zarr.open_group(store=manifest_store, mode="r")
    return None, _page_arrays(group, sample)


def _open_tifffile(path: Path, sample: int) -> tuple[Any, tuple[Any, ...]]:
    """Open through ``tifffile``'s own zarr store, one view per plane.

    The store is returned alongside the pages because ``to_dask`` wants the 3D
    array directly rather than a stack of per-plane views.
    """
    import tifffile

    store = tifffile.imread(path, aszarr=True)
    array = zarr.open(store, mode="r")

    if array.ndim == 2:  # single-page TIFF: one plane, already 2D
        return store, (array,)
    if array.ndim != 3:
        raise ValueError(
            f"{path.name} parses as a {array.ndim}D array {array.shape}; the "
            "tifffile reader handles single-sample (z, y, x) masks. Use "
            "reader='virtual', which selects a sample with sample=."
        )
    if sample != 0:
        raise ValueError(
            f"sample={sample} requires reader='virtual'; {path.name} parses as a "
            "single-sample volume under the tifffile reader."
        )
    return store, tuple(_TiffPlaneView(array, z) for z in range(array.shape[0]))


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
    #: Which backend fetched these pages -- ``"virtual"`` or ``"tifffile"``.
    backend: str = "virtual"
    #: The tifffile store, kept so ``to_dask`` can use the 3D array directly.
    #: ``None`` for the virtual backend, which stacks per-IFD arrays instead.
    store: Any = field(default=None, repr=False, compare=False)

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
            f"chunks={self.chunks}, dtype={self.dtype}, reader={self.backend!r})"
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

    def iter_plane_windows(
        self,
        z: int,
        block_shape: tuple[int, int] | None = None,
        *,
        target_bytes: int = DEFAULT_READ_BYTES,
    ) -> Iterator[tuple[tuple[int, int, int, int], np.ndarray]]:
        """Yield ``((y0, y1, x0, x1), block)`` for a z-plane, chunk-aligned.

        Same reads as :meth:`iter_plane_blocks`, with the window each block came
        from.  Statistics that are *positional* -- per-quadrant, per-band --
        need to know where a block sat without holding the plane to find out.
        """
        _, ny, nx = self.shape
        by, bx = block_shape if block_shape is not None else self.read_block_shape(target_bytes)
        page = self.pages[z]
        for y0 in range(0, ny, by):
            y1 = min(y0 + by, ny)
            for x0 in range(0, nx, bx):
                x1 = min(x0 + bx, nx)
                yield (y0, y1, x0, x1), np.asarray(page[y0:y1, x0:x1])

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
        for _, block in self.iter_plane_windows(
            z, block_shape, target_bytes=target_bytes
        ):
            yield block

    def to_dask(self, z_chunk: int = 1):
        """Stack the pages into a lazy ``(z, y, x)`` dask array.

        The one method the two backends cannot share: ``da.from_zarr`` needs a
        real zarr array, which the virtual backend has one of per IFD, while the
        tifffile store is already the 3D array.
        """
        import dask.array as da

        if self.backend == "tifffile":
            stacked = da.from_zarr(self.store)
            if stacked.ndim == 2:  # single-page TIFF
                stacked = stacked[None]
        else:
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
    reader: Reader = "auto",
    ifd_layout: IFDLayout = "nested",
    sample: int = 0,
) -> MaskVolume:
    """Open a multi-page TIFF as a lazy :class:`MaskVolume`.

    Parameters
    ----------
    path
        Path to the ``.tif``, or a URL when ``reader="virtual"``.
    registry
        Reuse a registry from :func:`local_registry` when opening many files, or
        supply an object store for remote data.  Virtual backend only; passing
        one is itself a signal that ``"auto"`` should route to ``"virtual"``.
    reader
        Which backend fetches the bytes -- ``"virtual"``, ``"tifffile"``, or
        ``"auto"`` (default) to route on the path.  See :func:`resolve_reader`.
        The two produce identical statistics; they differ in what they cost and
        in whether they can reach an object store.
    ifd_layout
        ``virtual_tiff`` layout, virtual backend only.  ``"nested"`` groups each
        IFD's samples under the IFD; ``"flat"`` puts one array per IFD at the
        root.  Both give one 2D array per z-plane for single-sample masks.
    sample
        Which sample (channel) to take from each IFD, for ``"nested"`` files
        that carry more than one.  Virtual backend only.
    """
    backend = resolve_reader(reader, path, registry)

    if backend == "tifffile":
        # Silence rather than error would put a "why is this still slow" -- or
        # worse, a silently local read of remote data -- a long way from here.
        if registry is not None:
            raise ValueError(
                "reader='tifffile' cannot use an ObjectStoreRegistry: it reads "
                "local files directly. Use reader='virtual' for object stores, "
                "or drop the registry."
            )
        if _is_remote(path):
            raise ValueError(
                f"reader='tifffile' cannot read {path!r}; use reader='virtual' "
                "with a registry for remote data."
            )
        if ifd_layout != "nested":
            raise ValueError(
                f"ifd_layout={ifd_layout!r} is a virtual_tiff option and has no "
                "effect under reader='tifffile'; use reader='virtual'."
            )

    path = Path(path).resolve()
    url = path.as_uri()
    if backend == "tifffile":
        store, pages = _open_tifffile(path, sample)
    else:
        store, pages = _open_virtual(path, url, registry, ifd_layout, sample)
    return MaskVolume(path=path, url=url, pages=pages, backend=backend, store=store)
