"""Lazy, chunk-aware access to segmentation TIFFs, local or on S3.

Recovered from the deleted ``zspan/loading.py`` (commit ``3267a2b^``) and fixed:
the original mangled every path through ``Path(...).resolve().as_uri()``, which
turned ``s3://bucket/key`` into ``file:///cwd/s3%3A/bucket/key``, so its
advertised object-store support never actually worked.  URLs are now passed
through untouched -- see :func:`as_url`.

Masks are one multi-page TIFF per FOV (one IFD per z-plane).  DAPI is one
single-page TIFF *per* z-plane, so :func:`open_dapi_volume` stacks a directory
of them; both end up as the same :class:`MaskVolume` handle.

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
``chunks``, same blocks, identical statistics.  (The deleted ``tests/test_zspan.py``
asserted that against a shared oracle; those tests went with the package, so the
equivalence is now checked notebook-side against ``tifffile.imread``.)  The
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

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Literal
from urllib.parse import urlsplit

import numpy as np
import obstore
import zarr
from obspec_utils.registry import ObjectStoreRegistry
from obstore.store import LocalStore
from virtual_tiff import VirtualTIFF

if TYPE_CHECKING:  # pragma: no cover - import cost only paid by type checkers
    import xarray as xr

__all__ = [
    "DAPI_PATTERN",
    "DEFAULT_READ_BYTES",
    "MaskVolume",
    "Reader",
    "as_url",
    "exists",
    "has_contents",
    "join_url",
    "list_files",
    "list_subdirs",
    "load_dapi",
    "load_masks",
    "local_registry",
    "make_registry",
    "open_dapi_volume",
    "open_mask_volume",
    "resolve_reader",
]

#: DAPI is stored as one single-page TIFF per z index, inside the FOV directory.
DAPI_PATTERN = "DAPI_decon_z*.tif"

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
#: Reads over a network are the case this really matters for: every chunk is
#: its own HTTP request, so the native strip size can cost a thousand round
#: trips per volume.  The optimum is hardware- and layout-dependent; time a few
#: values on your own data.  See :meth:`MaskVolume.read_block_shape`.
DEFAULT_READ_BYTES = 8 << 20  # 8 MiB


def _read_bytes(target_bytes: int | None) -> int:
    """Resolve the read budget, deferring to the module default when unset.

    Read at call time rather than baked into a default argument, so that
    ``sl.DEFAULT_READ_BYTES = 2 << 20`` in a notebook actually changes what the
    loaders do instead of being silently ignored.
    """
    return DEFAULT_READ_BYTES if target_bytes is None else target_bytes


def _is_remote(path: str | Path) -> bool:
    """True for a URL that is not a local file.

    A bare path has no scheme; ``file://`` is local; ``C:\\...`` parses as scheme
    ``"c"``, hence the length guard.
    """
    scheme = urlsplit(str(path)).scheme
    return len(scheme) > 1 and scheme != "file"


def as_url(path: str | Path) -> str:
    """A URL for either a local path or something already remote.

    Local paths become ``file://`` URLs.  Anything that already carries a scheme
    (``s3://``, ``gs://``, ``https://``) is returned untouched -- this is the
    whole bug fix.  Running ``Path()`` over ``s3://bucket/key`` silently
    collapses the double slash and then anchors the result to the working
    directory, producing a ``file://`` URL that points nowhere near the data.
    """
    text = str(path)
    if _is_remote(text) or text.startswith("file://"):
        return text
    return Path(text).resolve().as_uri()


def join_url(base: str | Path, *parts: str) -> str:
    """Join URL pieces with ``/``.

    ``Path("s3://b") / "x"`` gives ``s3:/b/x`` -- pathlib collapses the double
    slash -- so URLs are joined as text, never with the path operator.  A local
    directory is converted to a ``file://`` URL first, so one call covers both.
    """
    url = as_url(base)
    # strip trailing slashes from the *path*, never from the scheme separator:
    # a plain rstrip("/") turns "memory:///" into "memory:" and "s3://" into "s3:"
    scheme, sep, rest = url.partition("://")
    url = f"{scheme}://{rest.rstrip('/')}" if sep else url.rstrip("/")
    for part in parts:
        part = str(part).strip("/")
        if part:
            url = f"{url}/{part}"
    return url


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


def make_registry(
    base: str | Path,
    *,
    region: str | None = None,
    anonymous: bool = False,
    endpoint: str | None = None,
    **kwargs: Any,
) -> tuple[ObjectStoreRegistry, str]:
    """Build a registry for a local directory *or* a bucket, and return its URL.

    ``obstore.store.from_url`` already dispatches on the scheme -- ``file://``,
    ``s3://``, ``gs://``, ``az://`` -- so local and remote need no separate code
    path here or anywhere downstream.  That is what lets the S3 notebook be
    checked against the same functions running on local data.

    The store is registered under the *same* URL it was built from, because
    ``ObjectStoreRegistry.resolve`` matches on scheme, host and path prefix and
    hands back the trailing path; keys that do not line up resolve to nothing.

    Parameters
    ----------
    base
        Bucket URL (``s3://bucket/prefix``) or a local directory.
    region
        AWS region.  Required for S3, ignored elsewhere.
    anonymous
        Skip request signing, for a public bucket.
    endpoint
        Alternate S3-compatible endpoint (MinIO, Ceph, R2).

    Returns
    -------
    registry, url
        Pass ``registry`` to every loader; build paths under ``url``.
    """
    url = as_url(base)
    opts: dict[str, Any] = dict(kwargs)
    if region is not None:
        opts["region"] = region
    if anonymous:
        opts["skip_signature"] = True
    if endpoint is not None:
        opts["endpoint"] = endpoint
    return ObjectStoreRegistry({url: obstore.store.from_url(url, **opts)}), url


def _list_one_level(url: str, registry: ObjectStoreRegistry):
    """One level of listing under ``url``: ``(common_prefixes, object names)``.

    Object storage has no directories, only key prefixes, so ``Path.glob`` is
    not an option.  ``list_with_delimiter`` is the equivalent: it stops at the
    next ``/`` and reports the prefixes it stopped at alongside the objects
    directly inside.  It works on a ``LocalStore`` too, so this is the only
    directory walk in the module.
    """
    store, rel = registry.resolve(url)
    prefix = str(rel).strip("/")
    result = obstore.list_with_delimiter(store, prefix or None)
    names = [str(meta["path"]).rstrip("/").rsplit("/", 1)[-1] for meta in result["objects"]]
    dirs = [str(p).rstrip("/").rsplit("/", 1)[-1] for p in result["common_prefixes"]]
    return dirs, names


def list_subdirs(url: str, registry: ObjectStoreRegistry, pattern: str = "*") -> list[str]:
    """Directory names one level under ``url``, matching ``pattern``."""
    dirs, _ = _list_one_level(url, registry)
    return sorted(d for d in dirs if fnmatch.fnmatch(d, pattern))


def list_files(url: str, registry: ObjectStoreRegistry, pattern: str = "*") -> list[str]:
    """File names one level under ``url``, matching ``pattern``."""
    _, names = _list_one_level(url, registry)
    return sorted(n for n in names if fnmatch.fnmatch(n, pattern))


def exists(url: str, registry: ObjectStoreRegistry) -> bool:
    """True if a single object is present at ``url``."""
    store, rel = registry.resolve(url)
    try:
        obstore.head(store, str(rel).strip("/"))
    except Exception:
        return False
    return True


def has_contents(url: str, registry: ObjectStoreRegistry) -> bool:
    """True if anything at all is stored under ``url``.

    :func:`exists` asks about one object; a technique root is a *prefix*, and an
    object store has nothing to HEAD at a prefix.  Listing is the only way to
    ask -- and a bad bucket or a rejected credential raises rather than coming
    back empty, so both land as False here.
    """
    try:
        dirs, names = _list_one_level(url, registry)
    except Exception:
        return False
    return bool(dirs or names)


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
    url: str,
    registry: ObjectStoreRegistry,
    ifd_layout: IFDLayout,
    sample: int,
) -> tuple[None, tuple[zarr.Array, ...]]:
    """Parse the IFDs into a manifest store, one array per IFD.

    Takes a URL, not a path: the caller has already resolved one, and rebuilding
    it here is what broke the remote case in the original.
    """
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

    path: str | Path
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

    @property
    def name(self) -> str:
        """Last path segment, for messages.  ``path`` may be a URL, not a Path."""
        return str(self.path).rstrip("/").rsplit("/", 1)[-1]

    def __repr__(self) -> str:
        z, y, x = self.shape
        return (
            f"MaskVolume({self.name!r}, shape=({z}, {y}, {x}), "
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

    def read_block_shape(self, target_bytes: int | None = None) -> tuple[int, int]:
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
        target_bytes = _read_bytes(target_bytes)
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
        target_bytes: int | None = None,
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
        target_bytes: int | None = None,
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

    def to_dask(self, z_chunk: int = 1, target_bytes: int | None = None):
        """Stack the pages into a lazy ``(z, y, x)`` dask array.

        In-plane chunks come from :meth:`read_block_shape`, not from the TIFF's
        own tiles.  Native tiles are tiny -- a 4000-wide striped plane has 250
        of them -- and over an object store each one is its own HTTP request.
        Batching whole tiles up to a byte budget keeps every read tile-aligned,
        so no tile is fetched twice, while cutting the request count by one to
        two orders of magnitude.  It never changes the result, only the speed.

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
            block = self.read_block_shape(target_bytes)
            stacked = da.stack(
                [da.from_zarr(page, chunks=block) for page in self.pages]
            )
        return stacked.rechunk({0: z_chunk}) if z_chunk != 1 else stacked

    def to_xarray(self, name: str = "masks", z_chunk: int = 1,
                  target_bytes: int | None = None) -> "xr.DataArray":
        """Lazy ``xarray.DataArray`` with named ``(z, y, x)`` dims.

        Handy for napari (``viewer.add_labels(vol.to_xarray().data)``) and for
        any xarray-native downstream work.
        """
        import xarray as xr

        return xr.DataArray(
            self.to_dask(z_chunk=z_chunk, target_bytes=target_bytes),
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
        Path to the ``.tif``, or a URL (``s3://...``) when ``reader="virtual"``.
        URLs are passed through untouched; only local paths are resolved.
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
    # Resolve the URL *first*, and only ever with as_url.  The original built it
    # with Path(path).resolve().as_uri() unconditionally, which is what turned
    # every s3:// URL into a bogus local file:// one.
    url = as_url(path)

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
        local = Path(path).resolve()
        store, pages = _open_tifffile(local, sample)
        return MaskVolume(path=local, url=url, pages=pages, backend=backend,
                          store=store)

    if registry is None:
        if _is_remote(url):
            raise ValueError(
                f"{url!r} is remote and needs an ObjectStoreRegistry; build one "
                "with make_registry(bucket, region=...) and pass it as registry=."
            )
        # a lone local file still needs a store, rooted at its directory
        registry, _ = local_registry(Path(path).resolve().parent)

    store, pages = _open_virtual(url, registry, ifd_layout, sample)
    return MaskVolume(path=url, url=url, pages=pages, backend=backend, store=store)


def _registry_for(url: str, registry: ObjectStoreRegistry | None) -> ObjectStoreRegistry:
    """The registry to use, building a local one when a local path came bare."""
    if registry is not None:
        return registry
    if _is_remote(url):
        raise ValueError(
            f"{url!r} is remote and needs an ObjectStoreRegistry; build one with "
            "make_registry(bucket, region=...) and pass it as registry=."
        )
    return local_registry(Path(url.removeprefix("file://")).parent)[0]


def _z_index(name: str) -> int:
    """The z in ``DAPI_decon_z10.tif``.

    Sorting the filenames as text puts ``z10`` before ``z9``, which silently
    shuffles the stack; the number is what orders it.
    """
    match = re.search(r"z(\d+)", Path(name).stem)
    return int(match.group(1)) if match else 0


def open_dapi_volume(
    fov_url: str,
    registry: ObjectStoreRegistry | None = None,
    pattern: str = DAPI_PATTERN,
) -> tuple[MaskVolume, list[str]]:
    """Open a directory of one-plane DAPI TIFFs as one lazy ``(z, y, x)`` volume.

    DAPI is stored the other way round from the masks: where a mask volume is
    one file with an IFD per plane, DAPI is one *file* per plane.  Same parser
    either way -- ``VirtualTIFF(ifd=0)`` on each file -- and the pages are
    stacked into the same :class:`MaskVolume` handle, so everything downstream
    (chunking, ``to_dask``, ``to_xarray``) is shared.

    Returns the volume and the filenames it stacked, in z order.
    """
    fov_url = as_url(fov_url)
    registry = _registry_for(fov_url, registry)

    names = sorted(list_files(fov_url, registry, pattern), key=_z_index)
    if not names:
        raise FileNotFoundError(
            f"no DAPI files matching {pattern!r} under {fov_url}"
        )

    pages: list[zarr.Array] = []
    for name in names:
        manifest_store = VirtualTIFF(ifd=0)(join_url(fov_url, name), registry)
        group = zarr.open_group(store=manifest_store, mode="r")
        pages.extend(_page_arrays(group, 0))

    volume = MaskVolume(path=fov_url, url=fov_url, pages=tuple(pages),
                        backend="virtual", store=None)
    return volume, names


def load_masks(
    url: str | Path,
    registry: ObjectStoreRegistry | None = None,
    *,
    reader: Reader = "auto",
    target_bytes: int | None = None,
) -> np.ndarray:
    """Read a mask volume into memory, streamed in chunks on the way.

    A drop-in for the eager ``tifffile.imread``: same ``(z, y, x)`` array, same
    dtype, 2D promoted to 3D.  What changed is the journey -- the bytes arrive
    as ``target_bytes``-sized blocks rather than one whole-file read, which is
    what makes an object store viable.  The measurement code downstream still
    gets a plain resident NumPy array, so none of it has to change.
    """
    volume = open_mask_volume(url, registry, reader=reader)
    masks = np.asarray(volume.to_dask(target_bytes=target_bytes).compute())
    return masks[np.newaxis] if masks.ndim == 2 else masks


def load_dapi(
    fov_url: str,
    registry: ObjectStoreRegistry | None = None,
    pattern: str = DAPI_PATTERN,
    *,
    target_bytes: int | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Read a DAPI stack into memory.  Returns ``(array, filenames)``, as before.

    ``fov_url`` is the FOV *directory*, not a glob: object storage has no glob,
    so the pattern is applied to a listing instead.
    """
    volume, names = open_dapi_volume(fov_url, registry, pattern)
    stack = np.asarray(volume.to_dask(target_bytes=target_bytes).compute())
    return stack, names
