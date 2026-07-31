"""Compare segmentation runs by how far their masks span in z.

Masks are read straight out of multi-page TIFFs as lazy zarr arrays, so only the
tiles under analysis are ever in memory.  Two backends do that -- ``virtualizarr``
+ ``virtual_tiff``, which can front an object store, and ``tifffile``'s own
``aszarr`` store, which is faster on local files.  ``reader="auto"`` picks
between them; see :func:`zspan.resolve_reader`.

Typical use::

    from zspan import scan_segmentations, add_variant_column, plot_variant_summary

    result = scan_segmentations(
        "data/segmentations/cpdino", "**/masks.tif",
        level_names=("preprocessing", "model", "region", "fov"),
    )
    summary = add_variant_column(result.summary, ["preprocessing", "model"])
    plot_variant_summary(summary, group="variant")
"""

from .loading import (
    DEFAULT_READ_BYTES,
    MaskVolume,
    Reader,
    local_registry,
    open_mask_volume,
    resolve_reader,
)
from .metrics import (
    AREA_COLUMNS,
    SIZE_CHANGE_COLUMNS,
    Z_SPAN_COLUMNS,
    check_z_span,
    label_plane_areas,
    section_bounds,
    size_change_between_layers,
    summarise_z_spans,
)
from .plotting import (
    BLUES,
    CATEGORICAL,
    SEQUENTIAL_BLUE,
    apply_theme,
    plot_span_distribution,
    plot_variant_summary,
)
from .scan import (
    ScanResult,
    add_variant_column,
    find_mask_files,
    level_columns,
    scan_segmentations,
)
from .tuning import DEFAULT_TARGETS, best_read_size, tune_read_size

__version__ = "0.1.0"

__all__ = [
    "AREA_COLUMNS",
    "BLUES",
    "CATEGORICAL",
    "DEFAULT_READ_BYTES",
    "DEFAULT_TARGETS",
    "MaskVolume",
    "Reader",
    "ScanResult",
    "SEQUENTIAL_BLUE",
    "SIZE_CHANGE_COLUMNS",
    "Z_SPAN_COLUMNS",
    "add_variant_column",
    "apply_theme",
    "best_read_size",
    "check_z_span",
    "find_mask_files",
    "label_plane_areas",
    "level_columns",
    "local_registry",
    "open_mask_volume",
    "plot_span_distribution",
    "plot_variant_summary",
    "resolve_reader",
    "scan_segmentations",
    "section_bounds",
    "size_change_between_layers",
    "summarise_z_spans",
    "tune_read_size",
]
