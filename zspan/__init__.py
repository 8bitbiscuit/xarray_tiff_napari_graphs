"""Compare segmentation runs by how far their masks span in z.

Masks are read straight out of multi-page TIFFs as virtual zarr arrays
(``virtualizarr`` + ``virtual_tiff``), so only the tiles under analysis are ever
in memory.

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
    local_registry,
    open_mask_volume,
)
from .metrics import Z_SPAN_COLUMNS, check_z_span, summarise_z_spans
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
    scan_segmentations,
)
from .tuning import DEFAULT_TARGETS, best_read_size, tune_read_size

__version__ = "0.1.0"

__all__ = [
    "BLUES",
    "CATEGORICAL",
    "DEFAULT_READ_BYTES",
    "DEFAULT_TARGETS",
    "MaskVolume",
    "ScanResult",
    "SEQUENTIAL_BLUE",
    "Z_SPAN_COLUMNS",
    "add_variant_column",
    "apply_theme",
    "best_read_size",
    "check_z_span",
    "find_mask_files",
    "local_registry",
    "open_mask_volume",
    "plot_span_distribution",
    "plot_variant_summary",
    "scan_segmentations",
    "summarise_z_spans",
    "tune_read_size",
]
