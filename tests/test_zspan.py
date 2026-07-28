"""Tests for the z-span analysis.

The central guarantee is that the streaming implementation returns exactly what
the original ``scipy.ndimage.find_objects`` version returned -- so the reference
implementation lives here and the streaming one is checked against it.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest
import tifffile
from scipy import ndimage as ndi

matplotlib.use("Agg")

from zspan import (  # noqa: E402
    add_variant_column,
    check_z_span,
    find_mask_files,
    open_mask_volume,
    plot_span_distribution,
    plot_variant_summary,
    scan_segmentations,
    summarise_z_spans,
)

REFERENCE_COLUMNS = ["label", "z_start", "z_end", "z_span", "spans_multiple_layers"]


def reference_check_z_span(mask_volume, layer_span_cutoff: int = 1) -> pd.DataFrame:
    """The original in-memory implementation, kept as the oracle."""
    objects = ndi.find_objects(mask_volume)
    results = []
    for label, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        z = slices[0]
        span = z.stop - z.start
        results.append(
            {
                "label": label,
                "z_start": z.start,
                "z_end": z.stop - 1,
                "z_span": span,
                "spans_multiple_layers": span > layer_span_cutoff,
            }
        )
    return pd.DataFrame(results)


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("label").reset_index(drop=True)[REFERENCE_COLUMNS]


@pytest.fixture
def volume() -> np.ndarray:
    """A volume with gaps, an empty plane, and labels touching the edges."""
    rng = np.random.default_rng(11)
    data = rng.integers(0, 9, size=(8, 32, 29)).astype(np.int32)
    data[rng.random(data.shape) < 0.55] = 0
    data[4] = 0  # a plane with nothing on it
    data[0, 0, 0] = 7  # label present far from its other occurrences
    return data


@pytest.fixture
def tiff_tree(tmp_path):
    """A miniature two-variant segmentation tree of tiled multi-page TIFFs."""
    rng = np.random.default_rng(3)
    for variant, thickness in (("decon", 1), ("raw", 3)):
        for region in ("region_A", "region_B"):
            for fov in ("fov_01", "fov_02"):
                data = np.zeros((6, 64, 64), np.uint16)
                for label in range(1, 13):
                    z0 = int(rng.integers(0, 6 - thickness + 1))
                    y0 = int(rng.integers(0, 56))
                    x0 = int(rng.integers(0, 56))
                    data[z0 : z0 + thickness, y0 : y0 + 6, x0 : x0 + 6] = label
                path = tmp_path / variant / region / fov / "masks.tif"
                path.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(path, data, tile=(16, 16), photometric="minisblack")
    return tmp_path


# --------------------------------------------------------------------------- #
# equivalence with the original implementation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("block_shape", [None, (8, 8), (7, 13), (32, 29), (1, 1)])
def test_matches_reference_for_every_block_shape(volume, block_shape):
    """Tiling must not change the answer -- min/max over z is order-independent."""
    got = check_z_span(volume, block_shape=block_shape)
    assert normalise(got).equals(normalise(reference_check_z_span(volume)))


@pytest.mark.parametrize("cutoff", [0, 1, 2, 5])
def test_cutoff_matches_reference(volume, cutoff):
    got = check_z_span(volume, cutoff)
    assert normalise(got).equals(normalise(reference_check_z_span(volume, cutoff)))


def test_matches_reference_on_random_volumes():
    for seed in range(8):
        rng = np.random.default_rng(seed)
        data = rng.integers(0, 15, size=(7, 20, 23)).astype(np.int32)
        data[rng.random(data.shape) < 0.6] = 0
        assert normalise(check_z_span(data, block_shape=(6, 9))).equals(
            normalise(reference_check_z_span(data))
        )


def test_sparse_and_non_contiguous_labels():
    """Labels need not start at 1 or be contiguous."""
    data = np.zeros((4, 8, 8), np.int32)
    data[1:3, 0:2, 0:2] = 5
    data[2, 4:6, 4:6] = 99
    got = check_z_span(data)
    assert normalise(got).equals(normalise(reference_check_z_span(data)))
    assert set(got["label"]) == {5, 99}


# --------------------------------------------------------------------------- #
# the columns the reference implementation did not have
# --------------------------------------------------------------------------- #


def test_z_gap_is_reported():
    """n_planes < z_span flags a label with a hole in z -- a likely bad merge."""
    data = np.zeros((6, 10, 10), np.int32)
    data[0, 2:5, 2:5] = 1
    data[5, 2:5, 2:5] = 1
    data[2:4, 7:9, 7:9] = 2  # solid, no gap
    got = check_z_span(data).set_index("label")

    assert got.loc[1, "z_span"] == 6 and got.loc[1, "n_planes"] == 2
    assert got.loc[2, "z_span"] == 2 and got.loc[2, "n_planes"] == 2
    assert summarise_z_spans(got.reset_index())["n_with_z_gaps"] == 1


def test_voxel_counts_survive_tiling():
    """A label straddling tile borders must not be double-counted."""
    data = np.zeros((2, 16, 16), np.int32)
    data[0, 6:10, 6:10] = 1  # sits across the (8, 8) tile boundary
    for block_shape in (None, (8, 8), (4, 4), (16, 16)):
        got = check_z_span(data, block_shape=block_shape).set_index("label")
        assert got.loc[1, "n_voxels"] == 16
        assert got.loc[1, "n_planes"] == 1


def test_background_none_keeps_zero():
    data = np.zeros((3, 4, 4), np.int32)
    data[1, 0, 0] = 1
    assert 0 not in set(check_z_span(data)["label"])
    assert 0 in set(check_z_span(data, background=None)["label"])


def test_empty_volume_returns_empty_frame():
    got = check_z_span(np.zeros((3, 4, 4), np.int32))
    assert got.empty
    assert list(got.columns)[:4] == ["label", "z_start", "z_end", "z_span"]
    assert summarise_z_spans(got)["n_labels"] == 0


def test_negative_labels_rejected():
    data = np.zeros((2, 4, 4), np.int32)
    data[0, 0, 0] = -1
    with pytest.raises(ValueError, match="negative label"):
        check_z_span(data)


def test_rejects_non_3d_input():
    with pytest.raises(ValueError, match="3D"):
        check_z_span(np.zeros((4, 4), np.int32))


# --------------------------------------------------------------------------- #
# lazy loading
# --------------------------------------------------------------------------- #


def test_lazy_volume_matches_tifffile(tiff_tree):
    path = tiff_tree / "raw" / "region_A" / "fov_01" / "masks.tif"
    expected = tifffile.imread(path)
    lazy = open_mask_volume(path)

    assert lazy.shape == expected.shape
    assert lazy.chunks == (1, 16, 16)  # the TIFF's own tiles
    np.testing.assert_array_equal(lazy.read_plane(2), expected[2])
    np.testing.assert_array_equal(lazy.read_plane(3, slice(5, 40), slice(9, 33)),
                                  expected[3, 5:40, 9:33])


def test_lazy_and_eager_agree(tiff_tree):
    path = tiff_tree / "raw" / "region_B" / "fov_02" / "masks.tif"
    lazy = check_z_span(open_mask_volume(path))
    eager = reference_check_z_span(tifffile.imread(path).astype(np.int32))
    assert normalise(lazy).equals(normalise(eager))


@pytest.mark.parametrize("block_shape", [None, (16, 16), (32, 64), (64, 64)])
def test_plane_blocks_tile_the_plane_exactly(tiff_tree, block_shape):
    """Whatever the block size, the blocks must cover the plane once each."""
    path = tiff_tree / "decon" / "region_A" / "fov_01" / "masks.tif"
    lazy = open_mask_volume(path)
    _, ny, nx = lazy.shape

    blocks = list(lazy.iter_plane_blocks(0, block_shape))
    assert sum(b.size for b in blocks) == ny * nx

    # reassembling the blocks must reproduce the plane
    per_row = -(-nx // (block_shape[1] if block_shape else lazy.read_block_shape()[1]))
    rows = [blocks[i : i + per_row] for i in range(0, len(blocks), per_row)]
    rebuilt = np.vstack([np.hstack(row) for row in rows])
    np.testing.assert_array_equal(rebuilt, lazy.read_plane(0))


def test_read_block_shape_batches_chunks_to_a_byte_budget(tiff_tree):
    """The default read batches whole chunks instead of fetching one at a time.

    Reading a single native chunk per call is a throughput trap: each call
    crosses zarr's sync-to-async bridge, which costs far more than the data.
    """
    lazy = open_mask_volume(tiff_tree / "decon" / "region_A" / "fov_01" / "masks.tif")
    _, chunk_y, _ = lazy.chunks
    _, ny, nx = lazy.shape

    by, bx = lazy.read_block_shape(target_bytes=1 << 20)
    assert bx == nx  # full width
    assert by % chunk_y == 0 or by == ny  # whole chunks only
    assert by > chunk_y  # strictly bigger than one native chunk

    # a tiny budget still yields at least one whole chunk row, never zero
    small_y, _ = lazy.read_block_shape(target_bytes=1)
    assert small_y == chunk_y

    # a budget past the plane size clamps rather than overshooting
    big_y, _ = lazy.read_block_shape(target_bytes=1 << 40)
    assert big_y == ny


def test_block_size_never_changes_the_answer(tiff_tree):
    """The whole point of the byte budget: it is a speed knob, not a result knob."""
    path = tiff_tree / "raw" / "region_B" / "fov_01" / "masks.tif"
    lazy = open_mask_volume(path)
    baseline = check_z_span(lazy, block_shape=lazy.chunks[1:])
    for target in (1, 1 << 10, 1 << 20, 1 << 30):
        pd.testing.assert_frame_equal(baseline, check_z_span(lazy, target_bytes=target))


def test_xarray_and_dask_round_trip(tiff_tree):
    path = tiff_tree / "raw" / "region_A" / "fov_02" / "masks.tif"
    lazy = open_mask_volume(path)
    array = lazy.to_xarray()

    assert array.dims == ("z", "y", "x")
    assert array.shape == lazy.shape
    np.testing.assert_array_equal(array.values, tifffile.imread(path))
    # the same statistics whether fed the volume, the DataArray, or numpy
    assert normalise(check_z_span(array)).equals(normalise(check_z_span(lazy)))
    assert normalise(check_z_span(array.values)).equals(normalise(check_z_span(lazy)))


def test_flat_layout_gives_the_same_volume(tiff_tree):
    path = tiff_tree / "decon" / "region_B" / "fov_01" / "masks.tif"
    nested = open_mask_volume(path, ifd_layout="nested")
    flat = open_mask_volume(path, ifd_layout="flat")
    assert nested.shape == flat.shape
    np.testing.assert_array_equal(nested.read_plane(1), flat.read_plane(1))


# --------------------------------------------------------------------------- #
# scanning a tree
# --------------------------------------------------------------------------- #


def test_scan_finds_and_scores_every_volume(tiff_tree):
    result = scan_segmentations(
        tiff_tree, "**/masks.tif", level_names=("variant", "region", "fov")
    )
    assert len(find_mask_files(tiff_tree, "**/masks.tif")) == 8
    assert len(result.summary) == 8
    assert not result.failures
    assert {"variant", "region", "fov", "pct_multi_layer", "n_labels"} <= set(
        result.summary.columns
    )
    # thickness 1 vs 3 in the fixture must separate cleanly
    by_variant = result.summary.groupby("variant")["pct_multi_layer"].mean()
    assert by_variant["decon"] == 0.0
    assert by_variant["raw"] == 100.0


def test_scan_threads_match_serial(tiff_tree):
    kwargs = dict(pattern="**/masks.tif", level_names=("variant", "region", "fov"))
    serial = scan_segmentations(tiff_tree, max_workers=1, **kwargs)
    threaded = scan_segmentations(tiff_tree, max_workers=4, **kwargs)
    columns = ["relpath", "n_labels", "pct_multi_layer", "mean_z_span"]
    pd.testing.assert_frame_equal(serial.summary[columns], threaded.summary[columns])


def test_scan_labels_carry_level_columns(tiff_tree):
    result = scan_segmentations(
        tiff_tree, "**/masks.tif", level_names=("variant", "region", "fov")
    )
    assert {"variant", "region", "fov", "z_span", "label"} <= set(result.labels.columns)
    assert len(result.labels) == result.summary["n_labels"].sum()


def test_scan_without_labels(tiff_tree):
    result = scan_segmentations(
        tiff_tree, "**/masks.tif", level_names=("variant",), keep_labels=False
    )
    assert result.labels.empty
    assert len(result.summary) == 8


def test_level_names_align_to_trailing_directories(tiff_tree):
    """Naming only region/fov leaves the leading variant dir as level_0."""
    result = scan_segmentations(tiff_tree, "**/masks.tif", level_names=("region", "fov"))
    assert {"level_0", "region", "fov"} <= set(result.summary.columns)
    assert set(result.summary["level_0"]) == {"decon", "raw"}


def test_too_many_level_names_is_an_error(tiff_tree):
    with pytest.raises(ValueError, match="level names"):
        scan_segmentations(tiff_tree, "**/masks.tif", level_names=list("abcdef"))


def test_default_pattern_matches_region_fov_layout(tiff_tree):
    result = scan_segmentations(tiff_tree / "decon")
    assert len(result.summary) == 4
    assert set(result.summary["region"]) == {"region_A", "region_B"}


def test_unreadable_volume_is_recorded_not_raised(tiff_tree):
    broken = tiff_tree / "decon" / "region_A" / "fov_03"
    broken.mkdir(parents=True)
    (broken / "masks.tif").write_bytes(b"not a tiff")

    result = scan_segmentations(tiff_tree, "**/masks.tif", level_names=("region", "fov"))
    assert len(result.failures) == 1
    assert result.failures[0][0].name == "masks.tif"
    assert len(result.summary) == 8  # the good ones still came through


def test_missing_root_and_empty_match(tmp_path):
    with pytest.raises(NotADirectoryError):
        scan_segmentations(tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        scan_segmentations(tmp_path, "**/nothing.tif")


def test_add_variant_column(tiff_tree):
    result = scan_segmentations(
        tiff_tree, "**/masks.tif", level_names=("variant", "region", "fov")
    )
    joined = add_variant_column(result.summary, ["variant", "region"], name="combo")
    assert "decon / region_A" in set(joined["combo"])
    with pytest.raises(ValueError, match="none of the requested"):
        add_variant_column(result.summary, ["missing"])


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #


def test_plots_render(tiff_tree):
    result = scan_segmentations(
        tiff_tree, "**/masks.tif", level_names=("variant", "region", "fov")
    )
    summary, labels = result.summary, result.labels

    assert plot_variant_summary(summary, group="variant").axes
    assert plot_span_distribution(labels, group="variant").axes


def test_plot_rejects_unknown_columns(tiff_tree):
    result = scan_segmentations(tiff_tree, "**/masks.tif", level_names=("region", "fov"))
    with pytest.raises(KeyError, match="not in summary"):
        plot_variant_summary(result.summary, group="nope")
    with pytest.raises(KeyError, match="not in labels"):
        plot_span_distribution(result.labels, group="nope")


def test_distribution_warns_past_the_colour_ceiling(tiff_tree):
    result = scan_segmentations(tiff_tree, "**/masks.tif", level_names=("region", "fov"))
    labels = result.labels.assign(many=lambda d: "g" + (d.index % 11).astype(str))
    with pytest.warns(UserWarning, match="colour ceiling"):
        plot_span_distribution(labels, group="many")


# --------------------------------------------------------------------------- #
# read-size tuning
# --------------------------------------------------------------------------- #


def test_tune_read_size_reports_every_candidate(tiff_tree):
    from zspan import tune_read_size

    lazy = open_mask_volume(tiff_tree / "raw" / "region_A" / "fov_01" / "masks.tif")
    targets = (1 << 12, 1 << 16, 1 << 20)
    table = tune_read_size(lazy, targets=targets, repeats=1)

    assert list(table["target_bytes"]) == list(targets)
    assert {"block_shape", "reads_per_volume", "min_s", "median_s",
            "spread_pct", "vs_best"} <= set(table.columns)
    assert (table["min_s"] > 0).all()
    assert table["vs_best"].min() == pytest.approx(1.0)
    # bigger budgets never mean more round-trips
    assert list(table["reads_per_volume"]) == sorted(
        table["reads_per_volume"], reverse=True
    )


def test_tune_read_size_blocks_stay_chunk_aligned(tiff_tree):
    from zspan import tune_read_size

    lazy = open_mask_volume(tiff_tree / "raw" / "region_A" / "fov_01" / "masks.tif")
    _, chunk_y, _ = lazy.chunks
    _, ny, nx = lazy.shape
    for by, bx in tune_read_size(lazy, targets=(1 << 12, 1 << 20), repeats=1)["block_shape"]:
        assert bx == nx
        assert by % chunk_y == 0 or by == ny


def test_best_read_size_picks_the_middle_of_the_plateau():
    """A flat plateau must not resolve to its noisy edge."""
    from zspan import best_read_size

    # 2..32 MiB are all within noise; 1 and 64 are clearly worse
    table = pd.DataFrame(
        {
            "target_bytes": [mib << 20 for mib in (1, 2, 4, 8, 16, 32, 64)],
            "min_s": [1.70, 1.39, 1.35, 1.31, 1.34, 1.37, 1.63],
        }
    )
    assert best_read_size(table) == 8 << 20  # middle of {2,4,8,16,32}

    # a genuine single winner is still respected
    sharp = pd.DataFrame(
        {"target_bytes": [1 << 20, 2 << 20, 4 << 20], "min_s": [9.0, 1.0, 9.0]}
    )
    assert best_read_size(sharp) == 2 << 20


def test_best_read_size_accepts_a_volume_and_falls_back(tiff_tree):
    from zspan import DEFAULT_READ_BYTES, best_read_size

    path = tiff_tree / "raw" / "region_A" / "fov_01" / "masks.tif"
    chosen = best_read_size(path, targets=(1 << 12, 1 << 20), repeats=1)
    assert chosen in (1 << 12, 1 << 20)
    assert best_read_size(pd.DataFrame(columns=["target_bytes", "min_s"])) == DEFAULT_READ_BYTES


def test_tune_read_size_rejects_zero_repeats(tiff_tree):
    from zspan import tune_read_size

    with pytest.raises(ValueError, match="repeats"):
        tune_read_size(
            tiff_tree / "raw" / "region_A" / "fov_01" / "masks.tif", repeats=0
        )
