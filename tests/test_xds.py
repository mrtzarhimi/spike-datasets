from functools import partial

import h5py
import numpy as np
import pytest

from spike_datasets.binning import materialize, plan_view
from spike_datasets.types import RawSpan, Source
from spike_datasets.xds import discover_sessions, inspect_session, read_span
from tests.fixtures.xds import write_xds


@pytest.fixture
def sources(tmp_path):
    chewie = write_xds(tmp_path / "Chewie_20160928_001.mat")
    jango = write_xds(
        tmp_path / "Jango_20150730_001.mat",
        units=("elec4", "elec1", "elec3", "elec7🐒"),
        include_force=True,
    )
    return (
        Source(chewie, "chewie", chewie.stem, "synthetic"),
        Source(jango, "jango", jango.stem, "synthetic"),
    )


def test_discovers_each_dataset_in_date_and_recording_order(tmp_path):
    for name in (
        "Chewie_20160929_001.mat",
        "Jango_20151102_001.mat",
        "Chewie_20160928_10.mat",
        "Chewie_20150928_003.mat",
        "other_20160928_001.mat",
        "Jango_20150730_002.mat",
        "Chewie_20160928_2.mat",
        "Jango_20150730_001.mat",
        "Chewie_20160928_notes.mat",
    ):
        (tmp_path / name).touch()
    (tmp_path / "Chewie_20160927_001.mat").mkdir()

    for dataset, expected in (
        (
            "chewie",
            ("Chewie_20160928_2", "Chewie_20160928_10", "Chewie_20160929_001"),
        ),
        (
            "jango",
            ("Jango_20150730_001", "Jango_20150730_002", "Jango_20151102_001"),
        ),
    ):
        assert discover_sessions(tmp_path, dataset, revision="fixture-v1") == tuple(
            Source(tmp_path / f"{stem}.mat", dataset, stem, "fixture-v1")
            for stem in expected
        )


def test_reads_channel_major_counts_and_explicit_target(sources):
    chewie, jango = sources
    chewie_meta, jango_meta = map(inspect_session, sources)
    assert chewie_meta.target_field == "curs_v"
    assert chewie_meta.target_units == "recorded XDS units"
    assert chewie_meta.unit_ids == ("elec9", "elec2", "elec7")
    assert jango_meta.target_field == "force"
    assert jango_meta.target_units == "fixture force units"
    assert jango_meta.unit_ids == ("elec4", "elec1", "elec3", "elec7🐒")

    chewie_block = read_span(chewie, RawSpan(2, 5), with_behavior=True)
    np.testing.assert_array_equal(chewie_block.counts, [[2, 12, 22], [3, 13, 23], [4, 14, 24]])
    assert chewie_block.counts.dtype == np.uint16
    np.testing.assert_array_equal(chewie_block.targets, [[2.5, -2.5], [3.5, -3.5], [4.5, -4.5]])

    jango_block = read_span(jango, RawSpan(2, 5), with_behavior=True)
    np.testing.assert_array_equal(
        jango_block.counts, [[2, 12, 22, 32], [3, 13, 23, 33], [4, 14, 24, 34]]
    )
    np.testing.assert_array_equal(jango_block.targets, [[102, 202], [103, 203], [104, 204]])


def test_inspects_timebase_and_keeps_original_trial_rows(sources):
    chewie, _ = sources
    meta = inspect_session(chewie)

    assert (meta.session_id, meta.dataset, meta.revision) == (
        chewie.session_id,
        chewie.dataset,
        chewie.revision,
    )
    assert meta.native_bin_width_s == 0.001
    np.testing.assert_allclose(
        meta.native_timestamps, [0, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007]
    )
    assert [(trial.row_id, trial.start_s, trial.end_s) for trial in meta.trials] == [
        (0, 0.0, 0.002),
        (2, 0.004, 0.008),
    ]
    assert meta.invalid_trial_ids == (1,)
    assert [(trial.cue, trial.outcome, trial.direction) for trial in meta.trials] == [
        ("0.001", "R", "0.0"),
        ("0.005", "A", "180.0"),
    ]


@pytest.fixture
def signal_slices(monkeypatch):
    slices = []
    getitem = h5py.Dataset.__getitem__

    def record_slice(dataset, key):
        if dataset.name in ("/xds/spike_counts", "/xds/curs_v", "/xds/force"):
            slices.append((dataset.name, key))
        return getitem(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", record_slice)
    return slices


def test_inspection_reads_only_metadata(sources, signal_slices):
    for source in sources:
        inspect_session(source)
    assert signal_slices == []


def test_behavior_disabled_reads_only_requested_counts(sources, signal_slices):
    _, jango = sources
    block = read_span(jango, RawSpan(2, 5), with_behavior=False)
    assert block.targets is None
    assert signal_slices == [("/xds/spike_counts", (slice(None), slice(2, 5)))]


def test_behavior_reads_only_requested_target_and_counts(sources, signal_slices):
    _, jango = sources
    read_span(jango, RawSpan(2, 5), with_behavior=True)
    assert signal_slices == [
        ("/xds/spike_counts", (slice(None), slice(2, 5))),
        ("/xds/force", (slice(None), slice(2, 5))),
    ]


@pytest.mark.parametrize(
    "field", ["time_frame", "bin_width", "unit_names", "spike_counts", "force", "trial_end_time"]
)
def test_missing_required_field_identifies_source(sources, field):
    _, jango = sources
    with h5py.File(jango.path, "r+") as file:
        del file["xds"][field]
    with pytest.raises(ValueError, match=rf"{jango.session_id}.*{field}"):
        inspect_session(jango)


@pytest.mark.parametrize("ref_target", [None, "/xds/spike_counts", "/#refs#"])
def test_invalid_unit_references_identify_source(sources, signal_slices, ref_target):
    chewie, _ = sources
    with h5py.File(chewie.path, "r+") as file:
        names = file["xds/unit_names"]
        names[0, 0] = h5py.Reference() if ref_target is None else file[ref_target].ref

    with pytest.raises(ValueError, match=rf"{chewie.session_id}.*unit_names.*reference"):
        inspect_session(chewie)
    assert signal_slices == []


@pytest.mark.parametrize("field,shape", [("spike_counts", (3, 7)), ("curs_v", (2, 7))])
@pytest.mark.parametrize("read", [False, True])
def test_inconsistent_matrix_time_axes_identify_source(sources, signal_slices, field, shape, read):
    chewie, _ = sources
    with h5py.File(chewie.path, "r+") as file:
        del file["xds"][field]
        file["xds"].create_dataset(field, shape, dtype="float64")

    with pytest.raises(ValueError, match=rf"{chewie.session_id}.*{field}.*time"):
        if read:
            read_span(chewie, RawSpan(2, 5), with_behavior=True)
        else:
            inspect_session(chewie)
    assert signal_slices == []


def test_rejects_nonuniform_timestamps(sources):
    chewie, _ = sources
    with h5py.File(chewie.path, "r+") as file:
        file["xds/time_frame"][0, -1] = 0.008
    with pytest.raises(ValueError, match=rf"{chewie.session_id}.*time_frame"):
        inspect_session(chewie)


@pytest.mark.parametrize("start", [1.0, 4096.00024])
def test_inspects_regular_float32_timestamps(tmp_path, start):
    timestamps = (start + np.arange(8) * 0.001).astype(np.float32)
    path = write_xds(tmp_path / "Chewie_20160928_001.mat", timestamps=timestamps)
    source = Source(path, "chewie", path.stem, "synthetic")

    meta = inspect_session(source)

    assert meta.native_bin_width_s == 0.001
    assert meta.native_timestamps.dtype == np.float32
    np.testing.assert_array_equal(meta.native_timestamps, timestamps)


def test_inspects_64_ticks_with_float32_timestamps_and_width(tmp_path):
    timestamps = (np.arange(64) * 0.001).astype(np.float32)
    path = write_xds(
        tmp_path / "Chewie_20160928_001.mat", timestamps=timestamps, bin_width=np.float32(0.001)
    )
    meta = inspect_session(Source(path, "chewie", path.stem, "synthetic"))

    np.testing.assert_array_equal(meta.native_timestamps, timestamps)
    assert np.float32(meta.native_bin_width_s) == np.float32(0.001)
    assert type(meta.native_bin_width_s) is float


@pytest.mark.parametrize(
    "mode,width,spans",
    [
        ("trial", 0.02, (((2, 22), (22, 42)), ((52, 72), (72, 92), (92, 112)))),
        ("fixed", 0.02, (((0, 20), (20, 40)), ((60, 80), (80, 100), (100, 120)))),
        ("trial", 0.04, (((2, 42),), ((52, 92),))),
        ("fixed", 0.04, (((0, 40),), ((80, 120),))),
    ],
)
def test_materializes_float32_xds_grid_with_exact_group_spans(tmp_path, mode, width, spans):
    timestamps = (np.arange(120) * 0.001).astype(np.float32)
    path = write_xds(
        tmp_path / "Chewie_20160928_001.mat", timestamps=timestamps, bin_width=np.float32(0.001)
    )
    with h5py.File(path, "r+") as file:
        file["xds/trial_start_time"][0] = [0.0015, np.nan, 0.0515]
        file["xds/trial_end_time"][0] = [0.0445, np.nan, 0.1195]
    source = Source(path, "chewie", path.stem, "synthetic")

    view = plan_view(inspect_session(source), mode=mode, bin_width_s=width)
    data = materialize(view, [0, 1], partial(read_span, source), with_behavior=True)

    assert data.group_ids == (0, 1)
    assert tuple(block.group_id for block in data.blocks) == (0, 1)
    assert tuple(block.trial_id for block in data.blocks) == ((0, 2) if mode == "trial" else (None, None))
    for block, bounds in zip(data.blocks, spans):
        assert block.spans == tuple(RawSpan(start, stop) for start, stop in bounds)
        np.testing.assert_array_equal(block.counts[:, 0], [sum(range(a, b)) for a, b in bounds])


@pytest.mark.parametrize(
    "start,offsets",
    [
        (4096.0, np.array([0, 1, 3, 4, 5, 6, 7, 8]) * 0.001),
        (1.0, np.array([0, 1, 2.25, 3, 4, 5, 6, 7]) * 0.001),
        (1.0, np.array([0, 1, 1, 2, 3, 4, 5, 6]) * 0.001),
        (4096.0, np.arange(8) * 0.00146484375),
        (8192.0, np.arange(8) * 0.001),
    ],
    ids=["missing", "irregular", "repeated", "drift", "coarse"],
)
def test_rejects_invalid_float32_cadence(tmp_path, start, offsets):
    timestamps = (start + offsets).astype(np.float32)
    path = write_xds(tmp_path / "Chewie_20160928_001.mat", timestamps=timestamps)
    source = Source(path, "chewie", path.stem, "synthetic")

    with pytest.raises(ValueError, match=rf"{source.session_id}.*time_frame"):
        inspect_session(source)


@pytest.mark.parametrize("start", [0.0, 1.0])
def test_reports_grid_width_when_stored_width_has_lower_precision(tmp_path, start):
    timestamps = start + np.arange(64, dtype=np.float64) * 0.001
    path = write_xds(
        tmp_path / "Chewie_20160928_001.mat",
        timestamps=timestamps,
        bin_width=np.float32(0.001),
    )
    source = Source(path, "chewie", path.stem, "synthetic")

    meta = inspect_session(source)

    np.testing.assert_array_equal(meta.native_timestamps, timestamps)
    assert meta.native_bin_width_s == pytest.approx(0.001, rel=0, abs=1e-15)
    assert 0.04 / meta.native_bin_width_s == pytest.approx(40, rel=0, abs=1e-6)


def test_preserves_lower_precision_declared_width_when_endpoint_estimate_fails(tmp_path):
    width = np.float32(0.001)
    timestamps = np.arange(64, dtype=np.float64) * float(width)
    timestamps[48] -= 0.8e-9
    timestamps[63] += 0.8e-9
    path = write_xds(
        tmp_path / "Chewie_20160928_001.mat", timestamps=timestamps, bin_width=width
    )
    source = Source(path, "chewie", path.stem, "synthetic")

    meta = inspect_session(source)

    assert meta.native_bin_width_s == float(width)
    np.testing.assert_array_equal(meta.native_timestamps, timestamps)


@pytest.mark.parametrize(
    "offsets",
    [
        np.r_[np.arange(32), np.arange(33, 65)] * 0.001,
        np.arange(64) * 0.001 + np.where(np.arange(64) == 32, 0.00025, 0),
        np.arange(64) * 0.001 + np.minimum(np.arange(64), np.arange(63, -1, -1)) * 2e-10,
        np.arange(64) * 0.0011,
    ],
    ids=["missing", "irregular", "drift", "width-mismatch"],
)
def test_lower_precision_width_does_not_hide_malformed_cadence(tmp_path, offsets):
    path = write_xds(
        tmp_path / "Chewie_20160928_001.mat",
        timestamps=1.0 + offsets,
        bin_width=np.float32(0.001),
    )
    source = Source(path, "chewie", path.stem, "synthetic")

    with pytest.raises(ValueError, match=rf"{source.session_id}.*time_frame"):
        inspect_session(source)


def test_rejects_out_of_bounds_span_before_reading(sources, signal_slices):
    chewie, _ = sources
    with pytest.raises(ValueError, match=rf"{chewie.session_id}.*span"):
        read_span(chewie, RawSpan(6, 9), with_behavior=True)
    assert signal_slices == []


def test_reads_time_major_when_only_that_axis_matches_timestamps(sources):
    chewie, _ = sources
    with h5py.File(chewie.path, "r+") as file:
        xds = file["xds"]
        for name in ("spike_counts", "curs_v"):
            transposed = xds[name][...].T
            del xds[name]
            xds[name] = transposed

    block = read_span(chewie, RawSpan(2, 3), with_behavior=True)
    np.testing.assert_array_equal(block.counts, [[2, 12, 22]])
    np.testing.assert_array_equal(block.targets, [[2.5, -2.5]])


def test_missing_outcome_preserves_trial_row(sources):
    chewie, _ = sources
    with h5py.File(chewie.path, "r+") as file:
        file["xds/trial_result"][-1, 0] = 0

    meta = inspect_session(chewie)

    assert [(trial.row_id, trial.outcome) for trial in meta.trials] == [(0, "R"), (2, None)]


@pytest.mark.parametrize(
    "starts,valid_ids,invalid_ids",
    [
        ([0.0, 0.003, -0.001], (0,), (1, 2)),
        ([np.nan, 0.003, np.inf], (), (0, 1, 2)),
    ],
)
def test_retains_trailing_and_all_invalid_trial_ids(sources, starts, valid_ids, invalid_ids):
    chewie, _ = sources
    with h5py.File(chewie.path, "r+") as file:
        file["xds/trial_start_time"][0] = starts

    meta = inspect_session(chewie)

    assert tuple(trial.row_id for trial in meta.trials) == valid_ids
    assert meta.invalid_trial_ids == invalid_ids
