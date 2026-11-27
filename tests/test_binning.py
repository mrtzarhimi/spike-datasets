from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
import pytest

from spike_datasets.binning import materialize, plan_view
from spike_datasets.types import RawSpan, SessionMeta, TrialMeta


def session(n_samples, bounds, *, timestamps=None, unit_ids=("u1",)):
    return SessionMeta(
        dataset="chewie",
        session_id="2016-09-28",
        revision="fixture",
        native_bin_width_s=0.001,
        native_timestamps=(
            np.arange(n_samples) * 0.001 if timestamps is None else timestamps
        ),
        unit_ids=unit_ids,
        target_field="curs_v",
        target_units="cm/s",
        trials=tuple(
            TrialMeta(row, start * 0.001, stop * 0.001, outcome="recorded")
            for row, start, stop in bounds
        ),
    )


@dataclass(frozen=True)
class NativeBlock:
    counts: NDArray[np.number]
    targets: NDArray[np.number] | None


class ArrayReader:
    def __init__(self, counts, targets=None):
        self.counts = np.asarray(counts)
        self.targets = None if targets is None else np.asarray(targets)
        self.calls = []

    def __call__(self, span, *, with_behavior):
        self.calls.append((span, with_behavior))
        return NativeBlock(
            self.counts[span.start:span.stop],
            self.targets[span.start:span.stop] if with_behavior else None,
        )


def test_trial_bins_sum_spikes_average_behavior_and_drop_tail():
    meta = session(5, [(12, 0, 5)])
    view = plan_view(meta, mode="trial", bin_width_s=0.002)
    read = ArrayReader([[1], [2], [0], [3], [5]], [[2], [4], [6], [8], [10]])

    data = materialize(view, [0], read, with_behavior=True)

    assert data.meta.unit_ids == ("u1",)
    assert data.group_ids == (0,)
    (block,) = data.blocks
    assert block.group_id == 0
    assert block.trial_id == 12
    assert block.trial is meta.trials[0]
    assert block.spans == (RawSpan(0, 2), RawSpan(2, 4))
    np.testing.assert_array_equal(block.timestamps, [0.0, 0.002])
    np.testing.assert_array_equal(block.counts, [[3], [3]])
    assert np.issubdtype(block.counts.dtype, np.integer)
    np.testing.assert_array_equal(block.targets, [[3.0], [7.0]])
    np.testing.assert_array_equal(block.target_mask, [[True], [True]])
    assert view.dropped_spans == (RawSpan(4, 5),)
    assert read.calls == [(RawSpan(0, 4), True)]


def test_behavior_mask_marks_axes_independently():
    view = plan_view(
        session(4, [(3, 0, 4)], unit_ids=("first", "second")),
        mode="trial",
        bin_width_s=0.002,
    )
    read = ArrayReader(
        [[1, 10], [2, 20], [3, 30], [4, 40]],
        [[np.nan, 2], [6, 4], [8, -np.inf], [10, 12]],
    )

    (block,) = materialize(view, [0], read, with_behavior=True).blocks

    np.testing.assert_array_equal(block.target_mask, [[False, True], [True, False]])
    np.testing.assert_array_equal(block.targets, [[np.nan, 3.0], [9.0, np.nan]])
    np.testing.assert_array_equal(block.counts, [[3, 30], [7, 70]])


def test_trial_bounds_use_left_insertion_and_anchor_to_first_tick():
    meta = session(7, [(0, 1.1, 4.1), (1, 5, 7)])
    view = plan_view(meta, mode="trial", bin_width_s=0.002)

    assert view.groups[0].blocks[0].spans == (RawSpan(2, 4),)
    assert view.groups[1].blocks[0].spans == (RawSpan(5, 7),)
    assert view.dropped_spans == (RawSpan(4, 5),)


def test_default_width_uses_twenty_native_samples_on_one_millisecond_grid():
    view = plan_view(session(40, [(0, 0, 40)]), mode="fixed")

    assert view.bin_width_s == 0.02
    assert view.groups[0].blocks[0].spans == (RawSpan(0, 20), RawSpan(20, 40))


@pytest.mark.parametrize("width", [0.0, -0.001, 0.0005, 0.0025, np.inf, np.nan])
def test_invalid_bin_width_is_rejected(width):
    with pytest.raises(ValueError, match="bin width"):
        plan_view(session(10, [(0, 0, 10)]), mode="trial", bin_width_s=width)


@pytest.mark.parametrize(
    "ticks,native_width,timestamp_dtype,requested_width",
    [
        (40, np.float32(0.001), np.float32, 0.04),
        (100, 0.001, np.float64, np.float32(0.1)),
    ],
)
def test_whole_bin_uses_native_and_requested_width_precision(
    ticks, native_width, timestamp_dtype, requested_width
):
    timestamps = (np.arange(ticks) * float(native_width)).astype(timestamp_dtype)
    meta = replace(
        session(ticks, [(0, 0, ticks)], timestamps=timestamps), native_bin_width_s=native_width
    )

    view = plan_view(meta, mode="fixed", bin_width_s=requested_width)

    assert view.groups[0].blocks[0].spans == (RawSpan(0, ticks),)


@pytest.mark.parametrize("width", [0.0400001, np.float32(0.10025)])
def test_fractional_widths_stay_invalid_on_a_float32_grid(width):
    native_width = np.float32(0.001)
    timestamps = (np.arange(40) * float(native_width)).astype(np.float32)
    meta = replace(
        session(40, [(0, 0, 40)], timestamps=timestamps), native_bin_width_s=native_width
    )
    with pytest.raises(ValueError, match="whole number"):
        plan_view(meta, mode="trial", bin_width_s=width)


def test_fractional_width_outside_validated_cadence_is_rejected():
    native_width = np.float32(0.001)
    timestamps = (np.arange(120) * float(native_width)).astype(np.float32)
    meta = replace(
        session(120, [(0, 0, 120)], timestamps=timestamps), native_bin_width_s=native_width
    )

    with pytest.raises(ValueError, match="whole number"):
        plan_view(meta, mode="fixed", bin_width_s=0.040000004)


def test_nonuniform_timebase_is_rejected():
    meta = session(4, [(0, 0, 4)], timestamps=np.array([0, 0.001, 0.0025, 0.003]))
    with pytest.raises(ValueError, match="uniform"):
        plan_view(meta, mode="trial", bin_width_s=0.002)


@pytest.mark.parametrize("mode", ["trial", "fixed"])
@pytest.mark.parametrize("start", [1.0, 4096.00024])
def test_regular_float32_timebase_accepts_stored_timestamp_roundoff(mode, start):
    timestamps = (start + np.arange(6) * 0.001).astype(np.float32)
    start_ms = int(start * 1000)
    meta = session(6, [(0, start_ms - 1, start_ms + 6)], timestamps=timestamps)
    view = plan_view(meta, mode=mode, bin_width_s=0.002)

    read = ArrayReader(np.ones((6, 1), dtype=np.int16))
    (block,) = materialize(view, [0], read, with_behavior=False).blocks

    assert block.spans == (RawSpan(0, 2), RawSpan(2, 4), RawSpan(4, 6))
    np.testing.assert_array_equal(block.timestamps, timestamps[::2])
    assert block.timestamps.dtype == np.float32


@pytest.mark.parametrize("ticks", [[0, 1, 3, 4], [0, 1, 2.25, 3], [0, 1, 1, 2]])
@pytest.mark.parametrize("start", [1, 4096])
def test_float32_timebase_rejects_missing_irregular_or_repeated_ticks(ticks, start):
    timestamps = (start + np.asarray(ticks) * 0.001).astype(np.float32)
    start_ms = start * 1000
    meta = session(4, [(0, start_ms - 1, start_ms + 6)], timestamps=timestamps)
    with pytest.raises(ValueError, match="uniform"):
        plan_view(meta, mode="fixed", bin_width_s=0.002)


def test_float32_timebase_rejects_sustained_drift_from_nominal_grid():
    timestamps = (4096 + np.arange(4) * 0.00146484375).astype(np.float32)
    meta = session(4, [(0, 4095999, 4096006)], timestamps=timestamps)

    with pytest.raises(ValueError, match="uniform"):
        plan_view(meta, mode="fixed", bin_width_s=0.002)


def test_coarse_float32_timebase_cannot_hide_missing_ticks():
    timestamps = (16384.0 + np.array([0, 2, 4, 6]) * 0.001).astype(np.float32)
    meta = session(4, [(0, 16383999, 16384007)], timestamps=timestamps)
    with pytest.raises(ValueError, match="uniform"):
        plan_view(meta, mode="fixed", bin_width_s=0.002)


def test_short_bridge_yields_common_groups_and_fixed_bins_respect_ownership():
    meta = session(
        26,
        [
            (7, 5, 9),
            (11, 3, 6),
            (3, 0, 4),
            (19, 9, 13),
            (22, 14, 16),
            (31, 17, 21),
            (34, 27, 30),
        ],
    )

    trial = plan_view(meta, mode="trial", bin_width_s=0.004)
    fixed = plan_view(meta, mode="fixed", bin_width_s=0.004)

    assert trial.group_ids == fixed.group_ids == (0, 1, 2)
    for view in (trial, fixed):
        assert tuple(group.trial_ids for group in view.groups) == ((3, 11, 7), (19,), (31,))
        assert tuple(group.eligible_trial_ids for group in view.groups) == ((3, 7), (19,), (31,))
        assert tuple(group.owned_span for group in view.groups) == (
            RawSpan(0, 9), RawSpan(9, 17), RawSpan(17, 26)
        )
        assert view.excluded_trial_ids == (11, 22, 34)

    assert trial.dropped_spans == (RawSpan(3, 6), RawSpan(14, 16))
    assert fixed.dropped_spans == (RawSpan(8, 12), RawSpan(16, 20), RawSpan(24, 26))
    assert tuple(group.blocks[0].spans for group in fixed.groups) == (
        (RawSpan(0, 4), RawSpan(4, 8)),
        (RawSpan(12, 16),),
        (RawSpan(20, 24),),
    )
    read = ArrayReader(np.ones((26, 1)))
    (block,) = materialize(fixed, [1], read, with_behavior=False).blocks
    assert block.trial_id is None
    np.testing.assert_array_equal(block.counts, [[4]])
    assert read.calls == [(RawSpan(12, 16), False)]


@pytest.mark.parametrize("mode", ["trial", "fixed"])
def test_malformed_source_ids_are_excluded_without_affecting_group_ownership(mode):
    meta = session(16, [(0, 0, 4), (2, 6, 7), (4, 8, 16)])
    baseline = plan_view(meta, mode=mode, bin_width_s=0.004)
    with_invalid = replace(meta, invalid_trial_ids=(1, 3, 5))

    view = plan_view(with_invalid, mode=mode, bin_width_s=0.004)

    assert view.excluded_trial_ids == (1, 2, 3, 5)
    assert view.groups == baseline.groups


def test_recorded_overlaps_can_bridge_trials_between_native_ticks():
    meta = session(12, [(0, 0, 4.2), (1, 4.1, 4.8), (2, 4.7, 9)])
    view = plan_view(meta, mode="trial", bin_width_s=0.004)

    (group,) = view.groups
    assert group.trial_ids == (0, 1, 2)
    assert group.eligible_trial_ids == (0, 2)
    assert view.excluded_trial_ids == (1,)


def test_leading_and_trailing_gaps_belong_to_first_and_last_groups():
    meta = session(16, [(4, 2, 6), (9, 8, 12)])
    view = plan_view(meta, mode="fixed", bin_width_s=0.004)

    assert tuple(group.owned_span for group in view.groups) == (
        RawSpan(0, 8), RawSpan(8, 16)
    )
    assert tuple(group.blocks[0].spans for group in view.groups) == (
        (RawSpan(0, 4), RawSpan(4, 8)), (RawSpan(8, 12), RawSpan(12, 16))
    )


@pytest.mark.parametrize(
    ("mode", "expected_rows"),
    [
        ("trial", [RawSpan(0, 4), RawSpan(10, 14)]),
        ("fixed", [RawSpan(0, 4), RawSpan(12, 16)]),
    ],
)
def test_materialize_only_reads_selected_groups_in_view_order(mode, expected_rows):
    meta = session(16, [(0, 0, 4), (1, 5, 9), (2, 10, 14)])
    view = plan_view(meta, mode=mode, bin_width_s=0.004)
    counts = np.ones((16, 1))
    counts[5:10] = np.nan
    read = ArrayReader(counts)

    data = materialize(view, [2, 0], read, with_behavior=False)

    assert data.group_ids == (0, 2)
    assert tuple(block.group_id for block in data.blocks) == (0, 2)
    assert all(block.targets is None and block.target_mask is None for block in data.blocks)
    assert read.calls == [(span, False) for span in expected_rows]


def test_group_with_no_fixed_bins_cannot_be_materialized_alone():
    meta = session(16, [(0, 0, 4), (1, 5, 9), (2, 9, 15)])
    view = plan_view(meta, mode="fixed", bin_width_s=0.004)
    assert view.groups[1].n_bins == 0
    read = ArrayReader(np.zeros((16, 1)))

    with pytest.raises(ValueError, match="no bins"):
        materialize(view, [1], read, with_behavior=False)

    assert read.calls == []


@pytest.mark.parametrize("mode", ["trial", "fixed"])
def test_no_eligible_trial_is_rejected(mode):
    with pytest.raises(ValueError, match="complete trial bin"):
        plan_view(session(16, [(0, 2, 3)]), mode=mode, bin_width_s=0.004)


@pytest.mark.parametrize("invalid_count", [-1, 0.5, np.inf, np.nan, 1j])
def test_materialize_rejects_invalid_spikes_in_selected_bins(invalid_count):
    view = plan_view(session(4, [(0, 0, 4)]), mode="trial", bin_width_s=0.002)
    read = ArrayReader([[1], [2], [invalid_count], [0]])

    with pytest.raises(ValueError, match="spike counts"):
        materialize(view, [0], read, with_behavior=False)


def test_materialize_rejects_bin_sum_that_cannot_be_stored_as_int64():
    view = plan_view(session(4, [(0, 0, 4)]), mode="trial", bin_width_s=0.004)
    read = ArrayReader(np.full((4, 1), 2**62, dtype=np.int64))

    with pytest.raises(ValueError, match="int64"):
        materialize(view, [0], read, with_behavior=False)


def test_one_sample_width_tolerates_float_roundoff():
    meta = session(2, [(0, 0, 2)])
    view = plan_view(meta, mode="trial", bin_width_s=np.nextafter(0.001, 0.0))

    assert view.groups[0].blocks[0].spans == (RawSpan(0, 1), RawSpan(1, 2))


def test_finite_behavior_stays_valid_if_intermediate_sum_would_overflow():
    view = plan_view(session(2, [(0, 0, 2)]), mode="trial", bin_width_s=0.002)
    limit = np.finfo(np.float64).max
    read = ArrayReader([[0], [0]], [[limit, -limit], [limit, -limit]])

    (block,) = materialize(view, [0], read, with_behavior=True).blocks

    np.testing.assert_array_equal(block.target_mask, [[True, True]])
    np.testing.assert_array_equal(block.targets, [[limit, -limit]])
