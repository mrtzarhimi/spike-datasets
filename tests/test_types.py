from dataclasses import replace

import numpy as np
import pytest

from spike_datasets.types import BinnedBlock, RawSpan, SessionMeta, TrialMeta


@pytest.mark.parametrize("bounds", [(4, 3), (2, 2)])
def test_raw_span_rejects_reversed_and_empty_bounds(bounds):
    with pytest.raises(ValueError):
        RawSpan(*bounds)


def test_binned_block_rejects_mismatched_axes():
    with pytest.raises(ValueError):
        BinnedBlock(
            counts=np.zeros((3, 2), dtype=np.int16),
            targets=np.zeros((2, 2), dtype=np.float32),
            target_mask=np.ones((2, 2), dtype=bool),
            spans=(RawSpan(0, 2), RawSpan(2, 4), RawSpan(4, 6)),
            timestamps=np.array([0.01, 0.03, 0.05], dtype=np.float64),
        )


def test_binned_block_preserves_count_and_time_dtypes():
    counts = np.array([[1, 0], [0, 3]], dtype=np.uint16)
    timestamps = np.array([0.01, 0.03], dtype=np.float32)
    block = BinnedBlock(
        counts=counts,
        targets=None,
        target_mask=None,
        spans=(RawSpan(0, 20), RawSpan(20, 40)),
        timestamps=timestamps,
    )

    assert block.counts.dtype == counts.dtype
    assert block.timestamps.dtype == timestamps.dtype
    assert block.counts is counts
    assert block.timestamps is timestamps


@pytest.fixture
def session_meta():
    return SessionMeta(
        dataset="chewie",
        session_id="Chewie_20160928_001",
        revision="fixture",
        native_bin_width_s=0.001,
        native_timestamps=np.array([0.0, 0.001]),
        unit_ids=("elec1",),
        target_field="curs_v",
        target_units="recorded XDS units",
        trials=(TrialMeta(row_id=2, start_s=0.0, end_s=0.002),),
    )


def test_session_metadata_accepts_distinct_invalid_trial_ids(session_meta):
    assert session_meta.invalid_trial_ids == ()
    assert replace(session_meta, invalid_trial_ids=(0, 4)).invalid_trial_ids == (0, 4)


@pytest.mark.parametrize(
    "ids,error",
    [
        ((-1,), ValueError),
        ((1, 1), ValueError),
        ((2,), ValueError),
        ((True,), TypeError),
        ((1.5,), TypeError),
        ([1], TypeError),
    ],
)
def test_session_metadata_rejects_malformed_invalid_trial_ids(session_meta, ids, error):
    with pytest.raises(error, match="invalid_trial_ids must"):
        replace(session_meta, invalid_trial_ids=ids)
