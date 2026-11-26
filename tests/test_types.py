import numpy as np
import pytest

from spike_datasets.types import BinnedBlock, RawSpan


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
