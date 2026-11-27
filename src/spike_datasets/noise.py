"""Bounded additive noise calibrated on clean fit counts."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, eq=False)
class NoiseState:
    """Fitted population standard deviations in unit order and a noise factor."""

    scales: NDArray[np.float64]
    factor: float


def fit_noise(clean_fit: Iterable[NDArray], *, factor: float = 0.02) -> NoiseState:
    """Estimate per-unit scales from the supplied sample-by-unit fit blocks."""
    if not 0.0 <= factor <= 0.05:
        raise ValueError("noise factor must be finite and between 0.0 and 0.05")

    n_units = None
    n_bins = 0
    for counts in clean_fit:
        _validate_counts(counts)
        if n_units is None:
            n_units = counts.shape[1]
        elif counts.shape[1] != n_units:
            raise ValueError("fit blocks must have the same number of units")
        if counts.shape[0] == 0:
            continue

        values = counts.astype(np.uint64, copy=False)
        if n_bins == 0:
            reference = values[0].copy()
            mean = np.zeros(n_units)
            sum_squared = np.zeros(n_units)

        # Take differences before converting to float to preserve variation at
        # large count offsets. Form unsigned magnitudes to avoid overflow.
        offsets = (np.maximum(values, reference) - np.minimum(values, reference)).astype(
            np.float64
        )
        np.negative(offsets, out=offsets, where=values < reference)
        block_mean = offsets.mean(axis=0)
        offsets -= block_mean
        np.square(offsets, out=offsets)

        # Merge centered sums of squares, weighting by the number of bins.
        block_size = counts.shape[0]
        total = n_bins + block_size
        shift = block_mean - mean
        weight = block_size / total
        sum_squared += offsets.sum(axis=0) + np.square(shift) * (n_bins * weight)
        mean += shift * weight
        n_bins = total

    if n_bins == 0:
        raise ValueError("noise fitting requires at least one clean fit bin")
    scales = np.sqrt(sum_squared / n_bins)
    scales.setflags(write=False)
    return NoiseState(scales, float(factor))


def augment_fit(counts: NDArray, state: NoiseState, *, rng: np.random.Generator) -> NDArray:
    """Return augmented float features from a caller-seeded fit-only generator."""
    _validate_counts(counts)
    if counts.shape[1] != state.scales.size:
        raise ValueError("counts must match the fitted number of units")

    draws = rng.standard_normal(counts.shape)
    np.clip(draws, -3.0, 3.0, out=draws)
    base = counts.astype(np.float64)
    features = base + draws * state.scales * state.factor
    wide = np.dtype(np.longdouble)
    wide_type = wide.type
    wide_scales = state.scales.astype(wide)
    wide_factor = wide_type(state.factor)
    wide_limits = wide_type(3.0) * wide_scales * wide_factor
    wide_limits = np.nextafter(np.nextafter(wide_limits, wide_type(0)), wide_type(0))
    exact_float64 = _exact_float64_integers(counts)
    rounded_delta = features.astype(wide) - base.astype(wide)
    over_bound = np.abs(rounded_delta) > wide_limits
    needs_wide = ~exact_float64 | over_bound

    if np.any(needs_wide):
        max_count = int(counts.max()) if counts.size else 0
        wide_integer_limit = 1 << (np.finfo(wide).nmant + 1)
        if max_count > wide_integer_limit:
            raise ValueError("counts cannot be represented exactly as floating-point features")
        if np.finfo(wide).nmant > np.finfo(np.float64).nmant:
            wide_base = counts.astype(wide)
            wide_candidate = wide_base + draws.astype(wide) * wide_scales * wide_factor
            wide_delta = wide_candidate - wide_base
            np.copyto(wide_candidate, wide_base, where=np.abs(wide_delta) > wide_limits)
            result = features.astype(wide)
            np.copyto(result, wide_candidate, where=needs_wide)
            return result
        if np.any(~exact_float64):
            raise ValueError("counts cannot be represented exactly as floating-point features")
        np.copyto(features, base, where=over_bound)
    return features


def _exact_float64_integers(counts: NDArray) -> NDArray[np.bool_]:
    """Return which nonnegative integer counts survive a float64 round trip."""
    values = counts.astype(np.uint64, copy=False)
    exact = values <= (1 << 53)
    for shift in range(1, 12):
        lower = 1 << (52 + shift)
        in_range = values >= lower
        if shift < 11:
            upper = 1 << (53 + shift)
            in_range &= values < upper
        exact |= in_range & ((values & ((1 << shift) - 1)) == 0)
    return exact


def _validate_counts(counts: NDArray) -> None:
    if counts.ndim != 2 or counts.shape[1] == 0:
        raise ValueError("counts must be a two-dimensional array with at least one unit")
    if counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise ValueError("counts must be a nonnegative integer array")
