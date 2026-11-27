import numpy as np
import pytest

from spike_datasets.noise import augment_fit, fit_noise


def test_fit_noise_uses_per_unit_population_std_and_bounds_deltas():
    clean = np.array([[0, 4], [2, 4]], dtype=np.int16)
    state = fit_noise([clean])
    np.testing.assert_array_equal(state.scales, [1, 0])
    assert state.factor == 0.02

    counts = np.tile(clean, (2048, 1))
    original = counts.copy()
    augmented = augment_fit(counts, state, rng=np.random.default_rng(42))
    replay = augment_fit(counts, state, rng=np.random.default_rng(42))
    different = augment_fit(counts, state, rng=np.random.default_rng(43))

    assert np.issubdtype(augmented.dtype, np.floating)
    assert not np.shares_memory(augmented, counts)
    np.testing.assert_array_equal(augmented, replay)
    assert not np.array_equal(augmented, different)
    np.testing.assert_array_equal(augmented[:, 1], original[:, 1])
    deltas = augmented[:, 0] - original[:, 0]
    assert deltas.min() == pytest.approx(-0.06)
    assert deltas.max() == pytest.approx(0.06)
    np.testing.assert_array_equal(counts, original)
    np.testing.assert_array_equal(clean, [[0, 4], [2, 4]])
    assert counts.dtype == np.int16


def test_fit_noise_weights_bins_equally_across_blocks():
    fit = [np.array([[4, 7]]), np.array([[0, 27], [0, 27], [0, 27]])]

    state = fit_noise(block for block in fit)

    np.testing.assert_allclose(state.scales, np.sqrt([3, 75]))


def test_fit_noise_resolves_small_variation_at_large_integer_offsets():
    offset = 2**62
    fit = [np.array([[offset, offset + 1]]), np.array([[offset + 2, offset + 1]])]

    np.testing.assert_array_equal(fit_noise(fit).scales, [1.0, 0.0])


def test_augment_fit_preserves_constant_int64_values_beyond_float64_integer_precision():
    value = 2**53 + 1
    counts = np.array([[value]], dtype=np.int64)
    state = fit_noise([counts])

    features = augment_fit(counts, state, rng=np.random.default_rng(42))

    assert np.issubdtype(features.dtype, np.floating)
    assert int(features[0, 0]) == value
    np.testing.assert_array_equal(state.scales, [0.0])


def test_augment_fit_never_exceeds_the_bound_after_float_rounding():
    counts = np.tile(np.array([[2**52], [2**52 + 10]], dtype=np.int64), (500, 1))
    state = fit_noise([counts])

    features = augment_fit(counts, state, rng=np.random.default_rng(42))

    deltas = np.abs(features.astype(np.longdouble) - counts.astype(np.longdouble))
    bound = np.longdouble(3 * state.factor) * state.scales.astype(np.longdouble)
    assert np.all(deltas <= bound)


def test_zero_factor_returns_clean_values_as_float_features():
    counts = np.array([[0, 4], [2, 4]], dtype=np.uint16)
    state = fit_noise([counts], factor=0.0)

    features = augment_fit(counts, state, rng=np.random.default_rng(9))

    np.testing.assert_array_equal(state.scales, [1, 0])
    np.testing.assert_array_equal(features, counts)
    assert np.issubdtype(features.dtype, np.floating)
    assert not np.shares_memory(features, counts)


def test_maximum_factor_scales_the_seeded_draw_in_sample_order():
    counts = np.array([[0, 4], [2, 8]])
    state = fit_noise([counts], factor=0.05)
    draws = np.random.default_rng(0).standard_normal((2, 2))
    expected = counts + draws * [0.05, 0.1]

    features = augment_fit(counts, state, rng=np.random.default_rng(0))

    np.testing.assert_array_equal(features, expected)


@pytest.mark.parametrize("factor", [-0.001, 0.051, np.nan, np.inf, -np.inf])
def test_fit_noise_rejects_invalid_factors(factor):
    with pytest.raises(ValueError):
        fit_noise([np.array([[0], [2]])], factor=factor)


@pytest.mark.parametrize(
    "fit",
    [
        [],
        [np.empty((0, 2), dtype=int)],
        [np.empty((2, 0), dtype=int)],
        [np.array([0, 2])],
        [np.array([[0, 2]]), np.array([[0]])],
        [np.array([[0.0, np.nan]])],
        [np.array([[0.0, np.inf]])],
        [np.array([[-1, 0]])],
        [np.array([[0.0, 0.5]])],
    ],
)
def test_fit_noise_rejects_invalid_fit_bins(fit):
    with pytest.raises(ValueError):
        fit_noise(fit)


@pytest.mark.parametrize(
    "counts", [np.array([0, 2]), np.array([[0, 2]]), np.array([[np.nan]])]
)
def test_augment_fit_rejects_invalid_or_mismatched_counts(counts):
    state = fit_noise([np.array([[0], [2]])])

    with pytest.raises(ValueError):
        augment_fit(counts, state, rng=np.random.default_rng(42))
