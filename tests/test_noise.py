from fractions import Fraction

import numpy as np
import pytest

import spike_datasets.noise as noise_module
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

    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        with pytest.raises(ValueError, match="cannot be represented exactly"):
            augment_fit(counts, state, rng=np.random.default_rng(42))
        return

    features = augment_fit(counts, state, rng=np.random.default_rng(42))

    assert np.issubdtype(features.dtype, np.floating)
    assert int(features[0, 0]) == value
    np.testing.assert_array_equal(state.scales, [0.0])


def test_augment_fit_accepts_exact_large_integer_when_wide_precision_is_unavailable():
    value = 2**54
    counts = np.array([[value]], dtype=np.int64)
    state = fit_noise([counts])

    features = augment_fit(counts, state, rng=np.random.default_rng(42))

    assert int(features[0, 0]) == value


def test_augment_fit_handles_uint64_maximum_without_scalar_overflow():
    value = int(np.iinfo(np.uint64).max)
    counts = np.array([[value]], dtype=np.uint64)
    state = fit_noise([counts])

    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        with pytest.raises(ValueError, match="cannot be represented exactly"):
            augment_fit(counts, state, rng=np.random.default_rng(42))
        return

    features = augment_fit(counts, state, rng=np.random.default_rng(42))

    assert int(features[0, 0]) == value


def test_float64_only_fallback_allows_exact_large_counts_and_keeps_endpoint(monkeypatch):
    counts = np.array([[0], [2**54]], dtype=np.int64)
    state = fit_noise([np.array([[0], [2]], dtype=np.int64)], factor=1 / 32)
    monkeypatch.setattr(noise_module.np, "longdouble", np.float64)

    features = augment_fit(counts, state, rng=np.random.default_rng(988))

    assert features[0, 0] == 3 / 32
    assert int(features[1, 0]) == 2**54


@pytest.mark.parametrize(
    ("fit_values", "factor"),
    [((1, 21), 1 / 30), ((1, 40013), 1 / 60018)],
)
def test_augment_fit_clamps_near_integer_endpoints_without_losing_the_value(
    fit_values, factor
):
    fit = np.array(fit_values, dtype=np.int64).reshape(-1, 1)
    state = fit_noise([fit], factor=factor)

    features = augment_fit(np.array([[1]], dtype=np.int64), state, rng=np.random.default_rng(755))

    exact_limit = 3 * Fraction.from_float(float(state.scales[0])) * Fraction.from_float(factor)
    delta = Fraction(*features[0, 0].as_integer_ratio()) - 1
    assert features[0, 0] > 0
    assert abs(delta) <= exact_limit


def test_float64_only_fallback_detects_hidden_integer_overshoot(monkeypatch):
    fit = np.array([[1], [1 + 2**62]], dtype=np.int64)
    state = fit_noise([fit], factor=1 / 32)
    monkeypatch.setattr(noise_module.np, "longdouble", np.float64)

    feature = augment_fit(np.array([[1]], dtype=np.int64), state, rng=np.random.default_rng(755))[0, 0]

    exact_limit = 3 * Fraction.from_float(float(state.scales[0])) * Fraction.from_float(state.factor)
    delta = Fraction(*feature.as_integer_ratio()) - 1
    assert abs(delta) <= exact_limit


def test_float64_only_fallback_handles_the_smallest_valid_noise_factor(monkeypatch):
    factor = float.fromhex("0x0.0000000000001p-1022")
    state = fit_noise([np.array([[0], [1]], dtype=np.int64)], factor=factor)
    monkeypatch.setattr(noise_module.np, "longdouble", np.float64)

    feature = augment_fit(
        np.array([[0]], dtype=np.int64), state, rng=np.random.default_rng(988)
    )[0, 0]

    exact_limit = 3 * Fraction.from_float(float(state.scales[0])) * Fraction.from_float(factor)
    assert feature > 0
    assert Fraction(*feature.as_integer_ratio()) <= exact_limit


def test_wide_precision_endpoint_handles_the_smallest_valid_noise_factor():
    factor = float.fromhex("0x0.0000000000001p-1022")
    state = fit_noise([np.array([[0], [1]], dtype=np.int64)], factor=factor)

    feature = augment_fit(
        np.array([[0]], dtype=np.int64), state, rng=np.random.default_rng(988)
    )[0, 0]

    exact_limit = 3 * Fraction.from_float(float(state.scales[0])) * Fraction.from_float(factor)
    assert feature > 0
    assert Fraction(*feature.as_integer_ratio()) <= exact_limit


def test_augment_fit_never_exceeds_the_bound_after_float_rounding():
    counts = np.tile(np.array([[2**52], [2**52 + 10]], dtype=np.int64), (500, 1))
    state = fit_noise([counts])

    features = augment_fit(counts, state, rng=np.random.default_rng(42))

    bound = 3 * Fraction.from_float(state.factor) * Fraction.from_float(state.scales[0])
    deltas = [
        abs(Fraction(*feature.as_integer_ratio()) - int(count))
        for feature, count in zip(features[:, 0], counts[:, 0])
    ]
    assert all(delta <= bound for delta in deltas)


def test_augment_fit_uses_a_conservative_bound_for_clipped_endpoint_rounding():
    counts = np.array([[0], [2]], dtype=np.int64)
    state = fit_noise([counts], factor=1 / 32)

    features = augment_fit(counts, state, rng=np.random.default_rng(988))

    delta = Fraction(*features[0, 0].as_integer_ratio()) - int(counts[0, 0])
    exact_bound = 3 * Fraction.from_float(float(state.scales[0])) * Fraction.from_float(state.factor)
    assert features[0, 0] == 3 / 32
    assert delta <= exact_bound


def test_augment_fit_keeps_small_values_stable_when_large_values_need_wide_precision():
    state = fit_noise([np.array([[0], [2]], dtype=np.int64)])
    rng_seed = 12

    alone = augment_fit(np.array([[0]], dtype=np.int64), state, rng=np.random.default_rng(rng_seed))
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        with pytest.raises(ValueError, match="cannot be represented exactly"):
            augment_fit(
                np.array([[0], [2**53 + 1]], dtype=np.int64),
                state,
                rng=np.random.default_rng(rng_seed),
            )
        return
    batched = augment_fit(
        np.array([[0], [2**53 + 1]], dtype=np.int64), state, rng=np.random.default_rng(rng_seed)
    )

    assert alone[0, 0] == batched[0, 0]


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
