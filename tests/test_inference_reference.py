"""Reference behavior for paired bootstrap and model-swap inference."""

from __future__ import annotations

import numpy as np
import pytest

from more_metrics import (
    bootstrap_auc,
    bootstrap_idi,
    bootstrap_nri,
    compare_auc,
    compare_nri,
)

Y_BALANCED = np.array([0, 0, 1, 1])
PERFECT = np.array([0.1, 0.2, 0.8, 0.9])


def _assert_interval(interval, expected=None):
    assert len(interval) == 2
    assert np.all(np.isfinite(interval))
    assert interval[0] <= interval[1]
    if expected is not None:
        assert interval[0] == pytest.approx(expected)
        assert interval[1] == pytest.approx(expected)


def test_stratified_bootstrap_auc_has_exact_perfect_interval_and_bands():
    # Every within-class resample remains perfectly separated.  Stratification
    # also guarantees that no invalid one-class bootstrap sample is produced.
    result = bootstrap_auc(
        Y_BALANCED,
        PERFECT,
        n_resamples=99,
        confidence_level=0.95,
        random_state=1729,
    )

    assert result.auc == pytest.approx(1.0)
    _assert_interval(result.ci, 1.0)
    assert result.fpr_grid.ndim == 1
    assert result.fpr_grid.shape == result.tpr_lower.shape == result.tpr_upper.shape
    assert result.fpr_grid[0] == pytest.approx(0.0)
    assert result.fpr_grid[-1] == pytest.approx(1.0)
    assert np.all(np.diff(result.fpr_grid) > 0)
    assert np.all((result.tpr_lower >= 0.0) & (result.tpr_lower <= 1.0))
    assert np.all((result.tpr_upper >= 0.0) & (result.tpr_upper <= 1.0))
    assert np.all(result.tpr_lower <= result.tpr_upper)


def test_paired_stratified_bootstrap_nri_exact_interval():
    reference = np.full(4, 0.5)
    new = np.array([0.2, 0.2, 0.8, 0.8])

    categorical = bootstrap_nri(
        Y_BALANCED,
        reference,
        new,
        thresholds=[0.3, 0.7],
        n_resamples=99,
        random_state=1729,
    )
    continuous = bootstrap_nri(
        Y_BALANCED,
        reference,
        new,
        n_resamples=99,
        random_state=1729,
    )

    for result in (categorical, continuous):
        assert result.events == pytest.approx(1.0)
        assert result.nonevents == pytest.approx(1.0)
        assert result.total == pytest.approx(2.0)
        _assert_interval(result.ci, 2.0)


def test_paired_stratified_bootstrap_idi_exact_interval():
    # Within each class, all (reference, new) pairs are identical, so every
    # paired class-stratified resample has an IDI of 0.6.
    reference = np.array([0.4, 0.4, 0.6, 0.6])
    new = np.array([0.1, 0.1, 0.9, 0.9])

    result = bootstrap_idi(
        Y_BALANCED,
        reference,
        new,
        n_resamples=99,
        random_state=1729,
    )

    assert result.reference_slope == pytest.approx(0.2)
    assert result.new_slope == pytest.approx(0.8)
    assert result.events == pytest.approx(0.3)
    assert result.nonevents == pytest.approx(0.3)
    assert result.idi == pytest.approx(0.6)
    _assert_interval(result.ci, 0.6)


def test_bootstrap_results_are_reproducible_from_integer_seed():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.3, 0.7, 0.9])

    first = bootstrap_auc(y_true, scores, n_resamples=49, random_state=2024)
    second = bootstrap_auc(y_true, scores, n_resamples=49, random_state=2024)

    assert first.auc == pytest.approx(second.auc)
    np.testing.assert_allclose(first.ci, second.ci)
    np.testing.assert_allclose(first.fpr_grid, second.fpr_grid)
    np.testing.assert_allclose(first.tpr_lower, second.tpr_lower)
    np.testing.assert_allclose(first.tpr_upper, second.tpr_upper)


def test_bootstrap_accepts_numpy_generator():
    result = bootstrap_auc(
        Y_BALANCED,
        PERFECT,
        n_resamples=9,
        random_state=np.random.default_rng(7),
    )

    _assert_interval(result.ci, 1.0)


def test_compare_auc_identical_models():
    result = compare_auc(Y_BALANCED, PERFECT, PERFECT, n_resamples=31, random_state=11)

    assert result.difference == pytest.approx(0.0)
    assert result.p_value == pytest.approx(1.0)
    assert result.n_resamples == 31


def test_compare_nri_identical_models():
    categorical = compare_nri(
        Y_BALANCED,
        PERFECT,
        PERFECT,
        thresholds=[0.3, 0.7],
        n_resamples=31,
        random_state=11,
    )
    continuous = compare_nri(
        Y_BALANCED, PERFECT, PERFECT, n_resamples=31, random_state=11
    )

    for result in (categorical, continuous):
        assert result.difference == pytest.approx(0.0)
        assert result.p_value == pytest.approx(1.0)
        assert result.n_resamples == 31


def test_auc_model_swap_permutation_uses_finite_sample_correction():
    # New is perfect and reference is reversed. With twelve paired observations,
    # the only swaps as extreme as the observation are all-reference and
    # all-new assignments. This fixed seed draws neither in 19 permutations.
    y_true = np.repeat([0, 1], 6)
    new = np.r_[np.linspace(0.05, 0.25, 6), np.linspace(0.75, 0.95, 6)]
    reference = 1.0 - new
    n_resamples = 19

    result = compare_auc(
        y_true,
        reference,
        new,
        n_resamples=n_resamples,
        random_state=314159,
    )

    assert result.difference == pytest.approx(1.0)
    assert result.p_value == pytest.approx(1 / (n_resamples + 1))


def test_nri_model_swap_permutation_uses_finite_sample_correction():
    y_true = np.repeat([0, 1], 6)
    reference = np.full(12, 0.5)
    new = np.r_[np.full(6, 0.2), np.full(6, 0.8)]
    n_resamples = 19

    result = compare_nri(
        y_true,
        reference,
        new,
        thresholds=[0.3, 0.7],
        n_resamples=n_resamples,
        random_state=314159,
    )

    assert result.difference == pytest.approx(2.0)
    assert result.p_value == pytest.approx(1 / (n_resamples + 1))


@pytest.mark.parametrize("comparison", ["auc", "nri"])
def test_model_swap_comparison_is_symmetric_and_reproducible(comparison):
    y_true = np.array([0, 0, 0, 1, 1, 1])
    reference = np.array([0.1, 0.4, 0.6, 0.3, 0.7, 0.9])
    new = np.array([0.2, 0.3, 0.5, 0.4, 0.8, 0.95])
    kwargs = {"n_resamples": 49, "random_state": 2025}

    if comparison == "auc":
        forward = compare_auc(y_true, reference, new, **kwargs)
        repeat = compare_auc(y_true, reference, new, **kwargs)
        reverse = compare_auc(y_true, new, reference, **kwargs)
    else:
        forward = compare_nri(y_true, reference, new, **kwargs)
        repeat = compare_nri(y_true, reference, new, **kwargs)
        reverse = compare_nri(y_true, new, reference, **kwargs)

    assert repeat.difference == pytest.approx(forward.difference)
    assert repeat.p_value == pytest.approx(forward.p_value)
    assert reverse.difference == pytest.approx(-forward.difference)
    assert reverse.p_value == pytest.approx(forward.p_value)


@pytest.mark.parametrize("n_resamples", [0, -1, 1.5, True])
def test_inference_rejects_nonpositive_or_noninteger_resample_counts(n_resamples):
    with pytest.raises((TypeError, ValueError)):
        bootstrap_auc(Y_BALANCED, PERFECT, n_resamples=n_resamples, random_state=1)


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, -0.1, 1.1, np.nan])
def test_bootstrap_rejects_invalid_confidence_level(confidence_level):
    with pytest.raises(ValueError):
        bootstrap_auc(
            Y_BALANCED,
            PERFECT,
            n_resamples=9,
            confidence_level=confidence_level,
            random_state=1,
        )


def test_inference_reuses_metric_input_validation():
    with pytest.raises(ValueError):
        bootstrap_auc([0, 0], [0.1, 0.2], n_resamples=9, random_state=1)
    with pytest.raises(ValueError):
        bootstrap_nri(
            Y_BALANCED,
            [0.5] * 4,
            [0.2, 0.2, 0.8, 1.1],
            n_resamples=9,
            random_state=1,
        )
    with pytest.raises(ValueError):
        bootstrap_idi(
            Y_BALANCED,
            [0.5] * 4,
            [0.2, 0.2, 0.8, np.nan],
            n_resamples=9,
            random_state=1,
        )
