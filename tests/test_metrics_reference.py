"""Independent reference cases for the version 1.0 numerical metrics.

The expected values below are calculated directly from pair counts, category
movements, and class-conditional means.  They intentionally do not call a
second library implementation of the metric under test.
"""

from __future__ import annotations

import numpy as np
import pytest

from more_metrics import auc, categorical_nri, continuous_nri, idi

Y_BALANCED = np.array([0, 0, 1, 1])


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([0.1, 0.2, 0.8, 0.9], 1.0),  # all four event/nonevent pairs concordant
        ([0.9, 0.8, 0.2, 0.1], 0.0),  # all four pairs discordant
        ([0.5, 0.5, 0.5, 0.5], 0.5),  # four ties, each worth one half
        ([0.1, 0.6, 0.4, 0.8], 0.75),  # three of four pairs concordant
    ],
)
def test_auc_hand_calculated_pair_counts(scores, expected):
    result = auc(Y_BALANCED, scores)

    assert result.auc == pytest.approx(expected)
    assert result.fpr.ndim == result.tpr.ndim == 1
    assert result.fpr.shape == result.tpr.shape
    assert result.fpr[0] == pytest.approx(0.0)
    assert result.tpr[0] == pytest.approx(0.0)
    assert result.fpr[-1] == pytest.approx(1.0)
    assert result.tpr[-1] == pytest.approx(1.0)
    assert np.all(np.diff(result.fpr) >= 0)
    assert np.all(np.diff(result.tpr) >= 0)


def test_auc_roc_coordinates_for_perfect_predictions():
    result = auc(Y_BALANCED, [0.1, 0.2, 0.8, 0.9])

    np.testing.assert_allclose(result.fpr, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(result.tpr, [0.0, 1.0, 1.0])


@pytest.mark.parametrize(
    ("reference", "new", "expected_events", "expected_nonevents"),
    [
        # A common middle category moves all events up and all nonevents down.
        ([0.5] * 4, [0.2, 0.2, 0.8, 0.8], 1.0, 1.0),
        # The same movements in the harmful direction.
        ([0.2, 0.2, 0.8, 0.8], [0.5] * 4, -1.0, -1.0),
        # No category movement.
        ([0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8], 0.0, 0.0),
    ],
)
def test_categorical_nri_hand_calculated_extremes(
    reference, new, expected_events, expected_nonevents
):
    result = categorical_nri(Y_BALANCED, reference, new, thresholds=[0.3, 0.7])

    assert result.events == pytest.approx(expected_events)
    assert result.nonevents == pytest.approx(expected_nonevents)
    assert result.total == pytest.approx(expected_events + expected_nonevents)


def test_categorical_nri_partially_improved():
    # Events: two up, one down, one unchanged => (2 - 1) / 4 = 0.25.
    # Nonevents: one down, one up, two unchanged => (1 - 1) / 4 = 0.
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    reference = np.array([0.2, 0.4, 0.8, 0.6, 0.2, 0.4, 0.8, 0.6])
    new = np.array([0.4, 0.2, 0.8, 0.8, 0.1, 0.8, 0.4, 0.6])

    result = categorical_nri(y_true, reference, new, thresholds=[0.3, 0.7])

    assert result.events == pytest.approx(0.25)
    assert result.nonevents == pytest.approx(0.0)
    assert result.total == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("reference", "new", "expected_events", "expected_nonevents"),
    [
        ([0.5] * 4, [0.2, 0.2, 0.8, 0.8], 1.0, 1.0),
        ([0.2, 0.2, 0.8, 0.8], [0.5] * 4, -1.0, -1.0),
        ([0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8], 0.0, 0.0),
    ],
)
def test_continuous_nri_hand_calculated_extremes(
    reference, new, expected_events, expected_nonevents
):
    result = continuous_nri(Y_BALANCED, reference, new)

    assert result.events == pytest.approx(expected_events)
    assert result.nonevents == pytest.approx(expected_nonevents)
    assert result.total == pytest.approx(expected_events + expected_nonevents)


def test_continuous_nri_ties_are_no_movement():
    # Events: two up, one down, one tied => 0.25.
    # Nonevents: one down, one up, two tied => 0.
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    reference = np.array([0.2, 0.4, 0.8, 0.6, 0.2, 0.4, 0.8, 0.6])
    new = np.array([0.3, 0.2, 0.8, 0.7, 0.1, 0.5, 0.8, 0.6])

    result = continuous_nri(y_true, reference, new)

    assert result.events == pytest.approx(0.25)
    assert result.nonevents == pytest.approx(0.0)
    assert result.total == pytest.approx(0.25)


def test_idi_is_change_in_discrimination_slope():
    # Reference: mean(event)=0.7, mean(nonevent)=0.3, slope=0.4.
    # New:       mean(event)=0.85, mean(nonevent)=0.15, slope=0.7.
    y_true = np.array([1, 1, 0, 0])
    reference = np.array([0.6, 0.8, 0.4, 0.2])
    new = np.array([0.8, 0.9, 0.2, 0.1])

    result = idi(y_true, reference, new)

    assert result.reference_slope == pytest.approx(0.4)
    assert result.new_slope == pytest.approx(0.7)
    assert result.events == pytest.approx(0.15)
    assert result.nonevents == pytest.approx(0.15)
    assert result.idi == pytest.approx(0.3)
    assert result.idi == pytest.approx(result.events + result.nonevents)


def test_idi_reversed_and_identical_predictions():
    perfect = np.array([0.1, 0.1, 0.9, 0.9])
    reversed_scores = 1.0 - perfect

    reversed_result = idi(Y_BALANCED, perfect, reversed_scores)
    identical_result = idi(Y_BALANCED, perfect, perfect)

    assert reversed_result.reference_slope == pytest.approx(0.8)
    assert reversed_result.new_slope == pytest.approx(-0.8)
    assert reversed_result.events == pytest.approx(-0.8)
    assert reversed_result.nonevents == pytest.approx(-0.8)
    assert reversed_result.idi == pytest.approx(-1.6)
    assert identical_result.idi == pytest.approx(0.0)
    assert identical_result.events == pytest.approx(0.0)
    assert identical_result.nonevents == pytest.approx(0.0)


def test_constant_predictions_are_valid_degenerate_models():
    scores = np.full(4, 0.5)

    assert auc(Y_BALANCED, scores).auc == pytest.approx(0.5)
    assert continuous_nri(Y_BALANCED, scores, scores).total == pytest.approx(0.0)
    assert categorical_nri(
        Y_BALANCED, scores, scores, thresholds=[0.3, 0.7]
    ).total == pytest.approx(0.0)
    result = idi(Y_BALANCED, scores, scores)
    assert result.reference_slope == pytest.approx(0.0)
    assert result.new_slope == pytest.approx(0.0)
    assert result.idi == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("y_true", "scores"),
    [
        ([], []),
        ([0, 1], [0.2]),
        ([[0, 1]], [[0.2, 0.8]]),
        ([0, 1], [0.2, np.nan]),
        ([0, 1], [0.2, np.inf]),
        ([0, 2], [0.2, 0.8]),
        ([0, 0], [0.2, 0.8]),
        ([1, 1], [0.2, 0.8]),
    ],
)
def test_auc_rejects_invalid_shared_inputs(y_true, scores):
    with pytest.raises(ValueError):
        auc(y_true, scores)


def test_auc_accepts_finite_scores_outside_probability_range():
    assert auc([0, 1], [-2.0, 3.0]).auc == pytest.approx(1.0)


@pytest.mark.parametrize("metric", [continuous_nri, idi])
@pytest.mark.parametrize(
    ("reference", "new"),
    [
        ([-0.01, 0.8], [0.2, 0.8]),
        ([0.2, 1.01], [0.2, 0.8]),
        ([0.2, 0.8], [-0.01, 0.8]),
        ([0.2, 0.8], [0.2, 1.01]),
        ([0.2], [0.2, 0.8]),
        ([[0.2, 0.8]], [[0.2, 0.8]]),
    ],
)
def test_probability_metrics_reject_invalid_predictions(metric, reference, new):
    with pytest.raises(ValueError):
        metric([0, 1], reference, new)


@pytest.mark.parametrize(
    "thresholds",
    [
        [],
        [0.0],
        [1.0],
        [-0.1, 0.5],
        [0.5, 1.1],
        [0.7, 0.3],
        [0.3, 0.3],
        [np.nan],
        [[0.3, 0.7]],
    ],
)
def test_categorical_nri_rejects_invalid_thresholds(thresholds):
    with pytest.raises(ValueError):
        categorical_nri(
            Y_BALANCED,
            [0.2, 0.4, 0.6, 0.8],
            [0.1, 0.3, 0.7, 0.9],
            thresholds=thresholds,
        )
