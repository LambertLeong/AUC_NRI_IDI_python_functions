"""Regression coverage for validation paths exposed by the public API."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from more_metrics import (
    AUCResult,
    NRIResult,
    auc,
    bootstrap_auc,
    plot_reclassification,
    plot_roc_comparison,
)


def _result(**changes: object) -> AUCResult:
    values = {
        "auc": 0.5,
        "fpr": np.array([0.0, 1.0]),
        "tpr": np.array([0.0, 1.0]),
        "thresholds": np.array([np.inf, 0.0]),
    }
    values.update(changes)
    return AUCResult(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (["not", "numeric"], "numeric"),
        ([0.1 + 1j, 0.9], "real"),
    ],
)
def test_metric_validation_rejects_non_real_scores(values, message):
    with pytest.raises(ValueError, match=message):
        auc([0, 1], values)


@pytest.mark.parametrize(
    "grid",
    [
        [[0.0, 1.0]],
        [0.0],
        [0.1, 1.0],
        [0.0, 0.9],
        [0.0, np.nan, 1.0],
        [0.0, 1.1, 1.0],
        [0.0, 0.5, 0.5, 1.0],
    ],
)
def test_bootstrap_auc_rejects_invalid_fpr_grids(grid):
    with pytest.raises(ValueError):
        bootstrap_auc(
            [0, 1],
            [0.2, 0.8],
            n_resamples=2,
            fpr_grid=grid,
            random_state=1,
        )


def test_bootstrap_rejects_invalid_random_state():
    with pytest.raises(TypeError, match="random_state"):
        bootstrap_auc([0, 1], [0.2, 0.8], n_resamples=2, random_state="seed")


@pytest.mark.parametrize(
    "result",
    [
        object(),
        _result(auc=2.0),
        _result(fpr=np.array([0.0]), tpr=np.array([0.0, 1.0])),
        _result(fpr=np.array([0.0, 2.0])),
        _result(fpr_grid=np.array([0.0, 1.0])),
        _result(
            fpr_grid=np.array([0.0, 1.0]),
            tpr_lower=np.array([0.0]),
            tpr_upper=np.array([0.0, 1.0]),
        ),
        _result(
            fpr_grid=np.array([0.0, 1.0]),
            tpr_lower=np.array([0.8, 1.0]),
            tpr_upper=np.array([0.2, 1.0]),
        ),
        _result(ci=(0.9, 0.1)),
    ],
)
def test_roc_plot_rejects_malformed_results(result):
    with pytest.raises((TypeError, ValueError)):
        plot_roc_comparison(result, _result())  # type: ignore[arg-type]


def test_roc_plot_rejects_bad_labels_and_axes():
    with pytest.raises(ValueError, match="reference_label"):
        plot_roc_comparison(_result(), _result(), reference_label="")
    with pytest.raises(ValueError, match="new_label"):
        plot_roc_comparison(_result(), _result(), new_label="")
    with pytest.raises(TypeError, match="ax"):
        plot_roc_comparison(_result(), _result(), ax=object())  # type: ignore[arg-type]


def test_reclassification_threshold_label_validation_and_mapping():
    result = NRIResult(0.0, 0.0, 0.0, thresholds=(0.3, 0.7))
    figure, axes = plot_reclassification(
        [0, 1],
        [0.2, 0.8],
        [0.1, 0.9],
        result,
        threshold_labels={0.3: "Low", 0.7: "High"},
    )
    assert figure is axes.figure
    plt.close(figure)

    invalid_labels = ["one label", "string", {0.3: "Low"}]
    for labels in invalid_labels:
        with pytest.raises(ValueError):
            plot_reclassification(
                [0, 1],
                [0.2, 0.8],
                [0.1, 0.9],
                result,
                threshold_labels=labels,
            )


def test_reclassification_rejects_result_and_label_mismatches():
    with pytest.raises(TypeError, match="result"):
        plot_reclassification(
            [0, 1],
            [0.2, 0.8],
            [0.1, 0.9],
            object(),  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="threshold_labels"):
        plot_reclassification(
            [0, 1],
            [0.2, 0.8],
            [0.1, 0.9],
            NRIResult(0.0, 0.0, 0.0),
            threshold_labels=["Low"],
        )
    with pytest.raises(ValueError, match="new probabilities"):
        plot_reclassification(
            [0, 1],
            [0.2, 0.8],
            [0.1, 1.1],
            NRIResult(0.0, 0.0, 0.0),
        )
