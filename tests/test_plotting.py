from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyCollection

from more_metrics._types import AUCResult, IDIResult, NRIResult
from more_metrics.plotting import plot_reclassification, plot_roc_comparison


def _auc_result(*, band: bool = False) -> AUCResult:
    kwargs = {}
    if band:
        kwargs = {
            "fpr_grid": np.array([0.0, 0.5, 1.0]),
            "tpr_lower": np.array([0.0, 0.6, 1.0]),
            "tpr_upper": np.array([0.2, 1.0, 1.0]),
        }
    return AUCResult(
        auc=0.75,
        fpr=np.array([0.0, 0.0, 0.5, 1.0]),
        tpr=np.array([0.0, 0.5, 1.0, 1.0]),
        thresholds=np.array([np.inf, 0.8, 0.4, 0.1]),
        ci=(0.6, 0.9),
        **kwargs,
    )


def test_roc_comparison_creates_curves_band_and_clear_labels():
    figure, axes = plot_roc_comparison(
        _auc_result(band=True),
        _auc_result(),
        reference_label="Baseline",
        new_label="Updated",
    )

    assert figure is axes.figure
    assert len(axes.lines) == 3
    assert any(isinstance(item, PolyCollection) for item in axes.collections)
    assert axes.get_xlabel() == "False positive rate"
    assert axes.get_ylabel() == "True positive rate"
    legend_text = [text.get_text() for text in axes.get_legend().get_texts()]
    assert any("Baseline (AUC = 0.750" in text for text in legend_text)
    assert any("Updated (AUC = 0.750" in text for text in legend_text)
    assert "Chance" in legend_text
    plt.close(figure)


def test_roc_comparison_uses_supplied_axes():
    figure, supplied = plt.subplots()

    returned_figure, returned_axes = plot_roc_comparison(
        _auc_result(), _auc_result(), ax=supplied
    )

    assert returned_figure is figure
    assert returned_axes is supplied
    plt.close(figure)


def test_reclassification_draws_paired_risks_and_threshold_annotations():
    y_true = np.array([0, 0, 1, 1])
    reference = np.array([0.3, 0.4, 0.6, 0.7])
    new = np.array([0.1, 0.2, 0.8, 0.9])
    result = NRIResult(1.0, 1.0, 2.0, thresholds=(0.25, 0.75))

    figure, axes = plot_reclassification(
        y_true,
        reference,
        new,
        result,
        threshold_labels=["Low cutoff", "High cutoff"],
    )

    assert figure is axes.figure
    assert len(axes.collections) == 4
    assert len(axes.lines) == len(y_true) + 2
    assert axes.get_ylabel() == "Predicted risk"
    assert [tick.get_text() for tick in axes.get_xticklabels()] == [
        "Reference",
        "New",
    ]
    legend_text = [text.get_text() for text in axes.get_legend().get_texts()]
    assert {"Event", "Non-event", "Low cutoff", "High cutoff"} <= set(legend_text)
    assert "IDI" not in axes.get_title()
    plt.close(figure)


def test_reclassification_accepts_idi_result_and_supplied_axes():
    result = IDIResult(0.2, 0.1, 0.1, 0.3, 0.5)
    figure, supplied = plt.subplots()

    returned_figure, returned_axes = plot_reclassification(
        [0, 1], [0.3, 0.6], [0.2, 0.8], result, ax=supplied
    )

    assert returned_figure is figure
    assert returned_axes is supplied
    assert "IDI" not in returned_axes.get_title()
    plt.close(figure)


@pytest.mark.parametrize(
    ("y_true", "reference", "new"),
    [
        ([0, 1], [0.2], [0.2, 0.8]),
        ([0, 1], [-0.1, 0.8], [0.2, 0.8]),
        ([0, 2], [0.2, 0.8], [0.2, 0.8]),
        ([[0, 1]], [[0.2, 0.8]], [[0.2, 0.8]]),
    ],
)
def test_reclassification_rejects_invalid_inputs(y_true, reference, new):
    with pytest.raises(ValueError):
        plot_reclassification(
            y_true,
            reference,
            new,
            NRIResult(0.0, 0.0, 0.0),
        )


def test_plotting_does_not_show_save_or_print(monkeypatch, capsys):
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: pytest.fail("show called"))
    monkeypatch.setattr(
        plt.Figure, "savefig", lambda *args, **kwargs: pytest.fail("save called")
    )

    figure, _ = plot_roc_comparison(_auc_result(), _auc_result())

    assert capsys.readouterr() == ("", "")
    plt.close(figure)
