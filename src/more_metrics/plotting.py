"""Plotting helpers for numerical :mod:`more_metrics` results.

The functions in this module do not display or save figures.  They create a
new axes when needed and otherwise draw on the axes supplied by the caller.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._types import AUCResult, IDIResult, NRIResult

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


def _get_axes(ax: Axes | None) -> tuple[Figure, Axes]:
    if ax is None:
        figure, axes = plt.subplots()
        return figure, axes
    if not isinstance(ax, Axes):
        raise TypeError("ax must be a matplotlib.axes.Axes instance or None")
    return cast(Figure, ax.figure), ax


def _float_vector(values: ArrayLike, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional numeric array") from error
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_auc_result(result: AUCResult, name: str) -> None:
    if not isinstance(result, AUCResult):
        raise TypeError(f"{name} must be an AUCResult")
    if not np.isfinite(result.auc) or not 0.0 <= result.auc <= 1.0:
        raise ValueError(f"{name}.auc must be finite and within [0, 1]")

    fpr = _float_vector(result.fpr, f"{name}.fpr")
    tpr = _float_vector(result.tpr, f"{name}.tpr")
    if fpr.shape != tpr.shape:
        raise ValueError(f"{name}.fpr and {name}.tpr must have equal lengths")
    if np.any((fpr < 0.0) | (fpr > 1.0)) or np.any((tpr < 0.0) | (tpr > 1.0)):
        raise ValueError(f"{name} ROC coordinates must be within [0, 1]")

    band_values = (result.fpr_grid, result.tpr_lower, result.tpr_upper)
    if all(value is None for value in band_values):
        return
    if any(value is None for value in band_values):
        raise ValueError(
            f"{name} must provide fpr_grid, tpr_lower, and tpr_upper together"
        )

    assert result.fpr_grid is not None
    assert result.tpr_lower is not None
    assert result.tpr_upper is not None
    grid = _float_vector(result.fpr_grid, f"{name}.fpr_grid")
    lower = _float_vector(result.tpr_lower, f"{name}.tpr_lower")
    upper = _float_vector(result.tpr_upper, f"{name}.tpr_upper")
    if not (grid.shape == lower.shape == upper.shape):
        raise ValueError(f"{name} confidence-band arrays must have equal lengths")
    if (
        np.any((grid < 0.0) | (grid > 1.0))
        or np.any((lower < 0.0) | (lower > 1.0))
        or np.any((upper < 0.0) | (upper > 1.0))
        or np.any(lower > upper)
    ):
        raise ValueError(f"{name} confidence-band values must be ordered within [0, 1]")


def _auc_label(label: str, result: AUCResult) -> str:
    text = f"{label} (AUC = {result.auc:.3f}"
    if result.ci is not None:
        low, high = result.ci
        if not np.all(np.isfinite((low, high))) or low > high:
            raise ValueError("AUC confidence interval must be finite and ordered")
        text += f", CI {low:.3f}-{high:.3f}"
    return text + ")"


def plot_roc_comparison(
    reference_result: AUCResult,
    new_result: AUCResult,
    *,
    reference_label: str = "Reference",
    new_label: str = "New",
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot two ROC curves and any pointwise confidence bands they contain."""

    _validate_auc_result(reference_result, "reference_result")
    _validate_auc_result(new_result, "new_result")
    if not isinstance(reference_label, str) or not reference_label:
        raise ValueError("reference_label must be a non-empty string")
    if not isinstance(new_label, str) or not new_label:
        raise ValueError("new_label must be a non-empty string")

    figure, axes = _get_axes(ax)
    colors = ("C0", "C1")
    for result, label, color in (
        (reference_result, reference_label, colors[0]),
        (new_result, new_label, colors[1]),
    ):
        axes.plot(
            result.fpr,
            result.tpr,
            color=color,
            linewidth=2,
            label=_auc_label(label, result),
        )
        if result.fpr_grid is not None:
            assert result.tpr_lower is not None
            assert result.tpr_upper is not None
            axes.fill_between(
                result.fpr_grid,
                result.tpr_lower,
                result.tpr_upper,
                color=color,
                alpha=0.2,
                linewidth=0,
            )

    axes.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        color="0.5",
        linewidth=1,
        label="Chance",
    )
    axes.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0.0, 1.0),
        ylim=(0.0, 1.0),
        title="ROC comparison",
    )
    axes.legend()
    return figure, axes


def _threshold_text(
    thresholds: tuple[float, ...],
    threshold_labels: Sequence[str] | Mapping[float, str] | None,
) -> list[str]:
    if threshold_labels is None:
        return [f"Threshold {threshold:g}" for threshold in thresholds]
    if isinstance(threshold_labels, Mapping):
        try:
            return [str(threshold_labels[threshold]) for threshold in thresholds]
        except KeyError as error:
            raise ValueError(
                "threshold_labels mapping must contain every threshold"
            ) from error
    if isinstance(threshold_labels, str):
        raise ValueError("threshold_labels must contain one label per threshold")
    labels = [str(label) for label in threshold_labels]
    if len(labels) != len(thresholds):
        raise ValueError("threshold_labels must contain one label per threshold")
    return labels


def plot_reclassification(
    y_true: ArrayLike,
    reference: ArrayLike,
    new: ArrayLike,
    result: NRIResult | IDIResult,
    *,
    ax: Axes | None = None,
    threshold_labels: Sequence[str] | Mapping[float, str] | None = None,
) -> tuple[Figure, Axes]:
    """Plot paired changes in predicted risk, separated by observed class.

    Each thin line connects one observation's reference and new predicted
    probability.  Event and non-event observations use distinct colors.
    Categorical NRI thresholds are drawn as horizontal reference lines.
    """

    if not isinstance(result, (NRIResult, IDIResult)):
        raise TypeError("result must be an NRIResult or IDIResult")
    outcome = _float_vector(y_true, "y_true")
    reference_probability = _float_vector(reference, "reference")
    new_probability = _float_vector(new, "new")
    if not (outcome.shape == reference_probability.shape == new_probability.shape):
        raise ValueError("y_true, reference, and new must have equal lengths")
    if set(np.unique(outcome)) != {0.0, 1.0}:
        raise ValueError("y_true must contain exactly 0 and 1, with both present")
    if np.any((reference_probability < 0.0) | (reference_probability > 1.0)):
        raise ValueError("reference probabilities must be within [0, 1]")
    if np.any((new_probability < 0.0) | (new_probability > 1.0)):
        raise ValueError("new probabilities must be within [0, 1]")

    figure, axes = _get_axes(ax)
    x_pair = np.array([0.0, 1.0])
    class_styles = ((0.0, "C0", "Non-event"), (1.0, "C3", "Event"))
    for class_value, color, label in class_styles:
        mask = outcome == class_value
        for old, updated in zip(
            reference_probability[mask], new_probability[mask], strict=True
        ):
            axes.plot(x_pair, [old, updated], color=color, alpha=0.25, linewidth=1)
        axes.scatter(
            np.zeros(np.count_nonzero(mask)),
            reference_probability[mask],
            color=color,
            alpha=0.75,
            s=24,
            label=label,
            zorder=3,
        )
        axes.scatter(
            np.ones(np.count_nonzero(mask)),
            new_probability[mask],
            color=color,
            alpha=0.75,
            s=24,
            zorder=3,
        )

    thresholds = result.thresholds if isinstance(result, NRIResult) else None
    if threshold_labels is not None and not thresholds:
        raise ValueError(
            "threshold_labels can only be used with categorical NRI thresholds"
        )
    if thresholds:
        labels = _threshold_text(thresholds, threshold_labels)
        for threshold, label in zip(thresholds, labels, strict=True):
            axes.axhline(
                threshold,
                color="0.4",
                linestyle=":",
                linewidth=1,
                label=label,
            )

    axes.set(
        xticks=x_pair,
        xticklabels=("Reference", "New"),
        xlim=(-0.25, 1.25),
        ylim=(0.0, 1.0),
        ylabel="Predicted risk",
        title="Risk reclassification",
    )
    axes.legend()
    return figure, axes


__all__ = ["plot_reclassification", "plot_roc_comparison"]
