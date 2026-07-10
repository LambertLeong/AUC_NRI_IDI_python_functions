"""Deprecated interfaces retained for a controlled 0.11 migration."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .inference import bootstrap_auc, compare_auc, compare_nri
from .metrics import categorical_nri, continuous_nri
from .plotting import plot_roc_comparison

ArrayLike = npt.ArrayLike


def _deprecated(old: str, replacement: str) -> None:
    warnings.warn(
        f"{old} is deprecated and will be removed in a future release; "
        f"use {replacement} instead. See docs/migration-0.11-to-1.0.md.",
        DeprecationWarning,
        stacklevel=3,
    )


def bootstrap_results(
    y_truth: ArrayLike,
    y_pred: ArrayLike,
    num_bootstraps: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Deprecated legacy bootstrap helper.

    The old threshold-indexed averages mixed incompatible ROC coordinates and
    cannot faithfully represent the corrected fixed-FPR bootstrap.
    """

    _deprecated("bootstrap_results", "bootstrap_auc")
    raise RuntimeError(
        "bootstrap_results cannot represent the corrected 1.0 bootstrap. "
        "Use bootstrap_auc(...), then read ci, fpr_grid, tpr_lower, and "
        "tpr_upper. See docs/migration-0.11-to-1.0.md."
    )


def calculate_auc_with_ci(
    ground_truth: ArrayLike,
    predicted_probabilities: ArrayLike,
    num_bootstraps: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Deprecated legacy AUC interval helper.

    Its final value was a bootstrap standard deviation, while the corrected API
    reports a percentile interval, so the legacy tuple is not lossless.
    """

    _deprecated("calculate_auc_with_ci", "bootstrap_auc")
    raise RuntimeError(
        "calculate_auc_with_ci cannot safely encode a percentile confidence "
        "interval in its legacy six-item tuple. Use bootstrap_auc(...). "
        "See docs/migration-0.11-to-1.0.md."
    )


def plot_comparing_aucs(
    truth: ArrayLike,
    reference_model: ArrayLike,
    new_model: ArrayLike,
    n_bootstraps: int = 1000,
    save: bool = False,
) -> tuple[Figure, Axes]:
    """Deprecated ROC comparison wrapper returning ``(figure, axes)``."""

    _deprecated(
        "plot_comparing_aucs",
        "bootstrap_auc and plot_roc_comparison",
    )
    if save:
        raise ValueError(
            "Implicit saving is no longer supported. Call fig.savefig(...) "
            "on the figure returned by plot_roc_comparison."
        )
    reference = bootstrap_auc(
        truth,
        reference_model,
        n_resamples=n_bootstraps,
        random_state=42,
    )
    new = bootstrap_auc(
        truth,
        new_model,
        n_resamples=n_bootstraps,
        random_state=42,
    )
    return plot_roc_comparison(reference, new)


def calculate_nri(
    y_truth: ArrayLike,
    y_ref: ArrayLike,
    y_new: ArrayLike,
    risk_thresholds: Sequence[float],
) -> tuple[float, float, float]:
    """Deprecated categorical NRI wrapper preserving its three-item tuple."""

    _deprecated("calculate_nri", "categorical_nri")
    result = categorical_nri(y_truth, y_ref, y_new, risk_thresholds)
    return result.events, result.nonevents, result.total


def category_free_nri(
    y_truth: ArrayLike,
    y_ref: ArrayLike,
    y_new: ArrayLike,
) -> tuple[float, float, float]:
    """Deprecated continuous NRI wrapper preserving its three-item tuple."""

    _deprecated("category_free_nri", "continuous_nri")
    result = continuous_nri(y_truth, y_ref, y_new)
    return result.events, result.nonevents, result.total


def plot_idi(
    y_truth: ArrayLike,
    ref_model: ArrayLike,
    new_model: ArrayLike,
    thresholds: object,
    num_bootstraps: int = 1000,
    show: bool = True,
    save: bool = False,
) -> dict[str, object]:
    """Deprecated legacy plot that cannot represent standard IDI."""

    _deprecated("plot_idi", "idi and plot_reclassification")
    raise RuntimeError(
        "plot_idi used a risk-distribution area that is not standard IDI and "
        "cannot be preserved faithfully. Use idi(...) for the statistic and "
        "plot_reclassification(...) for visualization. See "
        "docs/migration-0.11-to-1.0.md."
    )


def auc_difference_pvalue(
    y_truth: ArrayLike,
    reference_model: ArrayLike,
    new_model: ArrayLike,
    num_bootstraps: int = 1000,
) -> tuple[float, float]:
    """Deprecated paired AUC comparison preserving its two-item tuple."""

    _deprecated("auc_difference_pvalue", "compare_auc")
    result = compare_auc(
        y_truth,
        reference_model,
        new_model,
        n_resamples=num_bootstraps,
        random_state=42,
    )
    return result.difference, result.p_value


def nri_pvalue(
    y_truth: ArrayLike,
    y_ref: ArrayLike,
    y_new: ArrayLike,
    risk_thresholds: Sequence[float],
    num_bootstraps: int = 1000,
) -> tuple[float, float]:
    """Deprecated paired NRI comparison preserving its two-item tuple."""

    _deprecated("nri_pvalue", "compare_nri")
    result = compare_nri(
        y_truth,
        y_ref,
        y_new,
        risk_thresholds,
        n_resamples=num_bootstraps,
        random_state=42,
    )
    return result.difference, result.p_value
