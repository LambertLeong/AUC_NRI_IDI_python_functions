"""Public API for more-metrics."""

from ._types import AUCResult, ComparisonResult, IDIResult, NRIResult
from .compatibility import (
    auc_difference_pvalue,
    bootstrap_results,
    calculate_auc_with_ci,
    calculate_nri,
    category_free_nri,
    nri_pvalue,
    plot_comparing_aucs,
    plot_idi,
)
from .inference import (
    bootstrap_auc,
    bootstrap_idi,
    bootstrap_nri,
    compare_auc,
    compare_nri,
)
from .metrics import auc, categorical_nri, continuous_nri, idi
from .plotting import plot_reclassification, plot_roc_comparison

__version__ = "1.0.0"

__all__ = [
    "AUCResult",
    "ComparisonResult",
    "IDIResult",
    "NRIResult",
    "__version__",
    "auc",
    "auc_difference_pvalue",
    "bootstrap_auc",
    "bootstrap_idi",
    "bootstrap_nri",
    "bootstrap_results",
    "calculate_auc_with_ci",
    "calculate_nri",
    "categorical_nri",
    "category_free_nri",
    "compare_auc",
    "compare_nri",
    "continuous_nri",
    "idi",
    "nri_pvalue",
    "plot_comparing_aucs",
    "plot_idi",
    "plot_reclassification",
    "plot_roc_comparison",
]
