"""Typed result objects returned by :mod:`more_metrics`.

The public result types live in a dependency-light module so numerical and
plotting code can share one stable contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
ConfidenceInterval = tuple[float, float]


@dataclass(frozen=True, slots=True)
class AUCResult:
    """Area under the ROC curve and optional bootstrap uncertainty."""

    auc: float
    fpr: FloatArray
    tpr: FloatArray
    thresholds: FloatArray
    ci: ConfidenceInterval | None = None
    fpr_grid: FloatArray | None = None
    tpr_lower: FloatArray | None = None
    tpr_upper: FloatArray | None = None


@dataclass(frozen=True, slots=True)
class NRIResult:
    """Event, non-event, and total net reclassification improvement."""

    events: float
    nonevents: float
    total: float
    thresholds: tuple[float, ...] | None = None
    ci: ConfidenceInterval | None = None


@dataclass(frozen=True, slots=True)
class IDIResult:
    """Integrated discrimination improvement and its components."""

    idi: float
    events: float
    nonevents: float
    reference_slope: float
    new_slope: float
    ci: ConfidenceInterval | None = None


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Observed paired difference and model-swap permutation p-value."""

    difference: float
    p_value: float
    n_resamples: int
