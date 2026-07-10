"""Resampling-based inference for binary prediction metrics.

Bootstrap samples are paired and stratified by outcome.  Comparison tests use
paired, per-observation model swaps, which retain the dependence between two
models evaluated on the same observations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from ._types import AUCResult, ComparisonResult, IDIResult, NRIResult
from ._validation import (
    readonly_copy,
    validate_confidence_level,
    validate_n_resamples,
)
from .metrics import auc, categorical_nri, continuous_nri, idi

ArrayLike: TypeAlias = npt.ArrayLike
RandomState: TypeAlias = int | np.random.Generator | None

_DEFAULT_FPR_GRID = np.linspace(0.0, 1.0, 101)


def _validate_fpr_grid(fpr_grid: ArrayLike | None) -> npt.NDArray[np.float64]:
    if fpr_grid is None:
        return _DEFAULT_FPR_GRID.copy()

    grid = np.asarray(fpr_grid, dtype=float)
    if grid.ndim != 1:
        raise ValueError("fpr_grid must be one-dimensional")
    if grid.size < 2:
        raise ValueError("fpr_grid must contain at least two values")
    if not np.all(np.isfinite(grid)):
        raise ValueError("fpr_grid must contain only finite values")
    if grid[0] != 0.0 or grid[-1] != 1.0:
        raise ValueError("fpr_grid must start at 0 and end at 1")
    if np.any((grid < 0.0) | (grid > 1.0)):
        raise ValueError("fpr_grid values must lie within [0, 1]")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("fpr_grid must be sorted with unique values")
    return grid.astype(np.float64, copy=True)


def _rng(random_state: RandomState) -> np.random.Generator:
    if isinstance(random_state, np.random.Generator):
        return random_state
    if random_state is not None and (
        isinstance(random_state, (bool, np.bool_))
        or not isinstance(random_state, (int, np.integer))
    ):
        raise TypeError("random_state must be an integer, Generator, or None")
    return np.random.default_rng(random_state)


def _arrays(
    y_true: ArrayLike, *predictions: ArrayLike
) -> tuple[npt.NDArray[np.int64], list[npt.NDArray[np.float64]]]:
    """Normalize inputs after a public metric has performed validation."""
    y = np.asarray(y_true, dtype=np.int64)
    values = [np.asarray(prediction, dtype=float) for prediction in predictions]
    return y, values


def _stratified_indices(
    y_true: npt.NDArray[np.int64], rng: np.random.Generator
) -> npt.NDArray[np.int64]:
    nonevents = np.flatnonzero(y_true == 0)
    events = np.flatnonzero(y_true == 1)
    sampled_nonevents = rng.choice(nonevents, size=nonevents.size, replace=True)
    sampled_events = rng.choice(events, size=events.size, replace=True)
    return np.concatenate((sampled_nonevents, sampled_events))


def _percentile_interval(
    estimates: npt.NDArray[np.float64], confidence_level: float
) -> tuple[float, float]:
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    return float(lower), float(upper)


def bootstrap_auc(
    y_true: ArrayLike,
    scores: ArrayLike,
    *,
    n_resamples: int = 1_000,
    confidence_level: float = 0.95,
    fpr_grid: ArrayLike | None = None,
    random_state: RandomState = None,
) -> AUCResult:
    """Estimate an AUC interval and pointwise ROC bands by bootstrap.

    Sampling is stratified by outcome, so every replicate has the same event
    and non-event counts as the original data.  ROC bands are percentile bands
    evaluated on ``fpr_grid`` and are pointwise rather than simultaneous.
    """
    estimate = auc(y_true, scores)
    count = validate_n_resamples(n_resamples)
    level = validate_confidence_level(confidence_level)
    grid = _validate_fpr_grid(fpr_grid)
    generator = _rng(random_state)
    y, values = _arrays(y_true, scores)
    score_values = values[0]

    auc_estimates = np.empty(count, dtype=float)
    interpolated_tpr = np.empty((count, grid.size), dtype=float)
    for index in range(count):
        sampled = _stratified_indices(y, generator)
        replicate = auc(y[sampled], score_values[sampled])
        auc_estimates[index] = replicate.auc
        interpolated_tpr[index] = np.interp(grid, replicate.fpr, replicate.tpr)

    tail = (1.0 - level) / 2.0
    tpr_lower, tpr_upper = np.quantile(interpolated_tpr, [tail, 1.0 - tail], axis=0)
    return AUCResult(
        auc=estimate.auc,
        fpr=estimate.fpr,
        tpr=estimate.tpr,
        thresholds=estimate.thresholds,
        ci=_percentile_interval(auc_estimates, level),
        fpr_grid=readonly_copy(grid),
        tpr_lower=readonly_copy(np.clip(tpr_lower, 0.0, 1.0)),
        tpr_upper=readonly_copy(np.clip(tpr_upper, 0.0, 1.0)),
    )


def bootstrap_nri(
    y_true: ArrayLike,
    reference: ArrayLike,
    new: ArrayLike,
    thresholds: Sequence[float] | ArrayLike | None = None,
    *,
    n_resamples: int = 1_000,
    confidence_level: float = 0.95,
    random_state: RandomState = None,
) -> NRIResult:
    """Estimate a percentile interval for categorical or continuous NRI."""
    if thresholds is None:
        estimate = continuous_nri(y_true, reference, new)
    else:
        estimate = categorical_nri(y_true, reference, new, thresholds)

    count = validate_n_resamples(n_resamples)
    level = validate_confidence_level(confidence_level)
    generator = _rng(random_state)
    y, values = _arrays(y_true, reference, new)
    reference_values, new_values = values
    totals = np.empty(count, dtype=float)

    for index in range(count):
        sampled = _stratified_indices(y, generator)
        if thresholds is None:
            replicate = continuous_nri(
                y[sampled], reference_values[sampled], new_values[sampled]
            )
        else:
            replicate = categorical_nri(
                y[sampled],
                reference_values[sampled],
                new_values[sampled],
                thresholds,
            )
        totals[index] = replicate.total

    return NRIResult(
        events=estimate.events,
        nonevents=estimate.nonevents,
        total=estimate.total,
        thresholds=estimate.thresholds,
        ci=_percentile_interval(totals, level),
    )


def bootstrap_idi(
    y_true: ArrayLike,
    reference: ArrayLike,
    new: ArrayLike,
    *,
    n_resamples: int = 1_000,
    confidence_level: float = 0.95,
    random_state: RandomState = None,
) -> IDIResult:
    """Estimate a paired, class-stratified percentile interval for IDI."""
    estimate = idi(y_true, reference, new)
    count = validate_n_resamples(n_resamples)
    level = validate_confidence_level(confidence_level)
    generator = _rng(random_state)
    y, values = _arrays(y_true, reference, new)
    reference_values, new_values = values
    estimates = np.empty(count, dtype=float)

    for index in range(count):
        sampled = _stratified_indices(y, generator)
        replicate = idi(y[sampled], reference_values[sampled], new_values[sampled])
        estimates[index] = replicate.idi

    return IDIResult(
        idi=estimate.idi,
        events=estimate.events,
        nonevents=estimate.nonevents,
        reference_slope=estimate.reference_slope,
        new_slope=estimate.new_slope,
        ci=_percentile_interval(estimates, level),
    )


def _swap_models(
    reference: npt.NDArray[np.float64],
    new: npt.NDArray[np.float64],
    swap: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    permuted_reference = np.where(swap, new, reference)
    permuted_new = np.where(swap, reference, new)
    return permuted_reference, permuted_new


def compare_auc(
    y_true: ArrayLike,
    reference: ArrayLike,
    new: ArrayLike,
    *,
    n_resamples: int = 1_000,
    random_state: RandomState = None,
) -> ComparisonResult:
    """Compare paired AUCs with a two-sided model-swap permutation test."""
    reference_estimate = auc(y_true, reference)
    new_estimate = auc(y_true, new)
    count = validate_n_resamples(n_resamples)
    generator = _rng(random_state)
    y, values = _arrays(y_true, reference, new)
    reference_values, new_values = values
    observed = new_estimate.auc - reference_estimate.auc

    if np.array_equal(reference_values, new_values):
        return ComparisonResult(observed, 1.0, count)

    extreme = 0
    for _ in range(count):
        swap = generator.integers(0, 2, size=y.size, dtype=np.int8).astype(bool)
        permuted_reference, permuted_new = _swap_models(
            reference_values, new_values, swap
        )
        difference = auc(y, permuted_new).auc - auc(y, permuted_reference).auc
        if abs(difference) >= abs(observed):
            extreme += 1

    return ComparisonResult(observed, (extreme + 1) / (count + 1), count)


def compare_nri(
    y_true: ArrayLike,
    reference: ArrayLike,
    new: ArrayLike,
    thresholds: Sequence[float] | ArrayLike | None = None,
    *,
    n_resamples: int = 1_000,
    random_state: RandomState = None,
) -> ComparisonResult:
    """Compare paired predictions using categorical or continuous NRI."""
    if thresholds is None:
        estimate = continuous_nri(y_true, reference, new)
    else:
        estimate = categorical_nri(y_true, reference, new, thresholds)

    count = validate_n_resamples(n_resamples)
    generator = _rng(random_state)
    y, values = _arrays(y_true, reference, new)
    reference_values, new_values = values
    observed = estimate.total

    if np.array_equal(reference_values, new_values):
        return ComparisonResult(observed, 1.0, count)

    extreme = 0
    for _ in range(count):
        swap = generator.integers(0, 2, size=y.size, dtype=np.int8).astype(bool)
        permuted_reference, permuted_new = _swap_models(
            reference_values, new_values, swap
        )
        if thresholds is None:
            difference = continuous_nri(y, permuted_reference, permuted_new).total
        else:
            difference = categorical_nri(
                y, permuted_reference, permuted_new, thresholds
            ).total
        if abs(difference) >= abs(observed):
            extreme += 1

    return ComparisonResult(observed, (extreme + 1) / (count + 1), count)


__all__ = [
    "bootstrap_auc",
    "bootstrap_idi",
    "bootstrap_nri",
    "compare_auc",
    "compare_nri",
]
