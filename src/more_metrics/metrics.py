"""Side-effect-free point estimates for binary prediction models."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_auc_score, roc_curve

from ._types import AUCResult, IDIResult, NRIResult
from ._validation import readonly_copy, validate_binary_inputs, validate_thresholds


def auc(y_true: npt.ArrayLike, scores: npt.ArrayLike) -> AUCResult:
    """Calculate ROC AUC and its ROC coordinates.

    Scores need only be finite; unlike risk probabilities they are not
    restricted to the unit interval.
    """
    y, prediction = validate_binary_inputs(
        y_true,
        scores,
        prediction_names=("scores",),
    )
    fpr, tpr, thresholds = roc_curve(y, prediction, drop_intermediate=True)
    if fpr.size > 2:
        # Keep the origin plus the final point at each repeated FPR. Recent
        # scikit-learn releases retain additional collinear vertical points,
        # which otherwise makes this public coordinate sequence version
        # dependent.
        keep = np.r_[True, fpr[1:-1] != fpr[2:], True]
        fpr = fpr[keep]
        tpr = tpr[keep]
        thresholds = thresholds[keep]
    estimate = float(roc_auc_score(y, prediction))
    return AUCResult(
        auc=estimate,
        fpr=readonly_copy(fpr),
        tpr=readonly_copy(tpr),
        thresholds=readonly_copy(thresholds),
    )


def categorical_nri(
    y_true: npt.ArrayLike,
    reference: npt.ArrayLike,
    new: npt.ArrayLike,
    thresholds: npt.ArrayLike,
) -> NRIResult:
    """Calculate categorical net reclassification improvement.

    ``thresholds`` partition probabilities using right-open risk categories:
    a probability equal to a threshold belongs to the category above it.
    """
    y, reference_probability, new_probability = validate_binary_inputs(
        y_true,
        reference,
        new,
        prediction_names=("reference", "new"),
        probabilities=True,
    )
    cut_points = validate_thresholds(thresholds)
    reference_category = np.searchsorted(
        cut_points, reference_probability, side="right"
    )
    new_category = np.searchsorted(cut_points, new_probability, side="right")
    movement = np.sign(new_category - reference_category)

    event_movement = movement[y == 1.0]
    nonevent_movement = movement[y == 0.0]
    events = float(np.mean(event_movement))
    nonevents = float(-np.mean(nonevent_movement))
    return NRIResult(
        events=events,
        nonevents=nonevents,
        total=events + nonevents,
        thresholds=tuple(float(value) for value in cut_points),
    )


def continuous_nri(
    y_true: npt.ArrayLike,
    reference: npt.ArrayLike,
    new: npt.ArrayLike,
) -> NRIResult:
    """Calculate continuous (category-free) NRI, with ties as no movement."""
    y, reference_probability, new_probability = validate_binary_inputs(
        y_true,
        reference,
        new,
        prediction_names=("reference", "new"),
        probabilities=True,
    )
    movement = np.sign(new_probability - reference_probability)
    events = float(np.mean(movement[y == 1.0]))
    nonevents = float(-np.mean(movement[y == 0.0]))
    return NRIResult(
        events=events,
        nonevents=nonevents,
        total=events + nonevents,
    )


def idi(
    y_true: npt.ArrayLike,
    reference: npt.ArrayLike,
    new: npt.ArrayLike,
) -> IDIResult:
    """Calculate standard integrated discrimination improvement.

    IDI is the new model's discrimination slope minus the reference model's
    discrimination slope.
    """
    y, reference_probability, new_probability = validate_binary_inputs(
        y_true,
        reference,
        new,
        prediction_names=("reference", "new"),
        probabilities=True,
    )
    events = y == 1.0
    nonevents = ~events

    reference_event_mean = float(np.mean(reference_probability[events]))
    reference_nonevent_mean = float(np.mean(reference_probability[nonevents]))
    new_event_mean = float(np.mean(new_probability[events]))
    new_nonevent_mean = float(np.mean(new_probability[nonevents]))

    event_component = new_event_mean - reference_event_mean
    nonevent_component = reference_nonevent_mean - new_nonevent_mean
    reference_slope = reference_event_mean - reference_nonevent_mean
    new_slope = new_event_mean - new_nonevent_mean
    return IDIResult(
        idi=event_component + nonevent_component,
        events=event_component,
        nonevents=nonevent_component,
        reference_slope=reference_slope,
        new_slope=new_slope,
    )
