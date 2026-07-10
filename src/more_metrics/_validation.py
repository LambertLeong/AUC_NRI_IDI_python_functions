"""Shared normalization and validation for numerical metric inputs."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral, Number, Real
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]


def _as_float_array(values: npt.ArrayLike, *, name: str) -> FloatArray:
    """Return a copied one-dimensional finite float array."""
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError) as exc:
        message = f"{name} must be a one-dimensional numeric array-like"
        raise ValueError(message) from exc

    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if raw.dtype.kind in {"U", "S"}:
        raise ValueError(f"{name} must contain numeric values")
    if raw.dtype.kind == "O" and not all(isinstance(value, Number) for value in raw):
        raise ValueError(f"{name} must contain numeric values")
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise ValueError(f"{name} must contain real numeric values")

    try:
        array = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def validate_binary_inputs(
    y_true: npt.ArrayLike,
    *predictions: npt.ArrayLike,
    prediction_names: Sequence[str] | None = None,
    probabilities: bool = False,
) -> tuple[FloatArray, ...]:
    """Normalize an outcome and one or more paired prediction arrays.

    Outcomes must contain exactly the values zero and one, including both
    classes. Boolean outcomes therefore satisfy the same contract.
    """
    y = _as_float_array(y_true, name="y_true")
    if y.size == 0:
        raise ValueError("inputs must not be empty")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("y_true must contain only binary values 0 and 1")
    if not np.any(y == 0.0) or not np.any(y == 1.0):
        raise ValueError("y_true must contain both classes 0 and 1")

    if prediction_names is None:
        names = tuple(f"prediction_{index}" for index in range(len(predictions)))
    else:
        names = tuple(prediction_names)
        if len(names) != len(predictions):
            raise ValueError("prediction_names must match the prediction inputs")

    normalized: list[FloatArray] = [y]
    for values, name in zip(predictions, names, strict=True):
        prediction = _as_float_array(values, name=name)
        if prediction.size != y.size:
            raise ValueError(f"{name} and y_true must have equal lengths")
        if probabilities and np.any((prediction < 0.0) | (prediction > 1.0)):
            raise ValueError(f"{name} must contain probabilities within [0, 1]")
        normalized.append(prediction)

    return tuple(normalized)


def validate_thresholds(thresholds: npt.ArrayLike) -> FloatArray:
    """Normalize categorical NRI cut points."""
    values = _as_float_array(thresholds, name="thresholds")
    if values.size == 0:
        raise ValueError("thresholds must not be empty")
    if np.any((values <= 0.0) | (values >= 1.0)):
        raise ValueError("thresholds must lie strictly within (0, 1)")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("thresholds must be sorted and unique")
    return values


def validate_n_resamples(n_resamples: object) -> int:
    """Validate a positive integer resample count."""
    if isinstance(n_resamples, (bool, np.bool_)) or not isinstance(
        n_resamples, Integral
    ):
        raise ValueError("n_resamples must be a positive integer")
    result = int(n_resamples)
    if result <= 0:
        raise ValueError("n_resamples must be a positive integer")
    return result


def validate_confidence_level(confidence_level: object) -> float:
    """Validate a finite confidence level strictly between zero and one."""
    if isinstance(confidence_level, (bool, np.bool_)) or not isinstance(
        confidence_level, Real
    ):
        raise ValueError("confidence_level must be a number within (0, 1)")
    result = float(confidence_level)
    if not np.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError("confidence_level must be within (0, 1)")
    return result


def readonly_copy(values: npt.ArrayLike) -> FloatArray:
    """Return a float copy whose write flag is disabled."""
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result
