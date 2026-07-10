from __future__ import annotations

from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import pytest

import more_metrics as mm

Y_TRUE = [0, 0, 1, 1]
REFERENCE = [0.30, 0.40, 0.60, 0.70]
NEW = [0.10, 0.20, 0.80, 0.90]
THRESHOLDS = [0.50]


def deprecated_call(function: Callable[..., Any], *args: object, **kwargs: object):
    message = (
        rf"{function.__name__} is deprecated.*use .*"
        r"docs/migration-0\.11-to-1\.0\.md"
    )
    with pytest.warns(DeprecationWarning, match=message):
        return function(*args, **kwargs)


@pytest.mark.parametrize(
    ("function", "args", "replacement"),
    [
        (mm.bootstrap_results, (Y_TRUE, REFERENCE), "bootstrap_auc"),
        (mm.calculate_auc_with_ci, (Y_TRUE, REFERENCE), "bootstrap_auc"),
        (
            mm.plot_idi,
            (Y_TRUE, REFERENCE, NEW, THRESHOLDS),
            "plot_reclassification",
        ),
    ],
)
def test_incompatible_wrappers_fail_with_migration_guidance(
    function: Callable[..., Any],
    args: tuple[object, ...],
    replacement: str,
) -> None:
    with pytest.raises(RuntimeError, match=replacement):
        deprecated_call(function, *args)


def test_nri_wrappers_preserve_legacy_three_item_tuples() -> None:
    categorical = deprecated_call(
        mm.calculate_nri,
        Y_TRUE,
        REFERENCE,
        NEW,
        THRESHOLDS,
    )
    continuous = deprecated_call(
        mm.category_free_nri,
        Y_TRUE,
        REFERENCE,
        NEW,
    )

    assert categorical == (0.0, 0.0, 0.0)
    assert continuous == (1.0, 1.0, 2.0)
    assert len(categorical) == len(continuous) == 3


def test_comparison_wrappers_preserve_legacy_two_item_tuples() -> None:
    auc_comparison = deprecated_call(
        mm.auc_difference_pvalue,
        Y_TRUE,
        REFERENCE,
        NEW,
        num_bootstraps=9,
    )
    nri_comparison = deprecated_call(
        mm.nri_pvalue,
        Y_TRUE,
        REFERENCE,
        NEW,
        THRESHOLDS,
        num_bootstraps=9,
    )

    assert auc_comparison == (0.0, 1.0)
    assert nri_comparison == (0.0, 1.0)
    assert len(auc_comparison) == len(nri_comparison) == 2


def test_plot_comparing_aucs_returns_figure_and_axes() -> None:
    figure, axes = deprecated_call(
        mm.plot_comparing_aucs,
        Y_TRUE,
        REFERENCE,
        NEW,
        n_bootstraps=9,
    )

    assert axes.figure is figure
    plt.close(figure)


def test_plot_comparing_aucs_rejects_implicit_saving() -> None:
    with pytest.raises(ValueError, match=r"fig\.savefig"):
        deprecated_call(
            mm.plot_comparing_aucs,
            Y_TRUE,
            REFERENCE,
            NEW,
            n_bootstraps=9,
            save=True,
        )


def test_public_exports_include_wrappers_without_accidental_modules() -> None:
    wrappers = {
        "auc_difference_pvalue",
        "bootstrap_results",
        "calculate_auc_with_ci",
        "calculate_nri",
        "category_free_nri",
        "nri_pvalue",
        "plot_comparing_aucs",
        "plot_idi",
    }

    assert wrappers <= set(mm.__all__)
    assert {"np", "plt", "sys", "Sequence"}.isdisjoint(mm.__all__)
    assert set(mm.__all__) <= set(dir(mm))
