# Migrating from 0.11 to 1.0

Version 1.0 replaces loosely structured return values and calculation/plotting
side effects with a small typed API. Statistical correctness takes priority
over reproducing 0.11 numerical output.

## Installation and imports

The distribution name remains `more-metrics` and the import name remains
`more_metrics`:

```bash
python -m pip install --upgrade more-metrics
```

```python
from more_metrics import auc, continuous_nri, idi
```

Version 1.0 supports Python 3.10 through 3.14. Import only documented public
names; implementation dependencies are no longer exposed from the package.

## Replacement table

| 0.11 interface | 1.0 replacement |
| --- | --- |
| `calculate_auc_with_ci(...)` | `auc(...)` for an estimate; `bootstrap_auc(...)` for an estimate and interval |
| `plot_comparing_aucs(...)` | calculate results, then call `plot_roc_comparison(reference_result, new_result)` |
| `calculate_nri(..., thresholds)` | `categorical_nri(..., thresholds)` |
| `category_free_nri(...)` | `continuous_nri(...)` |
| `plot_idi(...)` | `idi(...)` for standard IDI; `plot_reclassification(...)` for visualization |
| `auc_difference_pvalue(...)` | `compare_auc(...)` |
| `nri_pvalue(...)` | `compare_nri(...)` |
| `bootstrap_results(...)` | the metric-specific `bootstrap_auc`, `bootstrap_nri`, or `bootstrap_idi` |

Deprecated wrappers emit `DeprecationWarning` and give replacement guidance.
They are transitional. Code should move to the 1.0 functions rather than
silencing the warnings. Where corrected behavior cannot be represented by a
legacy return shape, the wrapper fails with an actionable error.

## Typed results replace tuples and dictionaries

Access fields by name:

```python
auc_result = bootstrap_auc(y_true, scores, random_state=42)
print(auc_result.auc, auc_result.ci)

nri_result = categorical_nri(y_true, reference, new, [0.10, 0.20])
print(nri_result.events, nri_result.nonevents, nri_result.total)

idi_result = idi(y_true, reference, new)
print(
    idi_result.idi,
    idi_result.events,
    idi_result.nonevents,
    idi_result.reference_slope,
    idi_result.new_slope,
)

comparison = compare_auc(y_true, reference, new, random_state=42)
print(comparison.difference, comparison.p_value)
```

Result objects are immutable. A point estimate has no confidence interval;
bootstrap functions return the corresponding result with `ci` populated.

## Corrected definitions

Standard IDI is now the difference between the new and reference
discrimination slopes:

\[
\left(\bar p_{n,1}-\bar p_{n,0}\right)
-\left(\bar p_{r,1}-\bar p_{r,0}\right).
\]

Risk-distribution plots are no longer described as an “IDI curve.” Calculate
IDI with `idi` and create a distinct visualization with
`plot_reclassification`.

Continuous NRI treats equal reference and new probabilities as no movement.
Categorical NRI validates ordered clinical thresholds and reports event and
non-event components separately. These corrections can change output relative
to 0.11.

## Inference and reproducibility

Replace implicit global randomness with `random_state`:

```python
result = bootstrap_idi(
    y_true,
    reference,
    new,
    n_resamples=5_000,
    confidence_level=0.95,
    random_state=2025,
)
```

Bootstrap resampling is paired and stratified by outcome. Comparisons use
paired model-swap permutation tests. Their finite-sample p-values use
`(extreme + 1) / (n_resamples + 1)`.

## Plotting

Plotting is now explicit and composable:

```python
reference_result = bootstrap_auc(y_true, reference, random_state=1)
new_result = bootstrap_auc(y_true, new, random_state=2)
fig, ax = plot_roc_comparison(reference_result, new_result)
fig.savefig("roc-comparison.png", dpi=150)
```

Plot functions return `(figure, axes)`. They do not call `show`, print values,
or save files. Pass `ax=` to draw into an existing Matplotlib axes.

## Stricter validation

Version 1.0 rejects:

- non-one-dimensional, empty, unequal-length, or non-finite inputs;
- outcomes other than exactly 0 and 1, or data missing either class;
- NRI or IDI probabilities outside `[0, 1]`;
- categorical thresholds that are unsorted, duplicated, or outside `(0, 1)`;
- invalid confidence levels or non-positive resample counts.

Validate upstream data rather than relying on implicit coercion.
