# more-metrics

[![DOI](https://zenodo.org/badge/365029676.svg)](https://doi.org/10.5281/zenodo.4741234)

`more-metrics` provides statistically explicit tools for comparing binary
probability models using:

- area under the receiver operating characteristic curve (AUC);
- categorical and continuous (category-free) net reclassification improvement
  (NRI);
- standard integrated discrimination improvement (IDI);
- paired, stratified bootstrap confidence intervals; and
- paired model-swap permutation tests.

The package separates numerical calculations from plotting. Metric functions
return immutable result objects; plotting functions consume those results and
never display or save figures implicitly.

## Installation

`more-metrics` requires Python 3.10 or later.

```bash
python -m pip install more-metrics
```

Matplotlib is installed as a required dependency. To work on the project, see
[Development](#development).

## Quick start

```python
import numpy as np

from more_metrics import (
    auc,
    bootstrap_auc,
    categorical_nri,
    compare_auc,
    continuous_nri,
    idi,
)

y_true = np.array([1, 1, 1, 0, 0, 0])
reference = np.array([0.70, 0.55, 0.45, 0.40, 0.30, 0.20])
new = np.array([0.90, 0.75, 0.60, 0.35, 0.20, 0.10])

reference_auc = auc(y_true, reference)
new_auc = bootstrap_auc(
    y_true,
    new,
    n_resamples=2_000,
    confidence_level=0.95,
    random_state=42,
)
categorical = categorical_nri(y_true, reference, new, thresholds=[0.25, 0.50])
continuous = continuous_nri(y_true, reference, new)
idi_result = idi(y_true, reference, new)
comparison = compare_auc(
    y_true,
    reference,
    new,
    n_resamples=2_000,
    random_state=42,
)

print(reference_auc.auc)
print(new_auc.auc, new_auc.ci)
print(categorical.events, categorical.nonevents, categorical.total)
print(continuous.total)
print(idi_result.idi, idi_result.reference_slope, idi_result.new_slope)
print(comparison.difference, comparison.p_value)
```

All stochastic functions accept `random_state=None`, an integer seed, or a
NumPy `Generator`. Use an integer or a generator with a controlled seed for
reproducible analysis.

See [the example notebook](examples/example.ipynb) for numerical and plotting
examples.

## Result objects

- `AUCResult`: `auc`, ROC coordinates, and optional `ci` and pointwise ROC band.
- `NRIResult`: `events`, `nonevents`, `total`, and optional `ci`.
- `IDIResult`: `idi`, `events`, `nonevents`, `reference_slope`, `new_slope`,
  and optional `ci`.
- `ComparisonResult`: `difference` and `p_value`.

Confidence intervals are absent on point-estimate results and populated by
bootstrap functions.

## Definitions

Let `D=1` denote an event, `D=0` a non-event, `p_r` the reference-model
probability, and `p_n` the new-model probability.

### AUC

```
AUC = ∫ TPR(u) du
```

where `u` is the false-positive rate. AUC measures ranking discrimination,
not calibration or clinical utility. An AUC of 0.5 corresponds to chance
ranking; 1.0 corresponds to perfect ranking.

### Categorical NRI

After assigning probabilities to risk categories defined by thresholds,

```
NRI_events = P(up | D=1) - P(down | D=1)
```

```
NRI_nonevents = P(down | D=0) - P(up | D=0)
```

```
NRI = NRI_events + NRI_nonevents
```

Positive event and non-event components indicate movement in the desired
direction. Thresholds must be prespecified, unique, sorted, and strictly
between 0 and 1.

### Continuous NRI

Continuous, or category-free, NRI uses any change in predicted probability as
movement:

```
NRI_events = P(p_n > p_r | D=1) - P(p_n < p_r | D=1)
```

```
NRI_nonevents = P(p_n < p_r | D=0) - P(p_n > p_r | D=0)
```

Ties count as no movement. Continuous NRI can be large even when changes are
small and should be reported with its two components and uncertainty.

### Standard IDI

The discrimination slope of a model is

```
S(p) = E[p | D=1] - E[p | D=0]
```

Standard IDI is the change in discrimination slopes:

```
IDI = S(p_n) - S(p_r)
```

Equivalently, its event component is the change in mean probability among
events and its non-event component is the decrease in mean probability among
non-events. IDI is a scalar difference in discrimination slopes. A
risk-distribution or reclassification plot is a visualization and is not an
“IDI curve.”

## Inference

Bootstrap intervals use paired, class-stratified resampling: an observation's
reference and new predictions remain paired, while each resample preserves the
event and non-event counts. Reported intervals are percentile intervals.
Pointwise ROC bands are evaluated on a fixed false-positive-rate grid and
clipped to `[0,1]`.

Comparison p-values use paired model-swap permutations. Within each
observation, the reference and new predictions are randomly exchanged to form
the null distribution. Finite-sample p-values use
`(b+1)/(B+1)`, so they are never zero.

## Limitations

- Inputs must describe a non-empty binary outcome containing both 0 and 1.
  Predictions must be finite; probabilities used by NRI and IDI must lie in
  `[0,1]`.
- AUC, NRI, and IDI answer different questions. None establishes calibration,
  causal benefit, or clinical usefulness.
- NRI depends strongly on the selected thresholds. Continuous NRI has no
  clinically meaningful categories and is especially easy to overinterpret.
- IDI is sensitive to prevalence and the prediction scale.
- Percentile bootstrap intervals and pointwise ROC bands are not simultaneous
  confidence bands and may perform poorly in very small samples.
- Inference assumes observations are independent. Clustered, censored,
  weighted, or externally validated data require methods not implemented here.

## Migrating from 0.11

Version 1.0 corrects metric definitions, makes randomness explicit, returns
typed result objects, and removes side effects from numerical functions. It
therefore includes intentional breaking changes. See the
[0.11-to-1.0 migration guide](docs/migration-0.11-to-1.0.md).

## Citation

If this software contributes to published work, use the project citation
metadata in [CITATION.cff](CITATION.cff):

> Lambert T. Leong. *more-metrics: AUC, NRI, and IDI for binary probability
> models*. Zenodo. <https://doi.org/10.5281/zenodo.4741234>

The project was motivated by the analysis in
[Leong et al. (2021)](https://doi.org/10.1038/s43856-021-00024-0).

## Development

```bash
git clone https://github.com/LambertLeong/AUC_NRI_IDI_python_functions.git
cd AUC_NRI_IDI_python_functions
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy src
pytest
python -m build
python -m twine check dist/*
```

Contribution expectations are in [CONTRIBUTING.md](CONTRIBUTING.md). The
project is licensed under GPL-3.0.
