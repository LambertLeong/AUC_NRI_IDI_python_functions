# Changelog

All notable changes are documented here. This project follows
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-07-09

### Added

- Immutable typed results: `AUCResult`, `NRIResult`, `IDIResult`, and
  `ComparisonResult`.
- Point-estimate functions for AUC, categorical and continuous NRI, and
  standard IDI.
- Paired, outcome-stratified percentile bootstrap confidence intervals and
  pointwise ROC bands.
- Paired model-swap permutation tests with reproducible random-state control
  and finite-sample corrected p-values.
- Side-effect-free ROC comparison and reclassification plotting functions.
- Python 3.10–3.14 packaging and automated quality checks.

### Changed

- Standard IDI is defined as the change in event/non-event discrimination
  slopes.
- Continuous NRI treats ties as no movement.
- Numerical functions return named result objects rather than ad hoc tuples or
  dictionaries.
- Plotting is separate from calculation and consistently returns
  `(figure, axes)`.
- Inputs and inference parameters are validated centrally and strictly.
- The project uses a `src` package layout and PEP 517/621 metadata.

### Deprecated

- `bootstrap_results`
- `calculate_auc_with_ci`
- `plot_comparing_aucs`
- `calculate_nri`
- `category_free_nri`
- `plot_idi`
- `auc_difference_pvalue`
- `nri_pvalue`

See [the migration guide](docs/migration-0.11-to-1.0.md) for replacements.

### Removed

- Implicit plotting, printing, saving, and global random-state use in
  numerical calculations.
- Accidental package exports of implementation dependencies and typing names.

## [0.11.0] - 2025-06-20

- Added and refined p-value helpers and reproducibility tests.
- Published the final pre-1.0 interface.

[Unreleased]: https://github.com/LambertLeong/AUC_NRI_IDI_python_functions/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/LambertLeong/AUC_NRI_IDI_python_functions/compare/v0.11.0...v1.0.0
[0.11.0]: https://pypi.org/project/more-metrics/0.11.0/
