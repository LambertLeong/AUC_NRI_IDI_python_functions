# 1.0 release-readiness report

## Verification status

The 1.0 implementation is ready for release preparation. The local integration
gate passed on Python 3.12:

- 105 tests with 95.89% branch coverage;
- Ruff lint and formatting checks;
- strict mypy checking;
- deterministic notebook execution;
- isolated PEP 517 wheel and source-distribution builds;
- Twine metadata validation; and
- independent wheel and source-distribution installation and import checks.

The Python 3.10–3.14 CI matrix remains the required cross-version gate.

## Corrected defects

- IDI now means the change in event/non-event discrimination slopes.
- Continuous NRI treats ties as no movement.
- Bootstrap samples remain paired and preserve event and non-event counts.
- ROC uncertainty uses percentile pointwise bands on a fixed false-positive-rate
  grid, clipped to the unit interval.
- AUC and NRI comparisons use paired model-swap permutations and the
  finite-sample p-value `(extreme + 1) / (n_resamples + 1)`.
- Stochastic APIs accept explicit random-state control and do not use global
  random state.
- Numerical functions no longer print, plot, display, or save as side effects.

## Intentional breaking changes

- Point estimates and inference return frozen typed result objects rather than
  unlabelled tuples or dictionaries.
- Input validation now requires one-dimensional, equal-length, finite data,
  both binary outcome classes, valid probabilities for NRI/IDI, and strictly
  ordered categorical thresholds within `(0, 1)`.
- Plotting is separate from numerical calculation and always returns
  `(figure, axes)`.
- The package uses a `src` layout and supports Python 3.10 or newer.
- Legacy bootstrap and `plot_idi` outputs that cannot represent corrected
  behavior fail with migration guidance.

## Deprecated interfaces

The following wrappers emit actionable `DeprecationWarning` messages:
`bootstrap_results`, `calculate_auc_with_ci`, `plot_comparing_aucs`,
`calculate_nri`, `category_free_nri`, `plot_idi`,
`auc_difference_pvalue`, and `nri_pvalue`.

Legacy NRI and comparison tuple shapes are preserved where they can safely
represent corrected behavior. See the
[migration guide](migration-0.11-to-1.0.md).

## Statistical methods and limitations

Confidence intervals are paired, class-stratified percentile bootstrap
intervals. ROC bands are pointwise, not simultaneous. Comparison tests are
Monte Carlo paired model-swap tests rather than exhaustive tests. These methods
assume independent observations and do not support clustered, censored, or
weighted data. AUC, NRI, and IDI measure discrimination or reclassification;
they do not establish calibration, causal benefit, or clinical utility.

## Remaining release actions

1. Run and pass the GitHub Actions matrix on Python 3.10–3.14.
2. Review the final version and changelog, then create the corresponding Git
   tag only after explicit release authorization.
3. Configure or confirm the `pypi` trusted-publishing environment.
4. Publish through the tag-driven workflow; do not upload manually.
5. Create the 1.0 archival record and update citation metadata if the archive
   issues a version-specific DOI.
