# Contributing

Contributions should preserve the statistical meaning, reproducibility, and
side-effect boundaries of the public API.

## Set up a development environment

```bash
git clone https://github.com/LambertLeong/AUC_NRI_IDI_python_functions.git
cd AUC_NRI_IDI_python_functions
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Make a change

1. Open an issue for statistical behavior changes or substantial API changes.
2. Keep metric calculations independent from plotting, console output, file
   output, and global random state.
3. Add hand-checkable tests for formulas and regression tests for defects.
4. Document public behavior and intentional compatibility changes.
5. Do not update expected values merely to match an implementation. Explain
   and independently verify every statistical expectation.

Use synthetic, de-identified, or redistributable test data only. Never commit
clinical or personal data.

## Run the quality checks

```bash
ruff check .
ruff format --check .
mypy src
pytest
python -m build
python -m twine check dist/*
```

Execute the example notebook against an installed wheel when changing the
public API or documentation. Generated caches, notebook outputs, coverage
files, and build artifacts should not be committed.

## Statistical review checklist

- Inputs remain paired when comparing models.
- Bootstrap samples preserve event and non-event counts.
- Permutation tests construct the null by swapping paired model predictions.
- Event and non-event component signs match their documented definitions.
- Ties, identical models, degenerate predictions, and boundary probabilities
  are tested.
- Stochastic APIs accept `random_state` and are reproducible.
- Uncertainty methods and their limitations are documented.

## Pull requests

Keep pull requests focused. Describe the user-visible change, statistical
rationale, tests run, compatibility impact, and unresolved limitations.
Maintainers may request an independent reference calculation for changes to a
metric or inference method.

By contributing, you agree that your contribution is licensed under
GPL-3.0, the repository's license.
