# Project Ouroboros

Optimal operating points, state-dependent Cramér–Rao bounds, nuisance robustness, and unbinned maximum-likelihood recovery of static and time-varying signals in shot-to-shot phase-randomized differential atom interferometers.

## Repository Layout
- `configs/` — Baseline interferometer parameter configuration.
- `src/ouroboros/` — Core marginal likelihood, dual-anchor high-$N$ Cramér–Rao bounds, closed-form $N^{-3/10}$ optimal-bias and $5\times 5$ nuisance Fisher laws, and unbinned static/AC/time-blocked MLE and ellipse comparators.
- `scripts/` — Reproducible theoretical benchmark and Zenodo data-analysis scripts.
- `tests/` — Automated verification tests (`pytest`).

Raw Zenodo CSV datasets (DOI `10.5281/zenodo.19592552`) are downloaded separately and are not bundled in this repository.

## Quick Start
```bash
uv sync --all-extras
uv run ruff check src tests scripts
uv run pytest -v
uv run python scripts/reproduce_crb.py
uv run python scripts/reproduce_theory_benchmarks.py
```

After downloading the CSV archive from DOI `10.5281/zenodo.19592552`, run the Zenodo reanalysis with:
```bash
uv run python scripts/analyze_baynham_zenodo.py --data-dir /path/to/zenodo_19592552/data
```
