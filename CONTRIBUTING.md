# Contributing to Nepali Sentiment · NEPSE

First off — thank you for considering a contribution. This is a research
codebase and reproducibility is the top priority, so this guide is short
but firm about a few things.

## Ground rules

- **Reproducibility first.** Every stochastic component must honour
  `nepse_nlp.utils.set_seed(seed)`. If you add randomness, thread the seed
  through. Never rely on global state for RNG.
- **No look-ahead.** Any model-evaluation code must train strictly on
  `[0, t-1]` when predicting `t`. The existing `tests/test_features.py::
  test_build_feature_matrix_no_lookahead` guards this; do not weaken it.
- **Honest reporting.** Backtest figures must include realistic NEPSE
  transaction costs (`backtest.NEPSE_COSTS`). Do not report results on
  simulated data as if they were live-market results.
- **Tests stay green.** `pytest tests` must pass on Python 3.10 / 3.11 /
  3.12 (the CI matrix). Do not merge if any test is red.

## Development setup

```bash
git clone https://github.com/Vaskoor/nepali-sentiment-nepse.git
cd nepali-sentiment-nepse
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + pytest/ruff/black/mypy
pip install -e ".[deep,api,scrape,track]"   # optional extras (torch, fastapi, ...)
pre-commit install               # activate the git hooks
```

## Typical workflow

1. **Branch:** `git switch -c feat/<short-description>` (or `fix/`, `docs/`).
2. **Make changes.** Keep functions small, type-hinted, and docstring'd.
3. **Run checks locally** before pushing:

   ```bash
   make lint       # ruff check + black --check
   make test       # pytest
   make pipeline   # end-to-end phase1->5 (offline, deterministic)
   ```

4. **Commit** using [Conventional Commits](https://www.conventionalcommits.org):
   `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `ci:`. The
   `pre-commit` hooks will auto-format + lint on commit.
5. **Open a PR.** Fill in the PR template; the CI matrix runs on push.

## Code style

- **Formatter:** `black` (line length 100). `ruff format` is also accepted.
- **Linter:** `ruff` with `select = ["E","F","I"]` (see `pyproject.toml`).
- **Types:** the package ships a `py.typed` marker (PEP 561); public
  functions must have type hints. `mypy` is configured but not blocking.

## Adding a new phase or experiment

1. Put the implementation under `src/nepse_nlp/` (keep it import-safe without
   optional deps — guard `torch`, `transformers`, etc. behind `try/except`).
2. Add an entry-point script under `scripts/phaseN_*.py` (offline-runnable
   with a deterministic simulated fallback).
3. Add unit tests under `tests/` (stdlib + numpy/pandas/sklearn/xgboost only).
4. Update `results/` outputs and the README results table.
5. Bump the version in `pyproject.toml` and `src/nepse_nlp/__init__.py`,
   and add a `CHANGELOG.md` entry.

## Reporting issues

Use the issue templates (Bug report / Feature request). For security issues,
see [`SECURITY.md`](SECURITY.md) — do **not** open a public issue.

## Citation

If this work contributes to your research, please cite via
[`CITATION.cff`](CITATION.cff) (GitHub renders a "Cite this repository"
button from it).
