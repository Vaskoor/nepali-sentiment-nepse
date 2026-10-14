<!-- Thanks for the PR!  Fill in the checklist; a maintainer will review. -->

## Summary
<!-- What does this PR change and why? Reference the issue if applicable: Closes #N -->

## Type of change
- [ ] `feat` — new capability (phase / model / metric)
- [ ] `fix` — bug fix (behavior change)
- [ ] `docs` — documentation only
- [ ] `test` — tests only
- [ ] `refactor` — no behavior change
- [ ] `ci`/`chore` — tooling, packaging, CI

## Reproducibility & honesty checklist
<!-- These are mandatory for any change touching models, data, or evaluation. -->
- [ ] New stochastic components thread `nepse_nlp.utils.set_seed(seed)`.
- [ ] New prediction code respects no-look-ahead walk-forward
      (`tests/test_features.py::test_build_feature_matrix_no_lookahead` still passes).
- [ ] Any new backtest figure includes realistic NEPSE transaction costs
      (`backtest.NEPSE_COSTS`: brokerage 0.4% + SEBON + DP fee + capital-gains).
- [ ] No live-market results reported as if from simulated data (or clearly labelled).
- [ ] `pytest tests` passes on Python 3.10 / 3.11 / 3.12.
- [ ] `ruff check` and `black --check` are clean.
- [ ] `make pipeline` still runs end-to-end offline (simulated fallback).
- [ ] If a new phase/experiment: added a `scripts/phaseN_*.py` entry point.
- [ ] Bumped version in `pyproject.toml` + `src/nepse_nlp/__init__.py` and
      added a `CHANGELOG.md` entry.

## Screenshots / output
<!-- If you changed a chart or a results table, paste the new output. -->

## Notes for reviewers
<!-- Anything reviewers should pay attention to, or that needs a decision. -->
