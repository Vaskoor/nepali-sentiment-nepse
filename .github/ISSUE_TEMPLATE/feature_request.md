---
name: Feature request
about: Suggest a new capability, model, dataset, or metric
title: "[feat] "
labels: enhancement, triage
---

## Motivation
<!-- What research question or gap does this address? Cite related work if relevant. -->

## Proposal
<!-- Describe the proposed feature at a high level. -->

## Where does it fit?
<!-- Which phase / module does this touch? e.g. src/nepse_nlp/calibration.py -->

- [ ] Phase 1 - data & preprocessing
- [ ] Phase 2 - Jev benchmark / calibration
- [ ] Phase 3 - hybrid pipeline
- [ ] Phase 4 - NEPSE prediction
- [ ] Phase 5 - backtest / deployment
- [ ] Phase 6 - paper / reproducibility
- [ ] Cross-cutting (CI, docs, packaging)

## Reproducibility / honesty checklist
- [ ] Any new stochastic component threads `set_seed(seed)`.
- [ ] Any new prediction code respects no-look-ahead walk-forward.
- [ ] Any new backtest figure includes realistic NEPSE transaction costs.
- [ ] Adds unit tests (numpy/pandas/sklearn/xgboost only, no network).
- [ ] Offline-runnable (simulated fallback when live data/API unavailable).

## Alternatives considered
<!-- What else did you consider and why is this better? -->
