# Nepali Sentiment — NEPSE Stock Prediction

**Sentiment Analysis of Nepali News for NEPSE Stock Prediction Using Jev (System One Decision Model) and Confidence-Gated Hybrid Architectures**

An end-to-end, reproducible research pipeline for low-resource Nepali financial sentiment analysis, calibration-aware confidence-gated hybrid inference, and sentiment-augmented next-day NEPSE index prediction with rigorous walk-forward backtesting.

> **Status:** Research codebase, MIT-licensed, CI-pipelined.
> **Substitution note:** If the **Jev** API is unavailable, this repo uses a deterministic, lexicon-driven *simulated* backend that exposes the exact same `classify(text) -> {label, confidence, probs}` contract. Switch to the real API by setting `backend="http"` + `JEV_API_KEY`. See `src/nepse_nlp/jev_benchmark.py`.

---

## Table of contents

1. [What this project does](#what-this-project-does)
2. [Repository layout](#repository-layout)
3. [Quick start](#quick-start)
4. [The 6-phase research plan](#the-6-phase-research-plan)
5. [Configuration](#configuration)
6. [Running the pipeline](#running-the-pipeline)
7. [API & dashboard](#api--dashboard)
8. [Docker](#docker)
9. [CI/CD pipeline](#cicd-pipeline)
10. [Evaluation metrics](#evaluation-metrics)
11. [Reproducibility & ethics](#reproducibility--ethics)
12. [Citation](#citation)

---

## What this project does

| Phase | Capability |
|-------|------------|
| 1 | Devanagari-aware preprocessing + async news scraping + NEPSE OHLCV collection (with deterministic simulated fallback so the pipeline always runs offline). |
| 2 | Jev **zero-shot** sentiment benchmark vs. NepaliBERT / MuRIL / XLM-R / GPT-4o-simulated baselines, including **Brier score**, **ECE**, reliability diagrams, and code-mix / sector stratification. |
| 3 | **Confidence-gated hybrid pipeline** (Jev fast → escalate to a heavier model below a **conformally-calibrated** threshold), with a **cost–accuracy Pareto frontier**. |
| 4 | Sentiment-augmented next-day NEPSE prediction (XGBoost / LogReg / LSTM) with **walk-forward** validation, feature importance, and a **sentiment-ablation** study. |
| 5 | Walk-forward **backtesting** with realistic NEPSE transaction costs (brokerage 0.4% + SEBON + DP fee + capital-gains), Sharpe / max-drawdown, plus **FastAPI** service + **Streamlit** dashboard. |
| 6 | LaTeX paper draft, MLflow experiment tracking, MIT release, GitHub Actions CI/CD. |

### Research questions

1. How well does Jev perform zero-shot sentiment classification on Nepali financial news vs. Nepali-specific and multilingual baselines?
2. Can a confidence-gated hybrid architecture beat a single large model on the **cost–accuracy frontier**?
3. Does sentiment augmentation improve NEPSE index prediction over time-series baselines?
4. How do **ordinal-scale bias**, **code-mixing**, and **domain shift** affect Jev's calibration in Nepali?

---

## Repository layout

```
nepali-sentiment-nepse/
├── .github/workflows/      # CI (ci.yml), Docker (docker.yml), CodeQL (codeql.yml)
├── api/                    # FastAPI service (Phase 5)
├── config/config.yaml      # single source of truth for all hyper-params
├── dashboard/              # Streamlit dashboard (Phase 5)
├── data/{raw,processed,external}/   # data (git-ignored except .gitkeep)
├── docs/architecture.md    # system design
├── models/                 # trained artifacts (git-ignored)
├── mlruns/                 # MLflow runs (git-ignored)
├── paper/main.tex          # LaTeX paper draft (Phase 6)
├── results/                # pipeline outputs (git-ignored except .gitkeep)
├── scripts/                # phase entry points: phase1..phase5
├── src/nepse_nlp/          # the Python package
│   ├── data_preprocessing.py
│   ├── scraper.py
│   ├── jev_benchmark.py
│   ├── baselines.py
│   ├── calibration.py
│   ├── conformal.py
│   ├── hybrid_pipeline.py
│   ├── features.py
│   ├── prediction.py
│   ├── backtest.py
│   └── utils.py
├── tests/                  # pytest unit tests (core + calibration + features)
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── requirements.txt
├── LICENSE                 # MIT
└── README.md
```

---

## Quick start

```bash
# 1. clone (replace YOUR-USERNAME after pushing)
git clone https://github.com/YOUR-USERNAME/nepali-sentiment-nepse.git
cd nepali-sentiment-nepse

# 2. create env & install
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # core + dev deps (numpy/pandas/sklearn/xgboost/pytest/ruff/black)

# 3. run the full pipeline end-to-end (offline, deterministic)
make pipeline                    # runs phase1 -> phase5

# 4. inspect outputs
ls results/                      # phase2_benchmark.json, phase3_pareto.json,
                                 # phase4_models.json, phase5_backtest.json, equity_curve.csv

# 5. run tests
make test
```

> The pipeline is **fully offline**: when network scraping is unavailable, `scraper.py` falls back to deterministic simulated Nepali financial headlines and a NEPSE-like OHLCV series (GARCH-ish volatility). All results are seed-stable (`seed=42`) so commits diff cleanly.

---

## The 6-phase research plan

| Phase | Module(s) | Script | Key output |
|-------|-----------|--------|------------|
| 1 — Data & preprocessing | `data_preprocessing`, `scraper` | `scripts/phase1_preprocess.py` | `data/processed/news_sentiment.csv`, `nepse_daily.csv` |
| 2 — Jev zero-shot benchmark | `jev_benchmark`, `baselines`, `calibration` | `scripts/phase2_benchmark.py` | `results/phase2_benchmark.json` (acc, F1, Brier, ECE per model) |
| 3 — Confidence-gated hybrid | `hybrid_pipeline`, `conformal` | `scripts/phase3_hybrid.py` | `results/phase3_pareto.json` (Pareto frontier + conformal threshold) |
| 4 — Sentiment-augmented prediction | `features`, `prediction` | `scripts/phase4_predict.py` | `results/phase4_models.json` (walk-forward metrics + ablation) |
| 5 — Backtesting & deployment | `backtest`, `api/main`, `dashboard/app` | `scripts/phase5_backtest.py` | `results/phase5_backtest.json`, `equity_curve.csv` |
| 6 — Paper & reproducibility | `paper/main.tex`, `pyproject.toml`, CI | — | LaTeX paper draft + containerized + CI-pipelined |

### 18-month Gantt (text)

```
2025        Q1   Q2   Q3   Q4 | 2026        Q1   Q2   Q3   Q4
Phase 1  |████████          |                  |                 |
Phase 2  |        ████████  |                  |                 |
Phase 3  |              ████|████              |                 |
Phase 4  |                  |██████████        |                 |
Phase 5  |                  |        ██████    |                 |
Phase 6  |                  |              ████|████████████████ |
```

---

## Configuration

All hyper-parameters live in [`config/config.yaml`](config/config.yaml) and are loaded via
`nepse_nlp.utils.load_config()`. Highlights:

```yaml
jev:
  backend: simulated        # or "http" (requires JEV_API_KEY env)
  labels_3: [negative, neutral, positive]
hybrid:
  default_threshold: 0.70
  conformal_alpha: 0.10      # distribution-free guarantee level
  cost_fast_usd: 0.0002
  cost_slow_usd: 0.0050
prediction:
  target: direction
  models: [xgboost, logreg, lstm]
backtest:
  initial_capital: 1000000   # NPR
  brokerage_pct: 0.0040       # ~NEPSE
  allow_short: false          # NEPSE has no shorting
```

---

## Running the pipeline

```bash
make pipeline        # phase 1 -> phase 5
# or individually:
make run-phase1
make run-phase2
make run-phase3
make run-phase4
make run-phase5
```

Each phase script writes artifacts under `results/` and prints a Markdown summary table to stdout.

---

## API & dashboard

```bash
# FastAPI (port 8000)
make api
# endpoints:
#   GET  /health
#   POST /sentiment   { "text": "नेपाल बैंकको नाफा २०% ले बढ्यो" }
#   POST /predict     { "date": "2024-06-10" }
#   GET  /metrics

# Streamlit dashboard (port 8501)
make dashboard
```

---

## Docker

```bash
make docker-build                       # build
docker compose up -d                    # api + dashboard
docker compose logs -f api
open http://localhost:8000/health
open http://localhost:8501
```

---

## CI/CD pipeline

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=githubactions)](.github/workflows/ci.yml)

`.github/workflows/ci.yml` runs on every push/PR:

1. **lint** — `ruff check` + `black --check` + structural YAML validation.
2. **test** — pytest matrix on **Python 3.10 / 3.11 / 3.12** across **Ubuntu + macOS**, with `pip install -e ".[dev]"`.
3. **pipeline-smoke** — runs `phase1 → phase5` scripts on a clean runner and asserts all expected artifacts exist (this is the reproducibility gate).
4. **docker-build** — builds the Dockerfile (no push on PRs; pushes to GHCR on `v*` tags via `docker.yml`).

Plus `codeql.yml` (weekly security scan) and `docker.yml` (tag-triggered GHCR push).

To run the same checks locally before pushing:

```bash
make lint test pipeline
```

---

## Evaluation metrics

| Domain | Metric | Where |
|--------|--------|-------|
| Sentiment | Accuracy, Macro/Micro F1, MCC | `jev_benchmark.benchmark` |
| Calibration | Brier score, ECE, reliability diagram, ordinal bias | `calibration` |
| Hybrid | Cost / 1k preds, latency (ms), escalation rate, Pareto frontier | `hybrid_pipeline.evaluate` / `pareto_frontier` |
| Stock | RMSE, MAE, directional accuracy, hit rate | `prediction.walk_forward` |
| Trading | Sharpe, max drawdown, cumulative return, win rate | `backtest.Backtester` |
| Reproducibility | seed-fixed, deterministic simulated data, CI smoke gate | `utils.set_seed`, CI |

---

## Reproducibility & ethics

- **Seeds:** every stochastic component (simulated data, model init, splits) honors `utils.set_seed(42)`.
- **No look-ahead:** walk-forward training window never exceeds `t-1`; verified by `tests/test_features.py::test_build_feature_matrix_no_lookahead`.
- **No market manipulation:** outputs are research signals, not investment advice. Trading costs modeled realistically; NEPSE has no shorting (enforced).
- **Data ethics:** scrapers respect `robots.txt`; only public news + public NEPSE data are used; no PII stored. Simulated fallback avoids hammering live sites during development.
- **Limitations:** simulated Jev backend approximates a calibrated System-One decision model — swap in the real API for production benchmarks.

---

## Citation

```bibtex
@misc{nepse_nlp_2025,
  title  = {Sentiment Analysis of Nepali News for NEPSE Stock Prediction
            Using Jev and Confidence-Gated Hybrid Architectures},
  author = {Nepse NLP Research},
  year   = {2025},
  url    = {https://github.com/YOUR-USERNAME/nepali-sentiment-nepse},
}
```

License: **MIT** — see [`LICENSE`](LICENSE).
