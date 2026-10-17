# Nepali Sentiment · NEPSE

> **Sentiment Analysis of Nepali News for NEPSE Stock Prediction Using Jev
> (System One Decision Model) and Confidence-Gated Hybrid Architectures.**

<p align="center">
  <a href="https://github.com/Vaskoor/nepali-sentiment-nepse/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Vaskoor/nepali-sentiment-nepse/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/Vaskoor/nepali-sentiment-nepse/actions/workflows/codeql.yml"><img alt="CodeQL" src="https://github.com/Vaskoor/nepali-sentiment-nepse/actions/workflows/codeql.yml/badge.svg"></a>
  <a href="https://github.com/Vaskoor/nepali-sentiment-nepse/actions/workflows/docker.yml"><img alt="Docker" src="https://github.com/Vaskoor/nepali-sentiment-nepse/actions/workflows/docker.yml/badge.svg"></a>
  <br/>
  <img alt="License" src="https://img.shields.io/badge/license-MIT-emerald.svg">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-teal.svg">
  <img alt="Tests" src="https://img.shields.io/badge/tests-26%2F26%20passing-emerald.svg">
  <img alt="Code style: ruff + black" src="https://img.shields.io/badge/code%20style-ruff%20%2B%20black-slate.svg">
  <img alt="Typed: PEP 561" src="https://img.shields.io/badge/typed-PEP%20561-emerald.svg">
  <img alt="Status: research" src="https://img.shields.io/badge/status-research-amber.svg">
  <img alt="Version" src="https://img.shields.io/badge/version-0.2.0-slate.svg">
</p>

---

## Abstract

Nepali is a **low-resource** language for financial NLP, yet the Nepal Stock
Exchange (NEPSE) is an emerging market where timely sentiment signals could
materially improve short-horizon index prediction. This project delivers the
first end-to-end study of **Jev**, a fast, calibrated *System-One* decision
model, on Nepali financial news, and proposes a **confidence-gated hybrid
architecture** that escalates uncertain Jev decisions to a heavier
multilingual model (XLM-R / NepaliBERT) below a **conformally-calibrated**
threshold. Aggregated daily sentiment feeds XGBoost / LogReg / LSTM
next-day NEPSE predictors validated under strict **expanding-window
walk-forward** validation, with a sentiment-ablation study and a realistic
NEPSE-cost backtest. All results ship with a deterministic **simulated Jev
backend** that preserves the live API contract, so the pipeline is fully
reproducible offline.

> **Not investment advice.** This is a research codebase. Backtest results on
> simulated data are illustrative of methodology, not of market performance.

---

## Table of contents

- [Highlights](#highlights)
- [Key results](#key-results)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
- [The six-phase research plan](#the-six-phase-research-plan)
- [Configuration](#configuration)
- [API & dashboard](#api--dashboard)
- [Docker](#docker)
- [CI/CD](#cicd)
- [Evaluation metrics](#evaluation-metrics)
- [Reproducibility & ethics](#reproducibility--ethics)
- [Substitution note (Jev API)](#substitution-note-jev-api)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

---

## Highlights

- **Devanagari-aware preprocessing** — Unicode NFC normalization, zero-width
  / BIDI stripping, a regex tokenizer that splits the danda `।` (U+0964) as a
  standalone token, code-mix ratio, and a 40-word Nepali stopword set.
- **Jev zero-shot benchmark** with full calibration — accuracy, macro/micro
  F1, MCC, **Brier score**, **ECE**, reliability diagrams, ordinal-scale
  bias, stratified by code-mix level and sector.
- **Confidence-gated hybrid pipeline** — split-conformal thresholds
  (distribution-free coverage), cost-accuracy **Pareto frontier**, with
  escalation to a heavier model only below the calibrated gate.
- **Sentiment-augmented NEPSE prediction** — walk-forward (expanding window,
  no look-ahead, unit-tested), with a sentiment **ablation** that quantifies
  the marginal value of the sentiment signal.
- **Realistic backtesting** — NEPSE transaction costs (brokerage 0.40% +
  SEBON 0.015% + DP fee NPR 25 + 5% capital-gains), no shorting (enforced),
  Sharpe / max-drawdown / win-rate.
- **Reproducible by construction** — seed-fixed (`seed=42`), deterministic
  simulated data, CI `pipeline-smoke` job re-runs Phase 1→5 on a clean
  runner and asserts every artifact.
- **Ships as a typed package** (`py.typed`, PEP 561) with a FastAPI service,
  Streamlit dashboard, multi-stage Dockerfile, IEEE LaTeX paper draft, and
  GitHub Actions CI/CD.

## Key results

> All numbers are from the deterministic simulated backend (`seed=42`,
> offline-runnable via `make pipeline`). Swap to the live Jev API for
> production benchmarks.

| Metric | Jev (sim) | Hybrid (τ=0.70) | XLM-R (sim) | GPT-4o (sim) |
|---|---|---|---|---|
| Accuracy | 0.85 | **0.86** | 0.81 | 0.87 |
| Macro-F1 | 0.83 | **0.85** | 0.79 | 0.86 |
| Brier ↓ | 0.21 | **0.18** | 0.20 | 0.16 |
| ECE ↓ | 0.40 | **0.16** | 0.19 | 0.12 |
| Cost / 1k preds | **$0.20** | $1.13 | $5.00 | $22.00 |

| NEPSE prediction (walk-forward directional accuracy) | technical-only | + sentiment | Δ |
|---|---|---|---|
| LogReg | 0.443 | 0.476 | +3.3 pp |
| XGBoost | 0.508 | **0.541** | +3.3 pp |
| LSTM* | 0.502 | 0.533 | +3.1 pp |

\* LSTM requires `torch`; falls back to LogReg if unavailable.

## Architecture

```
                 ┌─────────────── Phase 1 ───────────────┐
  Nepali news ──► scraper ──► preprocess_pipeline ──► news_sentiment.csv
  (MeroLagani/                                  (Devanagari NFC,
   ShareSansar/                                  clean, tokenize,
   NepaliPaisa/                                   code-mix detect)
   BizMandu)        ▲ simulated fallback if offline
                   │
  NEPSE OHLCV ─────► scraper.fetch_nepse_ohlcv ──► nepse_daily.csv
                   ▲ simulated GARCH-ish series if offline

                 ┌─────────────── Phase 2 ───────────────┐
  news_sentiment ─► JevClassifier ──► {label, conf, probs} ──► benchmark()
                  ► baselines (NepaliBERT/MuRIL/XLM-R/GPT-4o-sim)
                  ► calibration (Brier, ECE, reliability)  ──► phase2_benchmark.json

                 ┌─────────────── Phase 3 ───────────────┐
  Jev ─► if conf < τ ─► escalate (XLM-R) ─► hybrid_pipeline.evaluate
                                              + pareto_frontier
        τ from conformal.calibrate_conformal(val, α=0.10) ──► phase3_pareto.json

                 ┌─────────────── Phase 4 ───────────────┐
  nepse_daily + sent ─► features.build_feature_matrix ─► prediction.walk_forward
                       (RSI, MACD, MA, sent_*)             (XGBoost/LogReg/LSTM)
                       │                                  ─► phase4_models.json
                       └── sentiment ablation (drop sent_*)

                 ┌─────────────── Phase 5 ───────────────┐
  signals ─► backtest.Backtester.simulate (0.4% brokerage + SEBON + DP + CGT)
                                             ─► equity_curve.csv
                                             ─► phase5_backtest.json
  FastAPI  /sentiment /predict /metrics     Streamlit  dashboard
```

## Repository layout

```
nepali-sentiment-nepse/
├── .github/
│   ├── workflows/        # ci.yml · docker.yml · codeql.yml
│   ├── ISSUE_TEMPLATE/   # bug_report · feature_request · config
│   ├── pull_request_template.md
│   ├── dependabot.yml
│   └── FUNDING.yml
├── api/main.py           # FastAPI service (Phase 5)
├── config/config.yaml    # single source of truth for hyper-parameters
├── dashboard/app.py      # Streamlit dashboard (Phase 5)
├── data/{raw,processed,external}/   # data (git-ignored except .gitkeep)
├── docs/architecture.md  # system design
├── models/               # trained artifacts (git-ignored)
├── mlruns/               # MLflow runs (git-ignored)
├── paper/main.tex       # IEEE LaTeX paper draft (Phase 6)
├── results/             # pipeline outputs (git-ignored except .gitkeep)
├── scripts/             # phase1..phase5 entry points
├── src/nepse_nlp/        # the typed Python package (12 modules + py.typed)
│   ├── data_preprocessing.py · scraper.py · jev_benchmark.py
│   ├── baselines.py · calibration.py · conformal.py · hybrid_pipeline.py
│   ├── features.py · prediction.py · backtest.py · utils.py · __init__.py
│   └── py.typed          # PEP 561 marker
├── tests/                # pytest unit tests (26, all passing)
├── CITATION.cff          # GitHub "Cite this repository"
├── CHANGELOG.md · CONTRIBUTING.md · CODE_OF_CONDUCT.md · SECURITY.md
├── Dockerfile · docker-compose.yml · .pre-commit-config.yaml
├── Makefile · pyproject.toml · requirements.txt · LICENSE · README.md
```

## Quick start

```bash
# 1. clone
git clone https://github.com/Vaskoor/nepali-sentiment-nepse.git
cd nepali-sentiment-nepse

# 2. env + install (core + dev tooling)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 3. run the full pipeline end-to-end (offline, deterministic, ~30 s)
make pipeline        # phase1 -> phase5

# 4. inspect outputs
ls results/          # phase2_benchmark.json, phase3_pareto.json,
                     # phase4_models.json, phase5_backtest.json, equity_curve.csv

# 5. tests + lint
make test lint
```

> The pipeline is **fully offline**: when network scraping is unavailable,
> `scraper.py` falls back to deterministic simulated Nepali financial
> headlines and a NEPSE-like OHLCV series (GARCH-ish volatility). All
> results are seed-stable (`seed=42`).

## The six-phase research plan

| Phase | Module(s) | Script | Key output |
|-------|-----------|--------|------------|
| 1 — Data & preprocessing | `data_preprocessing`, `scraper` | `phase1_preprocess.py` | `news_sentiment.csv`, `nepse_daily.csv` |
| 2 — Jev zero-shot benchmark | `jev_benchmark`, `baselines`, `calibration` | `phase2_benchmark.py` | acc, F1, Brier, ECE per model |
| 3 — Confidence-gated hybrid | `hybrid_pipeline`, `conformal` | `phase3_hybrid.py` | Pareto frontier + conformal τ |
| 4 — Sentiment-augmented prediction | `features`, `prediction` | `phase4_predict.py` | walk-forward metrics + ablation |
| 5 — Backtesting & deployment | `backtest`, `api/main`, `dashboard/app` | `phase5_backtest.py` | equity curve + trading metrics |
| 6 — Paper & reproducibility | `paper/main.tex`, CI, Docker | — | LaTeX paper + containerized + CI-pipelined |

### 18-month Gantt

```
2025 Q1 Q2 Q3 Q4 | 2026 Q1 Q2 Q3 Q4
Phase 1 |████████        |                 |
Phase 2 |     ████████   |                 |
Phase 3 |         ████|████               |
Phase 4 |              |██████████        |
Phase 5 |              |     ██████       |
Phase 6 |              |          ████|████████████ |
```

## Configuration

All hyper-parameters live in [`config/config.yaml`](config/config.yaml) and are
loaded via `nepse_nlp.utils.load_config()`. Highlights:

```yaml
jev:
  backend: simulated          # or "http" (requires JEV_API_KEY env)
  labels_3: [negative, neutral, positive]
hybrid:
  default_threshold: 0.70
  conformal_alpha: 0.10        # distribution-free guarantee level
  cost_fast_usd: 0.0002
  cost_slow_usd: 0.0050
prediction:
  target: direction
  models: [xgboost, logreg, lstm]
backtest:
  initial_capital: 1000000     # NPR
  brokerage_pct: 0.0040         # ~NEPSE
  allow_short: false            # NEPSE has no shorting
```

## API & dashboard

```bash
make api         # FastAPI on :8000
#   GET  /health
#   POST /sentiment   {"text": "नेपाल बैंकको नाफा २०% ले बढ्यो"}
#   POST /predict     {"date": "2024-06-10"}
#   GET  /metrics

make dashboard   # Streamlit on :8501
```

## Docker

```bash
make docker-build               # build
docker compose up -d            # api + dashboard
open http://localhost:8000/health
open http://localhost:8501
```

Multi-stage Dockerfile (`python:3.11-slim` builder → slim runtime), with a
healthcheck on `/health` and volume mounts for `data/`, `results/`, `models/`,
`mlruns/`. On `v*` tags, `docker.yml` pushes the image to `ghcr.io`.

## CI/CD

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push/PR:

1. **lint** — `ruff check` + `black --check` + structural YAML validation.
2. **test** — pytest matrix on **Python 3.10 / 3.11 / 3.12** × **Ubuntu + macOS**
   (macOS installs `libomp` for xgboost).
3. **pipeline-smoke** — runs `phase1 → phase5` on a clean runner and asserts
   every expected artifact exists (the reproducibility gate).
4. **docker-build** — builds the Dockerfile (cache-enabled).

Plus `codeql.yml` (weekly security) and `docker.yml` (tag-triggered GHCR push).
`dependabot.yml` keeps deps current; `pre-commit` hooks enforce style on
commit.

## Evaluation metrics

| Domain | Metric | Where |
|--------|--------|-------|
| Sentiment | Accuracy, Macro/Micro F1, MCC | `jev_benchmark.benchmark` |
| Calibration | Brier, ECE, reliability, ordinal bias | `calibration` |
| Hybrid | Cost/1k preds, latency, escalation rate, Pareto frontier | `hybrid_pipeline` |
| Stock | RMSE, MAE, directional accuracy, hit rate | `prediction.walk_forward` |
| Trading | Sharpe, max drawdown, cumulative return, win rate | `backtest.Backtester` |
| Reproducibility | seed-fixed, deterministic data, CI smoke gate | `utils.set_seed`, CI |

## Reproducibility & ethics

- **Seeds:** every stochastic component honors `utils.set_seed(42)`.
- **No look-ahead:** walk-forward training never exceeds `t-1`; unit-tested.
- **No market manipulation:** research signals only, not investment advice.
- **Realistic costs:** brokerage 0.40% + SEBON + DP fee + capital-gains; no
  shorting (NEPSE rule, enforced).
- **Data ethics:** scrapers respect `robots.txt`; only public news + public
  NEPSE data; no PII stored; simulated fallback avoids hammering live sites.
- **Limitations:** the simulated Jev backend approximates a calibrated
  System-One decision model — swap to `backend="http"` + `JEV_API_KEY` for
  production benchmarks.

## Substitution note (Jev API)

Jev is a fast calibrated *System-One* decision model. Its public API is not
guaranteed to be available, so this repo ships a **deterministic,
lexicon-driven simulated backend** that implements the identical
`classify(text) -> {label, confidence, probs, source, backend, latency_ms}`
contract. To use the real endpoint:

```python
from nepse_nlp.jev_benchmark import JevClassifier
clf = JevClassifier(backend="http", api_key=os.environ["JEV_API_KEY"])
clf.classify("नेबिल बैंकको नाफा २० प्रतिशतले बढ्यो")
```

The substitution is documented in the module docstring of
`src/nepse_nlp/jev_benchmark.py` and in the LaTeX paper (§Limitations).

## Citation

If this contributes to your research, please cite via
[`CITATION.cff`](CITATION.cff) (GitHub renders a *"Cite this repository"*
button from it). BibTeX:

```bibtex
@software{kaphle_nepse_nlp_2026,
  title  = {Sentiment Analysis of Nepali News for NEPSE Stock Prediction
            Using Jev and Confidence-Gated Hybrid Architectures},
  author = {Kaphle, Bhaskar},
  year   = {2026},
  version = {0.2.0},
  url    = {https://github.com/Vaskoor/nepali-sentiment-nepse},
}
```

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the dev setup, the
reproducibility/honesty checklist, and the conventional-commit convention.
`pre-commit` hooks (`ruff`, `black`, `codespell`, `mypy`) enforce style on
commit. All interactions are governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

**MIT** — see [`LICENSE`](LICENSE).

## Database Architecture

This project uses a relational database (PostgreSQL-compatible). The full
schema DDL is in [`database/schema.sql`](database/schema.sql).

### Tables & Linkage

| Table | Purpose | Key FKs |
|-------|---------|---------|
| `users` | User accounts, auth, roles | root entity |
| (domain tables) | App-specific data | `user_id → users(id)` ON DELETE CASCADE |

### Table Linkage (ERD)

```
users (1) ──< (N) domain_tables
  │ id (PK)                      │ user_id (FK → users.id)
  │ email (UQ)                   │ created_at
  │ role
```

- **users** is the root; domain tables reference `users(id)` with `ON DELETE CASCADE`.
- **Indexes** on all FK columns + frequently-filtered columns (date, status, category).
- **Timestamps** (`created_at`) on all tables for auditability.
- See [`database/schema.sql`](database/schema.sql) for the complete DDL.

