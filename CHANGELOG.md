# Changelog

All notable changes to this project are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Repository-scaffolding for academic use: `CITATION.cff`, `CONTRIBUTING.md`,
  `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CHANGELOG.md`.
- `py.typed` marker (PEP 561) — the package now advertises inline types.
- `pre-commit` hooks (ruff, black, codespell, standard checks).
- `.github/ISSUE_TEMPLATE/`, pull-request template, `dependabot.yml`, `FUNDING.yml`.
- Substantially enhanced `README.md` with badges, architecture diagram,
  results table, reproducibility section, and BibTeX citation.

### Changed
- Bumped to `0.2.0`; improved `pyproject.toml` metadata (URLs, classifiers, scripts).

## [0.1.0] — 2026-10-14

### Added — initial release
- **Phase 1** — Devanagari-aware preprocessing (NFC normalize, zero-width/
  BIDI strip, danda-aware tokenizer, code-mix detection) + async news/NEPSE
  scraper with deterministic simulated fallback.
- **Phase 2** — Jev (System One) zero-shot benchmark with Brier / ECE /
  reliability + code-mix & sector stratification; NepaliBERT / MuRIL / XLM-R /
  GPT-4o-simulated baselines.
- **Phase 3** — confidence-gated hybrid pipeline with split-conformal
  thresholds and cost-accuracy Pareto frontier.
- **Phase 4** — sentiment-augmented NEPSE prediction (XGBoost / LogReg /
  LSTM) with expanding-window walk-forward validation and sentiment ablation.
- **Phase 5** — walk-forward backtesting with realistic NEPSE transaction
  costs (brokerage 0.4% + SEBON + DP fee + capital-gains, no shorting),
  FastAPI service, Streamlit dashboard.
- **Phase 6** — IEEE LaTeX paper draft, Dockerfile, docker-compose,
  `Makefile`, `requirements.txt`, `pyproject.toml`, MIT license.
- GitHub Actions CI (`ci.yml` lint + matrix test + pipeline-smoke +
  docker-build), `docker.yml` (GHCR on tags), `codeql.yml` (weekly security).
- 26 pytest unit tests (all passing offline).

### Substitution note
The Jev API is not publicly available; a deterministic lexicon-driven
**simulated backend** implements the identical `classify() -> {label,
confidence, probs}` contract so the pipeline remains reproducible. Set
`backend="http"` with a valid `JEV_API_KEY` for the live endpoint.
