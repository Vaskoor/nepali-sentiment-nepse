# Architecture

This document describes the system design of `nepali-sentiment-nepse`.

## 1. Data flow

```
                 ┌─────────────── Phase 1 ───────────────┐
  Nepali news ──► scraper ──► preprocess_pipeline ──► news_sentiment.csv
  (MeroLagani/                                  (Devanagari normalize,
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
  signals ─► backtest.Backtester.simulate (costs: 0.4%+SEBON+DP+CGT)
                                             ─► equity_curve.csv
                                             ─► phase5_backtest.json
  FastAPI  /sentiment /predict /metrics     Streamlit  dashboard
```

## 2. Package boundaries

- `data_preprocessing` — pure stdlib, no optional deps.
- `scraper` — optional `aiohttp`/`bs4`/`pandas` guarded; always degrades to simulated data.
- `jev_benchmark` / `baselines` — numpy; `transformers`/`torch` optional & guarded.
- `calibration` / `conformal` — numpy only.
- `hybrid_pipeline` — composes `jev_benchmark` + `baselines` + `conformal`.
- `features` / `prediction` / `backtest` — numpy/pandas/sklearn/xgboost; `torch` optional.
- `api` / `dashboard` — `fastapi`/`streamlit` guarded; lazy-import the pipeline.

## 3. Reproducibility contract

1. `utils.set_seed(42)` seeds Python/NumPy/PyTorch/TF.
2. All simulated data (headlines + NEPSE series) is seeded-deterministic.
3. Walk-forward training window never exceeds $t-1$ (no look-ahead); verified by `tests/test_features.py`.
4. CI `pipeline-smoke` job re-runs Phase 1→5 on a clean runner and asserts every artifact exists.

## 4. Cost model (Phase 3)

`cost_{1k} = α · c_fast + (1 − ρ(τ)) · c_slow`, where `ρ(τ)` is the escalation rate at threshold `τ`. Defaults: `c_fast=$0.0002/1k`, `c_slow=$0.005/1k`. The Pareto frontier sweeps `τ ∈ [0.30, 0.95]` and filters dominated (cost, accuracy) pairs.

## 5. NEPSE transaction costs (Phase 5)

- Brokerage: 0.40% (round-trip ~0.80%)
- SEBON fee: 0.015%
- DP fee: NPR 25 per trade
- Capital-gains tax: 5% on profit
- Shorting: **disallowed** (signals ∈ {+1, 0})
