"""Phase 5 CLI: walk-forward backtest of the NEPSE strategy.

Run with::

    python scripts/phase5_backtest.py

This script:

1. Reuses the **simulated** NEPSE OHLCV + sentiment frames (correlated
   so the signal is informative).
2. Builds the merged feature matrix via
   :func:`nepse_nlp.features.build_feature_matrix`.
3. Runs :func:`nepse_nlp.backtest.backtest_strategy` with an
   XGBoost-based NEPSEPredictor.
4. Prints the equity metrics (final equity, total return, Sharpe,
   max drawdown, win rate, n_trades).
5. Saves ``results/phase5_backtest.json`` (full metrics) and
   ``results/equity_curve.csv`` (the daily equity curve).

This is an **offline** script: no network access is required.

License: MIT (see LICENSE at repo root).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd

# --- sys.path bootstrap so nepse_nlp resolves when run from repo root ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    from nepse_nlp.utils import set_seed, get_logger
    _LOGGER = get_logger("phase5_backtest")
except Exception:  # pragma: no cover - degrade gracefully
    import logging
    _LOGGER = logging.getLogger("phase5_backtest")
    logging.basicConfig(level=logging.INFO)

    def set_seed(seed: int = 42) -> None:  # type: ignore[no-redef]
        import random
        random.seed(seed)
        np.random.seed(seed)

from nepse_nlp.features import build_feature_matrix  # noqa: E402
from nepse_nlp.backtest import backtest_strategy, NEPSE_COSTS  # noqa: E402

# Reuse the Phase 4 data simulators (lazy import; copy if missing).
try:
    from scripts.phase4_predict import (  # type: ignore  # noqa: E402
        _simulate_nepse_ohlcv,
        _simulate_news_sentiment,
    )
    _PHASE4_AVAILABLE = True
except Exception:
    _PHASE4_AVAILABLE = False


def _local_simulate_nepse_ohlcv(n_days: int = 500, seed: int = 42) -> pd.DataFrame:
    """Fallback NEPSE OHLCV simulator (mirrors Phase 4).

    The scraper helper returns either a flat close-price array or a
    full OHLCV DataFrame; we handle both shapes.
    """
    try:
        from nepse_nlp.scraper import simulate_nepse_series  # lazy
        sim = simulate_nepse_series(n_days=n_days, seed=seed)
    except Exception:
        sim = None

    if isinstance(sim, pd.DataFrame) and "close" in sim.columns:
        df = sim.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        else:
            df["date"] = pd.date_range(
                end=pd.Timestamp.today().normalize(), periods=len(df), freq="B",
            )
        return df

    close_arr = np.asarray(sim, dtype=float) if sim is not None else None
    if close_arr is None:
        rng = np.random.default_rng(seed)
        rets = rng.normal(0.0003, 0.012, size=n_days)
        close_arr = 2000.0 * np.exp(np.cumsum(rets))
    rng = np.random.default_rng(seed + 1)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(),
                          periods=n_days, freq="B")
    return pd.DataFrame({
        "date": dates,
        "open": close_arr * (1 - rng.uniform(0, 0.01, size=n_days)),
        "high": close_arr * (1 + rng.uniform(0, 0.02, size=n_days)),
        "low": close_arr * (1 - rng.uniform(0, 0.02, size=n_days)),
        "close": close_arr,
        "volume": rng.uniform(1e5, 1e7, size=n_days),
    })


def _local_simulate_news_sentiment(df_price: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Fallback news-sentiment simulator (mirrors Phase 4)."""
    rng = np.random.default_rng(seed)
    rets = df_price["close"].pct_change().fillna(0.0).values
    dates = df_price["date"].values
    records = []
    for i, d in enumerate(dates):
        prev_ret = rets[i - 1] if i > 0 else 0.0
        p_pos = 0.33 + 0.4 * np.tanh(prev_ret * 100.0)
        p_neg = 0.33 - 0.3 * np.tanh(prev_ret * 100.0)
        p_neu = 1.0 - p_pos - p_neg
        p_pos = max(0.05, p_pos); p_neg = max(0.05, p_neg)
        p_neu = max(0.05, p_neu)
        s = p_pos + p_neg + p_neu
        p_pos, p_neg, p_neu = p_pos / s, p_neg / s, p_neu / s
        n_articles = int(rng.integers(1, 5))
        labels = rng.choice(["positive", "negative", "neutral"],
                             size=n_articles, p=[p_pos, p_neg, p_neu])
        conf = rng.uniform(0.5, 0.99, size=n_articles)
        for j in range(n_articles):
            records.append({
                "date": pd.Timestamp(d),
                "label": labels[j],
                "confidence": float(conf[j]),
                "text": "simulated news headline",
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _equity_metrics_md(tm: Dict[str, float]) -> str:
    return (
        "| metric | value |\n"
        "| --- | --- |\n"
        f"| final equity | {tm.get('final_equity', 0.0):,.2f} |\n"
        f"| total return (%) | {tm.get('total_return_pct', 0.0):.4f} |\n"
        f"| Sharpe | {tm.get('sharpe', 0.0):.4f} |\n"
        f"| max drawdown | {tm.get('max_drawdown', 0.0):.4f} |\n"
        f"| win rate | {tm.get('win_rate', 0.0):.4f} |\n"
        f"| n trades | {tm.get('n_trades', 0)} |\n"
    )


def _ser(o):  # pragma: no cover - tiny helper
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, pd.Timestamp):
        return str(o)
    if isinstance(o, pd.Series):
        return o.to_dict()
    raise TypeError(f"unhandled type {type(o)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    set_seed(42)
    _LOGGER.info("Phase 5: walk-forward backtest")

    # 1. Data.
    if _PHASE4_AVAILABLE:
        df_price = _simulate_nepse_ohlcv(n_days=500, seed=42)
        df_news = _simulate_news_sentiment(df_price, seed=7)
    else:
        df_price = _local_simulate_nepse_ohlcv(n_days=500, seed=42)
        df_news = _local_simulate_news_sentiment(df_price, seed=7)
    _LOGGER.info("Loaded %d price rows and %d news records",
                 len(df_price), len(df_news))

    # 2. Features.
    feats = build_feature_matrix(df_price, df_news)
    _LOGGER.info("Feature matrix shape: %s", feats.shape)

    # 3. Backtest.
    _LOGGER.info("Running backtest_strategy (xgboost, direction)...")
    result = backtest_strategy(
        df=feats,
        model_type="xgboost",
        target="direction",
        min_train=120,
        step=1,
        initial_capital=1_000_000.0,
        costs=NEPSE_COSTS,
    )

    # 4. Print markdown summary.
    pm = result["prediction_metrics"]
    tm = result["trading_metrics"]
    print("\n## Phase 5 — Walk-forward backtest summary\n")
    print("### Prediction metrics\n")
    print(
        "| metric | value |\n"
        "| --- | --- |\n"
        f"| directional accuracy | {pm.get('directional_accuracy', 0.0):.4f} |\n"
        f"| hit rate | {pm.get('hit_rate', 0.0):.4f} |\n"
        f"| rmse | {pm.get('rmse', float('nan')):.6f} |\n"
        f"| mae | {pm.get('mae', float('nan')):.6f} |\n"
    )
    print("### Trading metrics\n")
    print(_equity_metrics_md(tm))

    # 5. Persist artifacts.
    out_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "phase5_backtest.json")
    payload = {
        "prediction_metrics": pm,
        "trading_metrics": tm,
        "n_trades_completed": tm.get("n_trades", 0),
        "costs": NEPSE_COSTS,
        "feature_matrix_shape": list(feats.shape),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_ser)
    _LOGGER.info("Saved %s", out_json)

    out_csv = os.path.join(out_dir, "equity_curve.csv")
    eq = result.get("equity_curve")
    if eq is not None and len(eq) > 0:
        # Index = dates, column = equity.
        df_eq = pd.DataFrame({"equity": eq.values}, index=eq.index)
        df_eq.index.name = "date"
        df_eq.to_csv(out_csv)
        _LOGGER.info("Saved %s", out_csv)
    else:
        _LOGGER.warning("Empty equity curve; skipping CSV write.")

    print(f"\nWrote {out_json}")
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
