"""Phase 4 CLI: NEPSE prediction model comparison + ablation.

Run with::

    python scripts/phase4_predict.py

This script:

1. Loads a **simulated** NEPSE OHLCV series and a **simulated**
   news-sentiment frame (correlated with returns so the signal is
   informative).
2. Builds the merged feature matrix via
   :func:`nepse_nlp.features.build_feature_matrix`.
3. Runs :func:`nepse_nlp.prediction.compare_models` across
   ``xgboost``, ``logreg``, and (if torch is installed) ``lstm``.
4. Runs the sentiment-vs-baseline ablation
   (:func:`nepse_nlp.prediction.ablation_sentiment_vs_baseline`).
5. Prints a markdown summary table and writes
   ``results/phase4_models.json`` with the full metrics.

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

# Lazy import of utils (lead agent 1-d provides it).
try:
    from nepse_nlp.utils import PROCESSED_DIR, get_logger, set_seed

    _LOGGER = get_logger("phase4_predict")
except Exception:  # pragma: no cover - degrade gracefully
    import logging

    _LOGGER = logging.getLogger("phase4_predict")
    logging.basicConfig(level=logging.INFO)

    def set_seed(seed: int = 42) -> None:  # type: ignore[no-redef]
        import random

        random.seed(seed)
        np.random.seed(seed)

    PROCESSED_DIR = None  # type: ignore[assignment]

from nepse_nlp.features import build_feature_matrix  # noqa: E402
from nepse_nlp.prediction import (  # noqa: E402
    ablation_sentiment_vs_baseline,
    compare_models,
)


# ---------------------------------------------------------------------------
# Data simulation
# ---------------------------------------------------------------------------
def _simulate_nepse_ohlcv(n_days: int = 500, seed: int = 42) -> pd.DataFrame:
    """Simulate a realistic NEPSE OHLCV series.

    Uses :func:`nepse_nlp.scraper.simulate_nepse_series` if available,
    otherwise falls back to a local random walk.

    The scraper helper returns either a flat close-price array or a
    full OHLCV DataFrame; we handle both shapes.
    """
    try:
        from nepse_nlp.scraper import simulate_nepse_series  # lazy

        sim = simulate_nepse_series(n_days=n_days, seed=seed)
    except Exception:
        _LOGGER.warning("simulate_nepse_series unavailable; using local random walk.")
        sim = None

    if isinstance(sim, pd.DataFrame) and "close" in sim.columns:
        # Already a full OHLCV frame; use as-is and ensure 'date' is a
        # proper datetime column.
        df = sim.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        else:
            df["date"] = pd.date_range(
                end=pd.Timestamp.today().normalize(),
                periods=len(df),
                freq="B",
            )
        return df

    # Flat array path (legacy or fallback).
    close_arr = np.asarray(sim, dtype=float) if sim is not None else None
    if close_arr is None:
        rng = np.random.default_rng(seed)
        rets = rng.normal(0.0003, 0.012, size=n_days)
        close_arr = 2000.0 * np.exp(np.cumsum(rets))
    rng = np.random.default_rng(seed + 1)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="B")
    df = pd.DataFrame(
        {
            "date": dates,
            "open": close_arr * (1 - rng.uniform(0, 0.01, size=n_days)),
            "high": close_arr * (1 + rng.uniform(0, 0.02, size=n_days)),
            "low": close_arr * (1 - rng.uniform(0, 0.02, size=n_days)),
            "close": close_arr,
            "volume": rng.uniform(1e5, 1e7, size=n_days),
        }
    )
    return df


def _simulate_news_sentiment(
    df_price: pd.DataFrame,
    seed: int = 7,
) -> pd.DataFrame:
    """Build a news-sentiment frame **correlated** with daily returns.

    The realised daily return at time ``t`` shifts the *next* day's
    news distribution, so the sentiment signal is informative but
    leak-free (the news we observe on day ``t`` is generated from day
    ``t-1``'s return).
    """
    rng = np.random.default_rng(seed)
    rets = df_price["close"].pct_change().fillna(0.0).values
    dates = df_price["date"].values
    len(dates)

    # Each day generates 1..4 news articles.
    records = []
    for i, d in enumerate(dates):
        # Sentiment is biased by yesterday's realised return (no leakage).
        prev_ret = rets[i - 1] if i > 0 else 0.0
        p_pos = 0.33 + 0.4 * np.tanh(prev_ret * 100.0)
        p_neg = 0.33 - 0.3 * np.tanh(prev_ret * 100.0)
        p_neu = 1.0 - p_pos - p_neg
        p_pos = max(0.05, p_pos)
        p_neg = max(0.05, p_neg)
        p_neu = max(0.05, p_neu)
        s = p_pos + p_neg + p_neu
        p_pos, p_neg, p_neu = p_pos / s, p_neg / s, p_neu / s

        n_articles = int(rng.integers(1, 5))
        labels = rng.choice(
            ["positive", "negative", "neutral"], size=n_articles, p=[p_pos, p_neg, p_neu]
        )
        conf = rng.uniform(0.5, 0.99, size=n_articles)
        for j in range(n_articles):
            records.append(
                {
                    "date": pd.Timestamp(d),
                    "label": labels[j],
                    "confidence": float(conf[j]),
                    "text": "simulated news headline",
                }
            )
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Markdown table helpers
# ---------------------------------------------------------------------------
def _metrics_md_table(model_metrics: Dict[str, Dict[str, float]]) -> str:
    cols = ["model", "rmse", "mae", "directional_accuracy", "hit_rate"]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = [header, sep]
    for m, r in model_metrics.items():
        rows.append(
            "| "
            + " | ".join(
                [
                    str(m),
                    f"{r.get('rmse', float('nan')):.6f}",
                    f"{r.get('mae', float('nan')):.6f}",
                    f"{r.get('directional_accuracy', 0.0):.4f}",
                    f"{r.get('hit_rate', 0.0):.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _ablation_md_table(ablation: Dict[str, Any]) -> str:
    w = ablation["with_sentiment"]
    b = ablation["baseline"]
    delta = ablation["delta_directional_accuracy"]
    return (
        "| variant | directional_accuracy | rmse |\n"
        "| --- | --- | --- |\n"
        f"| with sentiment | {w.get('directional_accuracy', 0.0):.4f} | "
        f"{w.get('rmse', float('nan')):.6f} |\n"
        f"| baseline (technical only) | "
        f"{b.get('directional_accuracy', 0.0):.4f} | "
        f"{b.get('rmse', float('nan')):.6f} |\n"
        f"| delta (sentiment - baseline) | {delta:+.4f} | -- |\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    set_seed(42)
    _LOGGER.info("Phase 4: NEPSE prediction model comparison")

    # 1. Build data.
    df_price = _simulate_nepse_ohlcv(n_days=500, seed=42)
    df_news = _simulate_news_sentiment(df_price, seed=7)
    _LOGGER.info("Loaded %d price rows and %d news records", len(df_price), len(df_news))

    # 2. Feature matrix.
    feats = build_feature_matrix(df_price, df_news)
    _LOGGER.info("Feature matrix shape: %s (after warm-up dropna)", feats.shape)

    # 3. Model comparison.
    _LOGGER.info("Running compare_models (direction target)...")
    cmp = compare_models(feats, target="direction", min_train=120, step=2)

    # 4. Ablation.
    _LOGGER.info("Running sentiment-vs-baseline ablation...")
    abl = ablation_sentiment_vs_baseline(
        feats,
        target="direction",
        min_train=120,
        step=2,
        model_type="xgboost",
    )

    # 5. Print markdown + persist JSON.
    print("\n## Phase 4 — NEPSE direction-prediction model comparison\n")
    print(_metrics_md_table(cmp))
    print("\n## Sentiment-vs-baseline ablation (xgboost)\n")
    print(_ablation_md_table(abl))

    out_dir = os.path.join(_REPO_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase4_models.json")

    # Custom JSON serialiser: skip the equity_curve Series.
    def _ser(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, (pd.Timestamp,)):
            return str(o)
        raise TypeError(f"unhandled type {type(o)}")

    payload = {
        "compare_models": cmp,
        "ablation": abl,
        "feature_matrix_shape": list(feats.shape),
        "n_features": len(feats.attrs.get("feature_cols", [])),
        "feature_cols": feats.attrs.get("feature_cols", []),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_ser)
    _LOGGER.info("Saved %s", out_path)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
