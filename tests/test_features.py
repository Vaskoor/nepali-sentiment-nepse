"""Pytest unit tests for nepse_nlp.features + nepse_nlp.prediction.

Run with::

    pytest tests/test_features.py -v

These tests assume only the always-available scientific stack
(numpy, pandas, sklearn, xgboost). All optional deps (torch, fastapi,
streamlit, plotly) are bypassed.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Ensure the project src dir is importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from nepse_nlp.features import (  # noqa: E402
    aggregate_daily_sentiment,
    build_feature_matrix,
    macd,
    moving_averages,
    rsi,
    technical_indicators,
)
from nepse_nlp.prediction import (  # noqa: E402
    NEPSEPredictor,
    walk_forward,
    compare_models,
    ablation_sentiment_vs_baseline,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
def _make_ohlcv(n_days: int = 250, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic OHLCV frame."""
    rng = np.random.default_rng(seed)
    # Mean-reverting random walk for close.
    rets = rng.normal(0.0002, 0.012, size=n_days)
    close = 2000.0 * np.exp(np.cumsum(rets))
    dates = pd.date_range(end=pd.Timestamp.today().normalize(),
                          periods=n_days, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "open": close * (1 - rng.uniform(0, 0.01, size=n_days)),
        "high": close * (1 + rng.uniform(0, 0.02, size=n_days)),
        "low": close * (1 - rng.uniform(0, 0.02, size=n_days)),
        "close": close,
        "volume": rng.uniform(1e5, 1e7, size=n_days),
    })
    return df


def _make_news(n_articles: int = 500, seed: int = 7) -> pd.DataFrame:
    """Build synthetic news-level sentiment records."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(),
                          periods=n_articles, freq="D")
    # Mix positive/negative/neutral labels with some day-clustering.
    base_idx = rng.integers(0, 3, size=n_articles)
    label_map = {0: "positive", 1: "negative", 2: "neutral"}
    labels = [label_map[i] for i in base_idx]
    conf = rng.uniform(0.5, 0.99, size=n_articles)
    return pd.DataFrame({
        "date": dates,
        "label": labels,
        "confidence": conf,
        "text": ["sample headline"] * n_articles,
    })


# ---------------------------------------------------------------------------
# RSI tests
# ---------------------------------------------------------------------------
def test_rsi_bounds():
    """RSI must lie within [0, 100] wherever it is not NaN."""
    df = _make_ohlcv(n_days=200)
    rsi_series = rsi(df["close"], period=14).dropna()
    assert (rsi_series >= 0).all()
    assert (rsi_series <= 100).all()


def test_rsi_period_warmup():
    """First ``period`` values of RSI should be NaN (warm-up)."""
    df = _make_ohlcv(n_days=50)
    rsi_series = rsi(df["close"], period=14)
    assert rsi_series.iloc[:14].isna().all()
    assert not rsi_series.iloc[14:].isna().all()


# ---------------------------------------------------------------------------
# MACD tests
# ---------------------------------------------------------------------------
def test_macd_shapes():
    """MACD lines must be same length as input."""
    df = _make_ohlcv(n_days=120)
    macd_line, signal_line, hist = macd(df["close"])
    assert len(macd_line) == len(df)
    assert len(signal_line) == len(df)
    assert len(hist) == len(df)
    # Histogram equals macd - signal by construction.
    np.testing.assert_allclose(
        hist.values, (macd_line - signal_line).values, equal_nan=True
    )


# ---------------------------------------------------------------------------
# Moving averages tests
# ---------------------------------------------------------------------------
def test_moving_averages_columns():
    df = _make_ohlcv(n_days=100)
    ma = moving_averages(df["close"], windows=(5, 10, 20))
    assert list(ma.columns) == ["ma_5", "ma_10", "ma_20"]
    assert len(ma) == len(df)


# ---------------------------------------------------------------------------
# Sentiment aggregation tests
# ---------------------------------------------------------------------------
def test_aggregate_daily_sentiment_columns():
    """aggregate_daily_sentiment must produce the expected columns."""
    df_news = _make_news(n_articles=200)
    agg = aggregate_daily_sentiment(df_news)
    expected_cols = {
        "sent_mean_conf",
        "sent_median_conf",
        "sent_std_conf",
        "sent_pos_ratio",
        "sent_neg_ratio",
        "sent_n_articles",
        "sent_momentum",
        "sent_dispersion",
    }
    assert expected_cols.issubset(set(agg.columns))
    # pos_ratio + neg_ratio should be in [0, 1].
    assert (agg["sent_pos_ratio"] >= 0).all()
    assert (agg["sent_pos_ratio"] <= 1).all()
    assert (agg["sent_neg_ratio"] >= 0).all()
    assert (agg["sent_neg_ratio"] <= 1).all()
    # n_articles should be a positive integer count.
    assert (agg["sent_n_articles"] > 0).all()
    # Momentum first row is 0 by construction.
    assert agg["sent_momentum"].iloc[0] == 0.0


def test_aggregate_daily_sentiment_empty():
    """Empty input -> empty DataFrame with all expected columns."""
    empty = aggregate_daily_sentiment(pd.DataFrame(columns=["date", "label", "confidence"]))
    assert "sent_mean_conf" in empty.columns
    assert "sent_pos_ratio" in empty.columns
    assert "sent_momentum" in empty.columns
    assert len(empty) == 0


# ---------------------------------------------------------------------------
# Technical indicators tests
# ---------------------------------------------------------------------------
def test_technical_indicators_adds_columns():
    df = _make_ohlcv(n_days=100)
    out = technical_indicators(df)
    for col in ("rsi", "macd", "macd_signal", "macd_hist",
                "ma_5", "ma_10", "ma_20", "ma_50",
                "returns", "log_ret", "volatility", "volume_zscore"):
        assert col in out.columns, f"missing {col}"


# ---------------------------------------------------------------------------
# Feature matrix + no-lookahead tests
# ---------------------------------------------------------------------------
def test_build_feature_matrix_no_lookahead():
    """Target next_ret must be genuinely shifted (no leakage)."""
    df_price = _make_ohlcv(n_days=200)
    df_news = _make_news(n_articles=400)
    feats = build_feature_matrix(df_price, df_news)

    # Target column must exist.
    assert "next_ret" in feats.columns
    assert "next_dir" in feats.columns

    # All feature columns must be NaN-free (dropna removed warm-up rows).
    feature_cols = feats.attrs.get("feature_cols", [])
    assert len(feature_cols) > 0
    assert not feats[feature_cols].isna().any().any()
    assert not feats["next_ret"].isna().any()
    assert not feats["next_dir"].isna().any()

    # Look-ahead check: a row's own `returns` column must be the
    # *current* period's return and must NOT equal next_ret (which is
    # the next period's return). Specifically, next_ret at row i should
    # equal returns at row i+1.
    aligned = pd.DataFrame({
        "next_ret": feats["next_ret"].values,
        "next_period_ret": feats["returns"].shift(-1).values,
    }).dropna()
    # next_ret[i] == returns[i+1] for the same close. This proves the
    # shift is genuinely +1.
    if len(aligned) > 0:
        np.testing.assert_allclose(
            aligned["next_ret"].values,
            aligned["next_period_ret"].values,
            atol=1e-8,
        )


# ---------------------------------------------------------------------------
# Walk-forward tests
# ---------------------------------------------------------------------------
def test_walk_forward_directional_accuracy_range():
    """walk_forward: predictions length > 0 and accuracy in [0, 1]."""
    df_price = _make_ohlcv(n_days=180, seed=11)
    df_news = _make_news(n_articles=300, seed=5)
    feats = build_feature_matrix(df_price, df_news)
    res = walk_forward(
        df=feats, model_type="xgboost", target="direction",
        min_train=120, step=5,
    )
    assert "metrics" in res
    assert "predictions" in res
    assert "actuals" in res
    assert "dates" in res
    assert len(res["predictions"]) > 0
    assert len(res["predictions"]) == len(res["actuals"])
    acc = res["metrics"]["directional_accuracy"]
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= res["metrics"]["hit_rate"] <= 1.0


def test_walk_forward_logreg_runs():
    """walk_forward should also work with logistic regression."""
    df_price = _make_ohlcv(n_days=180, seed=11)
    df_news = _make_news(n_articles=300, seed=5)
    feats = build_feature_matrix(df_price, df_news)
    res = walk_forward(
        df=feats, model_type="logreg", target="direction",
        min_train=120, step=10,
    )
    assert 0.0 <= res["metrics"]["directional_accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Predictor class tests
# ---------------------------------------------------------------------------
def test_predictor_fit_predict_xgboost():
    df_price = _make_ohlcv(n_days=180, seed=11)
    df_news = _make_news(n_articles=300, seed=5)
    feats = build_feature_matrix(df_price, df_news)
    feat_cols = feats.attrs["feature_cols"]
    train = feats.iloc[:150]
    test = feats.iloc[150:152]
    model = NEPSEPredictor(model_type="xgboost", target="direction", seed=42)
    model.fit(train[feat_cols], train["next_dir"])
    preds = model.predict(test[feat_cols])
    assert len(preds) == len(test)
    fi = model.feature_importance()
    assert len(fi) == len(feat_cols)
    # Importance values should be non-negative.
    assert all(v >= 0 for v in fi.values())


def test_predictor_lstm_fallback_without_torch():
    """If torch is unavailable, LSTM must fall back to logreg silently."""
    df_price = _make_ohlcv(n_days=180, seed=11)
    df_news = _make_news(n_articles=300, seed=5)
    feats = build_feature_matrix(df_price, df_news)
    feat_cols = feats.attrs["feature_cols"]
    train = feats.iloc[:150]
    test = feats.iloc[150:152]
    model = NEPSEPredictor(model_type="lstm", target="direction", seed=42)
    # If torch absent, model_type should have fallen back.
    try:
        import torch  # noqa: F401
    except Exception:
        assert model.model_type == "logreg"
    model.fit(train[feat_cols], train["next_dir"])
    preds = model.predict(test[feat_cols])
    assert len(preds) == len(test)


# ---------------------------------------------------------------------------
# Ablation tests
# ---------------------------------------------------------------------------
def test_ablation_sentiment_vs_baseline_returns_delta():
    df_price = _make_ohlcv(n_days=180, seed=11)
    df_news = _make_news(n_articles=300, seed=5)
    feats = build_feature_matrix(df_price, df_news)
    res = ablation_sentiment_vs_baseline(
        feats, target="direction", min_train=120, step=10,
    )
    assert "with_sentiment" in res
    assert "baseline" in res
    assert "delta_directional_accuracy" in res
    assert -1.0 <= res["delta_directional_accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------
def test_compare_models_runs():
    df_price = _make_ohlcv(n_days=180, seed=11)
    df_news = _make_news(n_articles=300, seed=5)
    feats = build_feature_matrix(df_price, df_news)
    out = compare_models(feats, target="direction", min_train=120, step=10)
    assert "xgboost" in out
    assert "logreg" in out
    for m, r in out.items():
        assert "directional_accuracy" in r
        assert 0.0 <= r["directional_accuracy"] <= 1.0


if __name__ == "__main__":
    # Allow running this file directly as a script (poor man's runner).
    import pytest as _pytest

    sys.exit(_pytest.main([__file__, "-v"]))
