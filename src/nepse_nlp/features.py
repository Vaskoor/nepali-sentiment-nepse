"""Feature engineering for NEPSE prediction.

Phase 4 — Sentiment-Augmented NEPSE Prediction.

This module combines **sentiment signals** (from Nepali news via the
hybrid pipeline) with classical **technical indicators** (RSI, MACD,
moving averages, volatility, volume z-score) to build a model-ready
feature matrix for downstream walk-forward prediction.

Public API
----------
- :func:`aggregate_daily_sentiment`  -- collapse news-level sentiment to daily.
- :func:`rsi`                        -- Relative Strength Index.
- :func:`macd`                       -- MACD (line, signal, hist).
- :func:`moving_averages`            -- simple moving averages.
- :func:`technical_indicators`       -- full technical feature stack.
- :func:`build_feature_matrix`      -- merged sentiment + technical frame.

All functions are pure (no I/O) and operate on :class:`pandas.DataFrame`
/ :class:`pandas.Series` objects, so they are trivially testable.

License: MIT (see LICENSE at repo root).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

# Lazy import of shared utilities. The utils module is provided by the
# lead agent (Task 1-d). If it is unavailable for any reason, we degrade
# gracefully to plain stdlib defaults so that this module still imports
# and can be syntax-checked / unit-tested in isolation.
try:  # pragma: no cover - import guard
    from nepse_nlp.utils import set_seed, get_logger, PROCESSED_DIR, MODELS_DIR

    _LOGGER = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging

    _LOGGER = logging.getLogger("nepse_nlp.features")
    PROCESSED_DIR = None  # type: ignore[assignment]
    MODELS_DIR = None  # type: ignore[assignment]

    def set_seed(seed: int = 42) -> None:  # type: ignore[no-redef]
        import random

        random.seed(seed)
        np.random.seed(seed)


__all__ = [
    "aggregate_daily_sentiment",
    "rsi",
    "macd",
    "moving_averages",
    "technical_indicators",
    "build_feature_matrix",
]


# ---------------------------------------------------------------------------
# Sentiment aggregation
# ---------------------------------------------------------------------------
def aggregate_daily_sentiment(df_news: pd.DataFrame) -> pd.DataFrame:
    """Aggregate news-level sentiment to daily features.

    Expected input columns (all optional but recommended)
    -----------------------------------------------
    - ``date`` (str / datetime): publication date (day granularity).
    - ``confidence`` (float): sentiment confidence in [0, 1].
    - ``label`` (str): one of {positive, negative, neutral} (LABELS_3).

    Output columns (all prefixed ``sent_``)
    --------------------------------------
    - ``sent_mean_conf``        -- daily mean confidence.
    - ``sent_median_conf``      -- daily median confidence.
    - ``sent_std_conf``         -- daily std of confidence.
    - ``sent_pos_ratio``        -- fraction of positive articles.
    - ``sent_neg_ratio``        -- fraction of negative articles.
    - ``sent_n_articles``       -- article count for the day.
    - ``sent_momentum``         -- mean_conf_t - mean_conf_{t-1}.
    - ``sent_dispersion``       -- std_conf / (mean_conf + 1e-9).

    Parameters
    ----------
    df_news : pandas.DataFrame
        News-level sentiment records.

    Returns
    -------
    pandas.DataFrame
        One row per trading day, indexed by date.
    """
    if df_news is None or len(df_news) == 0:
        return pd.DataFrame(
            columns=[
                "sent_mean_conf",
                "sent_median_conf",
                "sent_std_conf",
                "sent_pos_ratio",
                "sent_neg_ratio",
                "sent_n_articles",
                "sent_momentum",
                "sent_dispersion",
            ]
        )

    df = df_news.copy()
    # Normalize date column to day-precision.
    if "date" not in df.columns:
        raise ValueError("df_news must contain a 'date' column")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # Default confidence to 1.0 if missing (treat as maximally confident).
    if "confidence" not in df.columns:
        df["confidence"] = 1.0
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)

    # Build per-day aggregates.
    grouped = df.groupby("date")
    agg = pd.DataFrame(index=grouped.groups.keys())
    agg = agg.sort_index()
    agg.index = pd.to_datetime(agg.index)

    agg["sent_mean_conf"] = grouped["confidence"].mean()
    agg["sent_median_conf"] = grouped["confidence"].median()
    agg["sent_std_conf"] = grouped["confidence"].std().fillna(0.0)
    agg["sent_n_articles"] = grouped.size()

    # Sentiment mix ratios -- require a 'label' column.
    if "label" in df.columns:
        labels_norm = df["label"].astype(str).str.lower()
        is_pos = (labels_norm == "positive").astype(int)
        is_neg = (labels_norm == "negative").astype(int)
        df_tmp = df.assign(_pos=is_pos.values, _neg=is_neg.values)
        grp_tmp = df_tmp.groupby("date")
        agg["sent_pos_ratio"] = grp_tmp["_pos"].mean()
        agg["sent_neg_ratio"] = grp_tmp["_neg"].mean()
    else:
        # Without labels we cannot compute mix ratios -> default to 0.
        agg["sent_pos_ratio"] = 0.0
        agg["sent_neg_ratio"] = 0.0

    # Momentum (today - yesterday).
    agg["sent_momentum"] = agg["sent_mean_conf"].diff().fillna(0.0)
    # Dispersion = std / (mean + eps).
    agg["sent_dispersion"] = agg["sent_std_conf"] / (agg["sent_mean_conf"].abs() + 1e-9)

    # Ensure all expected columns exist even if input was sparse.
    for col in (
        "sent_mean_conf",
        "sent_median_conf",
        "sent_std_conf",
        "sent_pos_ratio",
        "sent_neg_ratio",
        "sent_n_articles",
        "sent_momentum",
        "sent_dispersion",
    ):
        if col not in agg.columns:
            agg[col] = 0.0

    return agg.sort_index()


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI).

    RSI = 100 - 100 / (1 + RS), where RS = avg_gain / avg_loss over
    ``period`` days using Wilder's smoothing.

    Parameters
    ----------
    close : pandas.Series
        Closing prices.
    period : int, default 14
        Lookback window.

    Returns
    -------
    pandas.Series
        RSI values, same index as ``close``, bounded in [0, 100].
    """
    close = pd.Series(close).astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's smoothing == EMA with alpha = 1/period.
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss.replace(0.0, np.nan))
    rsi_vals = 100.0 - (100.0 / (1.0 + rs))
    # Where avg_loss is 0 (no losses in window) -> RSI = 100.
    rsi_vals = rsi_vals.fillna(100.0)
    # Initial NaNs from warm-up.
    rsi_vals.iloc[: period] = np.nan
    return rsi_vals.rename("rsi")


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD indicator.

    Parameters
    ----------
    close : pandas.Series
        Closing prices.
    fast : int, default 12
        Fast EMA span.
    slow : int, default 26
        Slow EMA span.
    signal : int, default 9
        Signal line EMA span.

    Returns
    -------
    (macd_line, signal_line, hist) : tuple of pandas.Series
    """
    close = pd.Series(close).astype(float)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = (ema_fast - ema_slow).rename("macd")
    signal_line = macd_line.ewm(span=signal, adjust=False).mean().rename("macd_signal")
    hist = (macd_line - signal_line).rename("macd_hist")
    return macd_line, signal_line, hist


def moving_averages(close: pd.Series, windows=(5, 10, 20, 50)) -> pd.DataFrame:
    """Compute simple moving averages for multiple window sizes.

    Parameters
    ----------
    close : pandas.Series
        Closing prices.
    windows : iterable of int
        Window sizes (default 5, 10, 20, 50).

    Returns
    -------
    pandas.DataFrame
        One column per window, named ``ma_<window>``.
    """
    close = pd.Series(close).astype(float)
    out = pd.DataFrame(index=close.index)
    for w in windows:
        out[f"ma_{int(w)}"] = close.rolling(window=int(w), min_periods=1).mean()
    return out


def technical_indicators(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Augment an OHLCV frame with all technical features.

    Adds
    ----
    - ``rsi``                       -- 14-period RSI.
    - ``macd``, ``macd_signal``, ``macd_hist``.
    - ``ma_5``, ``ma_10``, ``ma_20``, ``ma_50``.
    - ``returns``                   -- pct change of close.
    - ``log_ret``                   -- log returns.
    - ``volatility``                -- rolling std of returns (20d).
    - ``volume_zscore``            -- 20d z-scored volume.

    Parameters
    ----------
    df_ohlcv : pandas.DataFrame
        Must contain at least ``close`` (and ``volume`` for the
        volume z-score feature).

    Returns
    -------
    pandas.DataFrame
        Copy of input with added columns.
    """
    if df_ohlcv is None or "close" not in df_ohlcv.columns:
        raise ValueError("df_ohlcv must contain a 'close' column")
    df = df_ohlcv.copy()
    close = df["close"].astype(float)

    # RSI
    df["rsi"] = rsi(close, period=14)

    # MACD
    macd_line, signal_line, hist = macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist

    # Moving averages
    ma_df = moving_averages(close, windows=(5, 10, 20, 50))
    for col in ma_df.columns:
        df[col] = ma_df[col]

    # Returns
    df["returns"] = close.pct_change()
    df["log_ret"] = np.log(close / close.shift(1))

    # Volatility (20d rolling std of returns).
    df["volatility"] = df["returns"].rolling(window=20, min_periods=5).std()

    # Volume z-score (only if volume column present).
    if "volume" in df.columns:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        roll_mean = vol.rolling(window=20, min_periods=5).mean()
        roll_std = vol.rolling(window=20, min_periods=5).std()
        df["volume_zscore"] = (vol - roll_mean) / (roll_std.replace(0.0, np.nan))
    else:
        df["volume_zscore"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Merged feature matrix
# ---------------------------------------------------------------------------
def build_feature_matrix(
    df_ohlcv: pd.DataFrame,
    df_sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """Merge OHLCV technical features with daily sentiment features.

    Pipeline
    --------
    1. Sort OHLCV by date, normalise index to datetime.
    2. Aggregate news sentiment to daily level (idempotent -- already
       aggregated frames pass through unchanged if they have ``sent_``
       columns).
    3. Forward-fill sentiment onto trading days (news from weekend is
       carried into Monday).
    4. Compute all technical indicators.
    5. Create target ``next_ret``  = close_{t+1} / close_t - 1.
    6. Create target ``next_dir`` = sign(next_ret) in {-1, 0, +1}.
    7. Drop rows with NaN in any feature column (warm-up period).

    The target is **strictly next-day**, so there is no look-ahead in
    the feature columns: a row's features only see information up to
    time ``t`` inclusive, while ``next_ret`` is the realised return at
    ``t+1``.

    Parameters
    ----------
    df_ohlcv : pandas.DataFrame
        Daily OHLCV with a ``date`` column or DatetimeIndex, must
        contain ``close``.
    df_sentiment : pandas.DataFrame
        Either news-level (with ``date`` + ``confidence`` + ``label``)
        or already aggregated daily (with ``sent_`` columns).

    Returns
    -------
    pandas.DataFrame
        Model-ready frame indexed by date with feature columns,
        ``next_ret`` and ``next_dir``.
    """
    if df_ohlcv is None or len(df_ohlcv) == 0:
        raise ValueError("df_ohlcv must be non-empty")

    df = df_ohlcv.copy()
    # Date normalisation.
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Sentiment -> daily.
    if df_sentiment is None or len(df_sentiment) == 0:
        sent_daily = pd.DataFrame(index=df.index)
        for col in (
            "sent_mean_conf",
            "sent_median_conf",
            "sent_std_conf",
            "sent_pos_ratio",
            "sent_neg_ratio",
            "sent_n_articles",
            "sent_momentum",
            "sent_dispersion",
        ):
            sent_daily[col] = 0.0
    else:
        # If already aggregated (has sent_ columns), assume it is daily.
        if any(c.startswith("sent_") for c in df_sentiment.columns):
            sent_daily = df_sentiment.copy()
            sent_daily.index = pd.to_datetime(sent_daily.index)
        else:
            sent_daily = aggregate_daily_sentiment(df_sentiment)
        sent_daily = sent_daily.sort_index()

    # Reindex onto trading days and forward-fill weekend news.
    sent_daily = sent_daily.reindex(df.index, method="ffill").fillna(0.0)

    # Technical indicators (operates on df with close column).
    df = technical_indicators(df)

    # Merge sentiment.
    for col in sent_daily.columns:
        df[col] = sent_daily[col].values

    # Targets: next-day close-to-close return.
    df["next_ret"] = (df["close"].shift(-1) / df["close"]) - 1.0
    df["next_dir"] = np.sign(df["next_ret"]).astype(float)

    # Define feature columns = everything except targets and raw OHLCV.
    exclude = {"open", "high", "low", "close", "volume", "next_ret", "next_dir"}
    feature_cols = [c for c in df.columns if c not in exclude]

    # Drop rows with NaN in any feature column (warm-up).
    df = df.dropna(subset=feature_cols + ["next_ret", "next_dir"]).copy()

    # Stash feature names as attribute for downstream callers.
    df.attrs["feature_cols"] = feature_cols
    df.attrs["target_ret"] = "next_ret"
    df.attrs["target_dir"] = "next_dir"
    return df
