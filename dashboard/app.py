"""Streamlit dashboard for NEPSE sentiment + prediction (Phase 5).

Run with::

    streamlit run dashboard/app.py

If ``streamlit`` is not installed, the file still imports cleanly
(prints a help message) so it can be syntax-checked in minimal
environments.

Sections
--------
1. Live sentiment classification (calls the FastAPI ``/sentiment``
   endpoint, falls back to a direct :class:`HybridPipeline` call).
2. NEPSE price chart with sentiment overlay.
3. Backtest equity curve + metrics table.
4. Model comparison bar chart.

License: MIT (see LICENSE at repo root).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

# --- sys.path bootstrap so nepse_nlp resolves when run from repo root ---
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# --- Optional dependency guards ---
try:  # pragma: no cover - optional dep
    import streamlit as st
    _ST_AVAILABLE = True
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]
    _ST_AVAILABLE = False

try:  # pragma: no cover
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None  # type: ignore[assignment]

try:  # pragma: no cover
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except Exception:  # pragma: no cover
    go = None  # type: ignore[assignment]
    _PLOTLY_AVAILABLE = False

try:  # pragma: no cover
    import altair as alt
    _ALTAIR_AVAILABLE = True
except Exception:  # pragma: no cover
    alt = None  # type: ignore[assignment]
    _ALTAIR_AVAILABLE = False

try:  # pragma: no cover
    import requests
    _REQUESTS_AVAILABLE = True
except Exception:  # pragma: no cover
    requests = None  # type: ignore[assignment]
    _REQUESTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Fallback for missing Streamlit
# ---------------------------------------------------------------------------
if not _ST_AVAILABLE:  # pragma: no cover - runtime guard
    print(
        "[dashboard.app] Streamlit is not installed -- cannot launch the "
        "dashboard UI. Install with: pip install streamlit\n"
        "Then run: streamlit run dashboard/app.py"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fetch_sentiment_via_api(text: str, base_url: str = "http://localhost:8000") -> Optional[Dict[str, Any]]:
    """Call the FastAPI ``/sentiment`` endpoint; return None on failure."""
    if not _REQUESTS_AVAILABLE:
        return None
    try:
        resp = requests.post(
            f"{base_url}/sentiment",
            json={"text": text},
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None


def _fetch_sentiment_via_pipeline(text: str) -> Dict[str, Any]:
    """Direct call to HybridPipeline as a fallback."""
    try:
        from nepse_nlp.hybrid_pipeline import HybridPipeline  # lazy
        from nepse_nlp.jev_benchmark import JevClassifier  # lazy
        clf = JevClassifier()
        pipe = HybridPipeline(classifier=clf)
        result = pipe.predict(text)
        if isinstance(result, dict):
            return result
        return {
            "label": getattr(result, "label", "unknown"),
            "confidence": float(getattr(result, "confidence", 0.0)),
            "source": getattr(result, "source", "hybrid"),
        }
    except Exception as exc:
        return {"label": "error", "confidence": 0.0, "source": f"error: {exc}"}


def _load_synthetic_nepese(n_days: int = 250, seed: int = 42) -> "pd.DataFrame":
    """Build a synthetic NEPSE OHLCV frame for demo / fallback."""
    try:
        from nepse_nlp.scraper import simulate_nepse_series  # lazy
    except Exception as exc:
        return pd.DataFrame()
    try:
        close = simulate_nepse_series(n_days=n_days, seed=seed)
    except Exception:
        # Fallback: simulate a random-walk close.
        import numpy as np
        rng = __import__("numpy").default_rng(seed)
        rets = rng.normal(0.0003, 0.012, size=n_days)
        close = 2000.0 * (1.0 + rets).cumprod()
    df = pd.DataFrame({
        "date": pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="B"),
        "open": close * (1 + (pd.Series(close).pct_change().fillna(0) * 0.5)),
        "high": close * (1 + pd.Series(close).pct_change().abs().fillna(0) * 0.7),
        "low": close * (1 - pd.Series(close).pct_change().abs().fillna(0) * 0.7),
        "close": close,
        "volume": pd.Series(close).rolling(5, min_periods=1).mean() * 1000,
    })
    return df


def _load_synthetic_sentiment(n_days: int = 250, seed: int = 42) -> "pd.DataFrame":
    """Build a synthetic daily-sentiment frame."""
    import numpy as np
    rng = np.default_rng(seed + 1)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days, freq="B")
    labels = rng.choice(["positive", "negative", "neutral"], size=n_days, p=[0.4, 0.3, 0.3])
    conf = rng.uniform(0.5, 0.99, size=n_days)
    return pd.DataFrame({"date": dates, "label": labels, "confidence": conf})


# ---------------------------------------------------------------------------
# UI sections (only built if Streamlit is available)
# ---------------------------------------------------------------------------
def _section_sentiment() -> None:
    st.subheader("1. Live sentiment classification")
    text = st.text_area(
        "Enter a Nepali news snippet / headline:",
        value="नेप्से आज उकालो भयो",
        height=80,
    )
    if st.button("Classify"):
        result = _fetch_sentiment_via_api(text)
        if result is None:
            st.info("API unavailable -- calling HybridPipeline directly.")
            result = _fetch_sentiment_via_pipeline(text)
        st.json(result)
        st.metric(
            label="Predicted label",
            value=str(result.get("label", "?")),
            delta=f"{result.get('confidence', 0):.2f}",
        )


def _section_price_chart() -> None:
    st.subheader("2. NEPSE price + sentiment overlay")
    n_days = st.slider("Lookback (trading days)", 30, 500, 250, step=10)
    df_price = _load_synthetic_nepese(n_days=n_days)
    df_sent = _load_synthetic_sentiment(n_days=n_days)
    if df_price is None or df_price.empty:
        st.warning("Price data unavailable.")
        return
    if _PLOTLY_AVAILABLE:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df_price["date"],
            open=df_price["open"],
            high=df_price["high"],
            low=df_price["low"],
            close=df_price["close"],
            name="NEPSE",
        ))
        # Overlay sentiment as a bar trace on a secondary axis.
        sent_proxy = df_sent["confidence"] * df_sent["label"].map(
            {"positive": 1, "neutral": 0, "negative": -1}
        ).fillna(0)
        fig.add_trace(go.Bar(
            x=df_sent["date"], y=sent_proxy, name="Sentiment",
            opacity=0.4, marker_color="orange", yaxis="y2",
        ))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right"))
        st.plotly_chart(fig, use_container_width=True)
    elif _ALTAIR_AVAILABLE:
        chart = alt.Chart(df_price).mark_line().encode(
            x="date:T", y="close:Q"
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.line_chart(df_price.set_index("date")["close"], use_container_width=True)


def _section_backtest() -> None:
    st.subheader("3. Backtest equity curve + metrics")
    if st.button("Run backtest (offline sim)"):
        try:
            from nepse_nlp.features import build_feature_matrix
            from nepse_nlp.backtest import backtest_strategy
            df_price = _load_synthetic_nepese(n_days=400)
            df_sent = _load_synthetic_sentiment(n_days=400)
            feats = build_feature_matrix(df_price, df_sent)
            result = backtest_strategy(
                df=feats, model_type="xgboost", target="direction",
                min_train=120, step=1,
            )
            tm = result["trading_metrics"]
            st.metric("Final equity", value=f"{tm['final_equity']:,.0f}")
            st.metric("Total return (%)", value=f"{tm['total_return_pct']:.2f}")
            st.metric("Sharpe", value=f"{tm['sharpe']:.2f}")
            st.metric("Max drawdown", value=f"{tm['max_drawdown']*100:.2f}%")
            st.metric("Win rate", value=f"{tm['win_rate']*100:.1f}%")
            st.metric("Trades", value=str(tm["n_trades"]))
            eq = result.get("equity_curve")
            if eq is not None and len(eq) > 0:
                if _PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=eq.index, y=eq.values, mode="lines", name="Equity",
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.line_chart(eq)
        except Exception as exc:
            st.error(f"Backtest failed: {exc}")


def _section_model_comparison() -> None:
    st.subheader("4. Model comparison")
    if st.button("Run compare_models"):
        try:
            from nepse_nlp.features import build_feature_matrix
            from nepse_nlp.prediction import compare_models
            df_price = _load_synthetic_nepese(n_days=400)
            df_sent = _load_synthetic_sentiment(n_days=400)
            feats = build_feature_matrix(df_price, df_sent)
            results = compare_models(feats, target="direction", step=5)
            # Build a small bar chart of directional accuracies.
            rows = [(m, r.get("directional_accuracy", 0.0)) for m, r in results.items()]
            df_cmp = pd.DataFrame(rows, columns=["model", "directional_accuracy"])
            if _PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Bar(x=df_cmp["model"], y=df_cmp["directional_accuracy"]))
                fig.update_layout(yaxis_title="Directional accuracy")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.bar_chart(df_cmp.set_index("model"))
            st.dataframe(df_cmp, use_container_width=True)
        except Exception as exc:
            st.error(f"Comparison failed: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not _ST_AVAILABLE:  # pragma: no cover
        return
    st.set_page_config(
        page_title="NEPSE Sentiment Dashboard",
        page_icon="📈",
        layout="wide",
    )
    st.title("NEPSE Sentiment & Prediction Dashboard")
    st.markdown(
        "Live Nepali news sentiment classification + NEPSE direction-"
        "prediction + backtest. Built on the Jev + Confidence-Gated "
        "Hybrid Architectures pipeline."
    )

    # Sidebar.
    st.sidebar.header("Configuration")
    model_choice = st.sidebar.selectbox(
        "Classifier",
        options=["Jev", "XLM-R", "Hybrid"],
        index=2,
        help="Jev = System One zero-shot; XLM-R = fine-tuned baseline; "
             "Hybrid = Jev+confidence-gated escalation to XLM-R.",
    )
    conf_threshold = st.sidebar.slider(
        "Confidence threshold (Jev -> escalate above)",
        min_value=0.50, max_value=0.99, value=0.80, step=0.01,
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Model: {model_choice} | Threshold: {conf_threshold:.2f}")

    # Sections.
    _section_sentiment()
    st.markdown("---")
    _section_price_chart()
    st.markdown("---")
    _section_backtest()
    st.markdown("---")
    _section_model_comparison()


if __name__ == "__main__" or _ST_AVAILABLE:  # pragma: no cover
    if _ST_AVAILABLE:
        main()
    else:
        print(
            "[dashboard.app] Streamlit not installed. "
            "Install with: pip install streamlit"
        )
