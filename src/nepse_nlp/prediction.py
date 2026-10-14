"""NEPSE prediction models (Phase 4).

This module implements the model layer for **Phase 4 — Sentiment-Augmented
NEPSE Prediction**. It exposes:

- :class:`NEPSEPredictor` -- a thin wrapper around XGBoost, Logistic
  Regression, and (optionally) an LSTM, with standardised fit / predict /
  feature-importance API.
- :func:`walk_forward` -- expanding-window walk-forward evaluation
  (strictly no look-ahead).
- :func:`compare_models` -- cross-model comparison.
- :func:`ablation_sentiment_vs_baseline` -- sentiment-vs-baseline
  ablation that reports the directional-accuracy delta attributable to
  sentiment features.

The :mod:`torch` dependency (for the LSTM option) is **optional**: if it
is unavailable, ``NEPSEPredictor("lstm", ...)`` silently falls back to a
logistic regression and logs a warning, so the rest of the pipeline
remains usable on CPU-only or minimal environments.

License: MIT (see LICENSE at repo root).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.preprocessing import StandardScaler

# Lazy / guarded torch import.
try:  # pragma: no cover - optional dep
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    _TORCH_AVAILABLE = False

# Lazy / guarded xgboost import (declared available, but kept robust).
try:  # pragma: no cover
    import xgboost as xgb

    _XGB_AVAILABLE = True
except Exception:  # pragma: no cover
    xgb = None  # type: ignore
    _XGB_AVAILABLE = False

# Lazy / shared utils import.
try:  # pragma: no cover
    from nepse_nlp.utils import get_logger, set_seed

    _LOGGER = get_logger(__name__)
except Exception:  # pragma: no cover
    import logging

    _LOGGER = logging.getLogger("nepse_nlp.prediction")

    def set_seed(seed: int = 42) -> None:  # type: ignore[no-redef]
        import random

        random.seed(seed)
        np.random.seed(seed)


__all__ = [
    "NEPSEPredictor",
    "walk_forward",
    "compare_models",
    "ablation_sentiment_vs_baseline",
]

# Sentinel list of feature names to ignore when stripping sentiment for
# the ablation study.
_SENTIMENT_PREFIX = "sent_"


# ---------------------------------------------------------------------------
# LSTM (optional)
# ---------------------------------------------------------------------------
class _LSTMClassifier(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """A minimal single-layer LSTM binary/3-class classifier.

    Only constructed when ``torch`` is available. Falls back gracefully
    elsewhere (the class is the bare ``object``).
    """

    def __init__(self, n_features: int, hidden: int = 32, n_classes: int = 3):
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("torch is required for the LSTM model")
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):  # type: ignore[no-untyped-def]
        # x: (batch, seq, feat)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


# ---------------------------------------------------------------------------
# XGBoost helper
# ---------------------------------------------------------------------------
def _xgb_classifier_kwargs(n_classes: int, seed: int) -> dict:
    """Build kwargs for XGBClassifier that handle binary/multiclass.

    xgboost 2.x infers ``num_class`` automatically from the training
    labels, but the ``objective`` and ``eval_metric`` must match the
    class count: binary classification uses ``binary:logistic`` +
    ``logloss``, while multi-class uses ``multi:softprob`` +
    ``mlogloss``. Hard-coding ``num_class=3`` breaks binary data with
    an ``IndexError`` (the classic "arrays used as indices must be of
    integer type" traceback), so we choose the kwargs dynamically.
    """
    if n_classes <= 2:
        return dict(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            random_state=seed,
            eval_metric="logloss",
            tree_method="hist",
            verbosity=0,
        )
    return dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_classes,
        random_state=seed,
        eval_metric="mlogloss",
        tree_method="hist",
        verbosity=0,
    )


# ---------------------------------------------------------------------------
# Main predictor class
# ---------------------------------------------------------------------------
class NEPSEPredictor:
    """Unified NEPSE prediction interface.

    Parameters
    ----------
    model_type : str
        One of {"xgboost", "logreg", "lstm"}. The LSTM requires ``torch``
        to be installed; otherwise it falls back to ``logreg``.
    target : str
        "direction" (classification on next-day sign) or "return"
        (regression on next-day return).
    seed : int
        Random seed used by every underlying estimator.

    Notes
    -----
    Features are standardised for the ``logreg`` and ``lstm`` variants;
    XGBoost is trained on raw features (tree models are scale-invariant).
    """

    _VALID_MODELS = {"xgboost", "logreg", "lstm"}
    _VALID_TARGETS = {"direction", "return"}

    def __init__(
        self,
        model_type: str = "xgboost",
        target: str = "direction",
        seed: int = 42,
    ) -> None:
        if model_type not in self._VALID_MODELS:
            raise ValueError(f"model_type must be in {self._VALID_MODELS}; got {model_type!r}")
        if target not in self._VALID_TARGETS:
            raise ValueError(f"target must be in {self._VALID_TARGETS}; got {target!r}")

        set_seed(seed)
        self.seed = seed
        self.target = target
        self._classification = target == "direction"

        # LSTM fallback.
        if model_type == "lstm" and not _TORCH_AVAILABLE:
            _LOGGER.warning(
                "torch is not installed -- LSTM requested but falling "
                "back to logistic regression. Install torch to enable the "
                "LSTM backend."
            )
            model_type = "logreg"
        self.model_type = model_type

        # Standardisation is applied for logreg / lstm only.
        self.scale = model_type in {"logreg", "lstm"}
        self.scaler: Optional[StandardScaler] = None
        self.model = None
        self._lstm: Optional[_LSTMClassifier] = None
        self.feature_names_: List[str] = []
        # Track label encoding for direction classification.
        self._classes_: Optional[np.ndarray] = None
        # Number of input features -- needed for LSTM init.
        self._n_features: Optional[int] = None
        # Number of unique classes observed at fit time -- used to pick
        # the correct XGBoost objective (binary vs multi-class).
        self._n_known_classes: Optional[int] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _y_to_classes(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map {-1, 0, +1} (or floats) to non-negative class indices.

        Returns the class indices and the unique sorted labels.
        """
        y_arr = np.asarray(y).ravel()
        classes = np.unique(y_arr)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        idx = np.array([class_to_idx[v] for v in y_arr], dtype=int)
        return idx, classes

    # ------------------------------------------------------------------
    # Estimator factory
    # ------------------------------------------------------------------
    def _make_estimator(self, n_features: int):
        """Construct the underlying estimator instance.

        The XGBoost classifier's objective / eval_metric / num_class
        are chosen dynamically from :func:`_xgb_classifier_kwargs`
        based on the number of classes seen during :meth:`fit` (stored
        in ``self._n_known_classes``). This avoids the well-known
        xgboost 2.x ``IndexError`` when binary data is trained with a
        multi-class objective.
        """
        if self.model_type == "xgboost":
            if not _XGB_AVAILABLE:  # pragma: no cover
                raise RuntimeError("xgboost is not available")
            if self._classification:
                n_cls = max(2, int(self._n_known_classes or 2))
                return xgb.XGBClassifier(**_xgb_classifier_kwargs(n_cls, self.seed))
            return xgb.XGBRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=self.seed,
                tree_method="hist",
                verbosity=0,
            )
        if self.model_type == "logreg":
            if self._classification:
                return LogisticRegression(
                    max_iter=1000,
                    random_state=self.seed,
                    class_weight="balanced",
                )
            return LinearRegression()
        if self.model_type == "lstm":
            if not _TORCH_AVAILABLE:  # pragma: no cover
                raise RuntimeError("torch is not available")
            # Build the LSTM lazily; class count derived from y in fit.
            self._lstm = _LSTMClassifier(
                n_features=n_features,
                hidden=32,
                n_classes=3,
            )
            return self._lstm
        raise ValueError(f"Unknown model_type {self.model_type!r}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X, y) -> "NEPSEPredictor":
        """Fit the model on (X, y).

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix.
        y : pandas.Series or numpy.ndarray
            Target. For ``target='direction'``, expects values in
            {-1, 0, +1} (or arbitrary discrete values); for
            ``target='return'``, expects continuous returns.

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values.astype(float)
        else:
            X_arr = np.asarray(X, dtype=float)
            self.feature_names_ = [f"f{i}" for i in range(X_arr.shape[1])]
        y_arr = np.asarray(y).ravel()
        if isinstance(y, pd.Series):
            y_arr = y.values

        self._n_features = X_arr.shape[1]

        # Compute the number of distinct classes in the training target
        # BEFORE building the estimator. This is used to pick the right
        # XGBoost objective (binary vs multi-class).
        if self._classification:
            self._n_known_classes = int(np.unique(y_arr).size)
        else:
            self._n_known_classes = None

        # Build / reset model.
        self.model = self._make_estimator(self._n_features)

        # Standardise for logreg / lstm.
        if self.scale:
            self.scaler = StandardScaler()
            X_arr = self.scaler.fit_transform(X_arr)

        if self.model_type == "lstm" and _TORCH_AVAILABLE:
            # Encode classes for cross-entropy.
            idx, classes = self._y_to_classes(y_arr)
            self._classes_ = classes
            self._train_lstm(X_arr, idx)
            return self

        if self._classification:
            idx, classes = self._y_to_classes(y_arr)
            self._classes_ = classes
            self.model.fit(X_arr, idx)
        else:
            self.model.fit(X_arr, y_arr)
        return self

    # ------------------------------------------------------------------
    # LSTM training loop (guarded)
    # ------------------------------------------------------------------
    def _train_lstm(self, X: np.ndarray, y_idx: np.ndarray) -> None:
        """Internal minimal LSTM training loop."""
        if not _TORCH_AVAILABLE:  # pragma: no cover
            raise RuntimeError("torch not available")
        n_samples, n_features = X.shape
        # Build a trivial sequence of length 1 so the LSTM sees a
        # (batch, 1, feat) tensor.
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y_t = torch.tensor(y_idx, dtype=torch.long)
        n_classes = int(self._classes_.size) if self._classes_ is not None else 3
        # Rebuild LSTM with the right class count.
        self._lstm = _LSTMClassifier(
            n_features=n_features,
            hidden=32,
            n_classes=n_classes,
        )
        optim = torch.optim.Adam(self._lstm.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        self._lstm.train()
        n_epochs = 30
        batch_size = 32
        for _ in range(n_epochs):
            perm = torch.randperm(n_samples)
            for b in range(0, n_samples, batch_size):
                batch_idx = perm[b : b + batch_size]
                xb = X_t[batch_idx]
                yb = y_t[batch_idx]
                optim.zero_grad()
                out = self._lstm(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                optim.step()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def _prep_X(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(float)
        else:
            X_arr = np.asarray(X, dtype=float)
        if self.scale and self.scaler is not None:
            X_arr = self.scaler.transform(X_arr)
        return X_arr

    def predict(self, X) -> np.ndarray:
        """Return predicted labels (direction) or values (return)."""
        X_arr = self._prep_X(X)
        if self.model_type == "lstm" and _TORCH_AVAILABLE:
            self._lstm.eval()
            with torch.no_grad():
                # (batch, 1, feat)
                X_t = torch.tensor(X_arr, dtype=torch.float32).unsqueeze(1)
                logits = self._lstm(X_t)
                idx_pred = logits.argmax(dim=-1).cpu().numpy()
            if self._classes_ is not None:
                return self._classes_[idx_pred]
            return idx_pred.astype(float)
        if self._classification:
            idx_pred = self.model.predict(X_arr)
            if self._classes_ is not None:
                return self._classes_[idx_pred]
            return idx_pred.astype(float)
        return self.model.predict(X_arr)

    def predict_proba(self, X) -> np.ndarray:
        """Class probabilities for classification, NaNs for regression."""
        if not self._classification:
            # Regression has no probabilities.
            return np.full((len(X), 1), np.nan)
        X_arr = self._prep_X(X)
        if self.model_type == "lstm" and _TORCH_AVAILABLE:
            self._lstm.eval()
            with torch.no_grad():
                X_t = torch.tensor(X_arr, dtype=torch.float32).unsqueeze(1)
                logits = self._lstm(X_t)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            return probs
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_arr)
        # Fallback for sklearn models without predict_proba.
        idx_pred = self.model.predict(X_arr)
        n = len(idx_pred)
        out = np.zeros((n, 3))
        out[np.arange(n), idx_pred.astype(int)] = 1.0
        return out

    def feature_importance(self) -> Dict[str, float]:
        """Return {feature_name: importance}.

        For XGBoost, uses the native feature_importances_.
        For logistic regression, returns |coef_| summed across classes.
        For LinearRegression, returns |coef_| per feature.
        For LSTM, returns uniform weights (placeholder).
        """
        if not self.feature_names_:
            return {}
        if self.model_type == "xgboost":
            fi = getattr(self.model, "feature_importances_", None)
            if fi is None:
                return {n: 0.0 for n in self.feature_names_}
            return dict(zip(self.feature_names_, map(float, fi)))
        if self.model_type == "logreg":
            if self._classification:
                coef = getattr(self.model, "coef_", None)
                if coef is None:
                    return {n: 0.0 for n in self.feature_names_}
                importance = np.abs(coef).sum(axis=0)
            else:
                coef = getattr(self.model, "coef_", None)
                importance = (
                    np.abs(coef).ravel() if coef is not None else np.zeros(len(self.feature_names_))
                )
            return dict(zip(self.feature_names_, map(float, importance)))
        if self.model_type == "lstm":
            return {n: 1.0 / len(self.feature_names_) for n in self.feature_names_}
        return {}


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------
def walk_forward(
    df: pd.DataFrame,
    model_type: str = "xgboost",
    target: str = "direction",
    min_train: int = 120,
    step: int = 1,
    seed: int = 42,
) -> Dict:
    """Expanding-window walk-forward evaluation.

    At time ``t`` the model is trained on rows ``[0..t-1]`` and asked
    to predict row ``t``. This guarantees no look-ahead leakage.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of :func:`nepse_nlp.features.build_feature_matrix`. Must
        contain feature columns plus ``next_ret`` / ``next_dir``.
    model_type : str
        Passed to :class:`NEPSEPredictor`.
    target : str
        "direction" (classification) or "return" (regression).
    min_train : int, default 120
        Minimum training rows before starting to produce predictions.
    step : int, default 1
        Stride through the test set (use > 1 to subsample for speed).
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: ``predictions``, ``actuals``, ``dates``, ``metrics``.
        ``metrics`` has ``rmse``, ``mae``, ``directional_accuracy``,
        ``hit_rate``.
    """
    if df is None or len(df) < min_train + 1:
        raise ValueError(f"Need at least {min_train + 1} rows; got {0 if df is None else len(df)}")

    feature_cols = df.attrs.get(
        "feature_cols",
        [c for c in df.columns if c not in {"next_ret", "next_dir"}],
    )
    y_col = "next_dir" if target == "direction" else "next_ret"

    predictions: List[float] = []
    actuals: List[float] = []
    dates: List = []

    n = len(df)
    # Test indices.
    test_indices = list(range(min_train, n, step))

    for t in test_indices:
        # Train on rows strictly before t (expanding window).
        train_df = df.iloc[:t]
        test_row = df.iloc[[t]]
        X_train = train_df[feature_cols]
        y_train = train_df[y_col]
        X_test = test_row[feature_cols]
        y_test = test_row[y_col].iloc[0]

        try:
            model = NEPSEPredictor(
                model_type=model_type,
                target=target,
                seed=seed,
            )
            model.fit(X_train, y_train)
            y_hat = float(model.predict(X_test)[0])
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.warning("walk_forward prediction failed at t=%s: %s", t, exc)
            continue
        predictions.append(y_hat)
        actuals.append(float(y_test))
        dates.append(df.index[t])

    # Metrics.
    preds = np.array(predictions, dtype=float)
    acts = np.array(actuals, dtype=float)
    metrics: Dict[str, float] = {}
    if len(preds) > 0:
        metrics["rmse"] = float(np.sqrt(mean_squared_error(acts, preds)))
        metrics["mae"] = float(mean_absolute_error(acts, preds))
        # Directional accuracy = sign agreement (ignoring zeros).
        signed_pred = np.sign(preds)
        signed_act = np.sign(acts)
        nonzero = signed_act != 0
        if nonzero.sum() > 0:
            metrics["directional_accuracy"] = float(
                (signed_pred[nonzero] == signed_act[nonzero]).mean()
            )
        else:
            metrics["directional_accuracy"] = 0.0
        # Hit rate: fraction of correct (rounded) sign predictions.
        metrics["hit_rate"] = float((signed_pred == signed_act).sum() / max(len(signed_act), 1))
    else:
        metrics["rmse"] = float("nan")
        metrics["mae"] = float("nan")
        metrics["directional_accuracy"] = 0.0
        metrics["hit_rate"] = 0.0

    return {
        "predictions": predictions,
        "actuals": actuals,
        "dates": [str(d) for d in dates],
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------
def compare_models(
    df: pd.DataFrame,
    target: str = "direction",
    min_train: int = 120,
    step: int = 1,
) -> Dict[str, Dict[str, float]]:
    """Run walk_forward for every available model type.

    Returns
    -------
    dict
        ``{model_type: metrics}``.
    """
    out: Dict[str, Dict[str, float]] = {}
    candidate_models = ["xgboost", "logreg"]
    if _TORCH_AVAILABLE:
        candidate_models.append("lstm")
    for m in candidate_models:
        try:
            res = walk_forward(
                df=df,
                model_type=m,
                target=target,
                min_train=min_train,
                step=step,
            )
            out[m] = res["metrics"]
        except Exception as exc:  # pragma: no cover - defensive
            _LOGGER.warning("compare_models: model %s failed: %s", m, exc)
            out[m] = {
                "rmse": float("nan"),
                "mae": float("nan"),
                "directional_accuracy": 0.0,
                "hit_rate": 0.0,
            }
    return out


# ---------------------------------------------------------------------------
# Sentiment-vs-baseline ablation
# ---------------------------------------------------------------------------
def ablation_sentiment_vs_baseline(
    df: pd.DataFrame,
    target: str = "direction",
    min_train: int = 120,
    step: int = 1,
    model_type: str = "xgboost",
) -> Dict:
    """Compare sentiment-augmented model vs technical-only baseline.

    The function partitions the feature columns into two groups:

    - **Sentiment features**: names starting with ``sent_``.
    - **Baseline features**: everything else (technical indicators).

    For each group it runs :func:`walk_forward` with the same model
    type and returns the directional-accuracy delta attributable to the
    sentiment features.

    Returns
    -------
    dict
        ``{
            "with_sentiment": metrics,
            "baseline": metrics,
            "delta_directional_accuracy": float,
        }``
    """
    feature_cols: List[str] = list(
        df.attrs.get(
            "feature_cols",
            [c for c in df.columns if c not in {"next_ret", "next_dir"}],
        )
    )
    sent_cols = [c for c in feature_cols if c.startswith(_SENTIMENT_PREFIX)]
    base_cols = [c for c in feature_cols if not c.startswith(_SENTIMENT_PREFIX)]

    # If no sentiment columns are present, the ablation is trivial.
    if not sent_cols:
        res_full = walk_forward(
            df=df,
            model_type=model_type,
            target=target,
            min_train=min_train,
            step=step,
        )
        return {
            "with_sentiment": res_full["metrics"],
            "baseline": res_full["metrics"],
            "delta_directional_accuracy": 0.0,
        }

    # Re-tag the feature_cols attribute to select sub-feature sets.
    df_full = df.copy()
    df_full.attrs["feature_cols"] = feature_cols

    df_base = df.copy()
    df_base.attrs["feature_cols"] = base_cols

    res_full = walk_forward(
        df=df_full,
        model_type=model_type,
        target=target,
        min_train=min_train,
        step=step,
    )
    res_base = walk_forward(
        df=df_base,
        model_type=model_type,
        target=target,
        min_train=min_train,
        step=step,
    )
    delta = (
        res_full["metrics"]["directional_accuracy"] - res_base["metrics"]["directional_accuracy"]
    )
    return {
        "with_sentiment": res_full["metrics"],
        "baseline": res_base["metrics"],
        "delta_directional_accuracy": float(delta),
    }
