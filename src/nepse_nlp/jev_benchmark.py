"""Jev (System-One decision model) zero-shot classifier + benchmark harness.

Jev substitution
----------------
Jev is a fast System-One decision model whose public API is not guaranteed.
If the Jev API is unavailable, this module falls back to a deterministic
simulated backend so the full pipeline remains reproducible. Set
``backend='http'`` with a valid ``api_key`` to use the real Jev endpoint
at ``https://api.jev.ai/v1/decide``.

The simulated backend derives a deterministic pseudo-confidence from text
features (Nepali + English sentiment lexicon word counts, code-mix ratio,
length) and a softmax over labels. The same input always produces the same
output, so downstream calibration / conformal results are reproducible.

The HTTP backend is documented for integration but is NOT exercised in
this offline build (raising an informative RuntimeError) so that the
module can be imported and unit-tested in CI without network access.

MIT License

Copyright (c) 2025 Nepali Sentiment NEPSE Research Project
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "LABELS_3",
    "LABELS_5",
    "JevClassifier",
    "benchmark",
    "stratified_analysis",
    "POSITIVE_LEXICON",
    "NEGATIVE_LEXICON",
]

# ---------------------------------------------------------------------------
# Guarded imports of shared utilities (provided by lead in parallel).
# These are wrapped in try/except so that this module remains importable in
# CI before the lead's utils.py lands.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only when utils is available
    from nepse_nlp.utils import get_logger, set_seed  # type: ignore
except Exception:  # pragma: no cover - stub fallback
    import logging
    import random

    def set_seed(seed: int = 42) -> None:
        random.seed(seed)
        try:
            np.random.seed(seed)
        except Exception:
            pass

    def get_logger(name: str):
        return logging.getLogger(name)


# Guarded import of the sibling calibration module.
try:
    from nepse_nlp.calibration import (  # type: ignore
        brier_score,
        expected_calibration_error,
        reliability_data,
    )

    _HAS_CALIBRATION = True
except Exception:  # pragma: no cover
    _HAS_CALIBRATION = False
    brier_score = None  # type: ignore
    expected_calibration_error = None  # type: ignore
    reliability_data = None  # type: ignore


# ---------------------------------------------------------------------------
# Label sets
# ---------------------------------------------------------------------------
LABELS_3: List[str] = ["negative", "neutral", "positive"]
LABELS_5: List[str] = [
    "very_negative",
    "negative",
    "neutral",
    "positive",
    "very_positive",
]


# ---------------------------------------------------------------------------
# Small Nepali + English sentiment lexicon (15 positive + 15 negative)
# ---------------------------------------------------------------------------
POSITIVE_LEXICON: List[str] = [
    "बढ्यो",
    "नाफा",
    "उच्च",
    "मजबूत",
    "वृद्धि",
    "रेकर्ड",
    "सकारात्मक",
    "surge",
    "profit",
    "growth",
    "record",
    "strong",
    "high",
    "gain",
    "rally",
]
NEGATIVE_LEXICON: List[str] = [
    "घट्यो",
    "घाटा",
    "नोक्सान",
    "कमजोर",
    "गिरावट",
    "चिन्ता",
    "loss",
    "fall",
    "decline",
    "weak",
    "drop",
    "crash",
    "worry",
    "debt",
    "fraud",
]


# ---------------------------------------------------------------------------
# Text feature helpers (numpy + stdlib only, fully deterministic)
# ---------------------------------------------------------------------------
def _lexicon_counts(text: str) -> Dict[str, int]:
    """Return positive/negative lexicon word counts (case-insensitive)."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    t = text.lower()
    pos = sum(1 for w in POSITIVE_LEXICON if w.lower() in t)
    neg = sum(1 for w in NEGATIVE_LEXICON if w.lower() in t)
    return {"pos": pos, "neg": neg}


def _code_mix_ratio(text: str) -> float:
    """Crude code-mix estimate: fraction of ASCII alphanumeric characters.

    A text that is entirely Devanagari returns ~0.0; a text that is entirely
    English returns ~1.0; a code-mixed sentence sits in between.
    """
    if not text:
        return 0.0
    ascii_alnum = sum(1 for c in text if c.isascii() and c.isalnum())
    total_alnum = sum(1 for c in text if c.isalnum())
    if total_alnum == 0:
        return 0.0
    return ascii_alnum / total_alnum


def _length(text: str) -> int:
    """Whitespace-tokenised length."""
    return len(text.split()) if text else 0


def _softmax(x: Sequence[float]) -> List[float]:
    """Numerically-stable softmax over a 1-D sequence."""
    if not x:
        return []
    m = max(x)
    exps = [math.exp(v - m) for v in x]
    s = sum(exps)
    if s <= 0:
        n = len(x)
        return [1.0 / n for _ in x]
    return [e / s for e in exps]


# ---------------------------------------------------------------------------
# JevClassifier
# ---------------------------------------------------------------------------
class JevClassifier:
    """Jev (System-One) zero-shot sentiment classifier.

    Parameters
    ----------
    backend : str
        "simulated" (default, offline) or "http" (real Jev API).
    api_key : str | None
        Required for the "http" backend; an informative RuntimeError is
        raised at classification time if missing.
    labels : sequence[str]
        Label set; defaults to :data:`LABELS_3`.
    seed : int
        Seed for the global RNG (used by :func:`set_seed`).

    Notes
    -----
    The simulated backend is deterministic and depends only on text
    features: positive/negative lexicon counts, code-mix ratio, and length.
    A softmax over per-class logits yields the probability vector; the
    argmax label and its probability are returned as the prediction.
    """

    def __init__(
        self,
        backend: str = "simulated",
        api_key: Optional[str] = None,
        labels: Sequence[str] = LABELS_3,
        seed: int = 42,
    ):
        if backend not in ("simulated", "http"):
            raise ValueError(f"backend must be 'simulated' or 'http'; got {backend!r}")
        self.backend = backend
        self.api_key = api_key
        self.labels = list(labels)
        self.seed = int(seed)
        set_seed(self.seed)
        self.logger = get_logger("nepse_nlp.jev_benchmark.JevClassifier")

    # ----- Simulated backend ---------------------------------------------
    def _simulated_score(self, text: str) -> Dict:
        """Deterministic pseudo-confidence from text features.

        Positive evidence boosts the positive logit; negative evidence
        boosts the negative logit; a balanced, longer text boosts neutral
        slightly. Code-mix ratio introduces a mild shift toward English-
        style positive bias. A softmax over the three logits yields the
        probability vector.

        For the 5-class label set the 3-class probabilities are split onto
        the ordinal extremes (very_negative + negative, positive +
        very_positive) so that the argmax can land on the outer classes
        when evidence is strong.
        """
        if not isinstance(text, str) or not text:
            text = ""
        counts = _lexicon_counts(text)
        pos = counts["pos"]
        neg = counts["neg"]
        cm = _code_mix_ratio(text)
        L = _length(text)

        logit_neg = 0.6 * neg + 0.1 * (1.0 - cm)
        logit_neu = 0.3 - 0.05 * abs(pos - neg) + 0.002 * L
        logit_pos = 0.6 * pos + 0.1 * cm

        logits = [logit_neg, logit_neu, logit_pos]
        probs = _softmax(logits)

        if len(self.labels) == 5:
            five = np.zeros(5, dtype=np.float64)
            five[0] = probs[0] * 0.45  # very_negative
            five[1] = probs[0] * 0.55  # negative
            five[2] = probs[1]  # neutral
            five[3] = probs[2] * 0.55  # positive
            five[4] = probs[2] * 0.45  # very_positive
            s = float(five.sum())
            if s > 0:
                five = five / s
            probs = list(five)

        idx = int(np.argmax(probs))
        label = self.labels[idx]
        confidence = float(probs[idx])
        return {
            "label": label,
            "confidence": confidence,
            "probs": {self.labels[i]: float(p) for i, p in enumerate(probs)},
        }

    # ----- HTTP backend (documented, not exercised offline) --------------
    def _http_score(self, text: str) -> Dict:
        """Call the real Jev API at ``https://api.jev.ai/v1/decide``.

        NOT implemented in this offline build. A production integration
        would POST JSON ``{"text": text, "labels": self.labels}`` to the
        endpoint with ``Authorization: Bearer <api_key>`` and parse the
        returned label probabilities.

        Raises
        ------
        RuntimeError
            If no ``api_key`` was supplied, or always (this offline build
            does not exercise the network path).
        """
        if not self.api_key:
            raise RuntimeError(
                "Jev http backend requires an api_key. Set api_key=... "
                "or use backend='simulated' for offline reproduction."
            )
        raise RuntimeError(
            "Jev http backend is not implemented in this offline build. "
            "Set backend='simulated' for reproducible runs."
        )

    # ----- Public API ----------------------------------------------------
    def classify(self, text: str) -> Dict:
        """Classify a single text.

        Returns
        -------
        dict with keys:
            ``label``      : str
            ``confidence``  : float in [0, 1]
            ``probs``       : dict[str, float]
            ``backend``     : "simulated" | "http"
            ``latency_ms``  : float
        """
        t0 = time.perf_counter()
        if self.backend == "http":
            result = self._http_score(text)
        else:
            result = self._simulated_score(text)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if self.backend == "simulated":
            # Simulated fast latency floor so cost modelling in the hybrid
            # pipeline is realistic. Real Jev runs in ~1-5ms on commodity
            # hardware.
            latency_ms = max(latency_ms, 0.8 + 0.01 * _length(text))
        result["backend"] = self.backend
        result["latency_ms"] = float(latency_ms)
        return result

    def classify_batch(self, texts: List[str]) -> List[Dict]:
        """Classify a batch of texts sequentially."""
        return [self.classify(t) for t in texts]


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def benchmark(
    y_true: List[str], y_pred: List[str], y_conf: List[float], y_probs: List[Dict]
) -> Dict:
    """Compute a battery of classification + calibration metrics.

    Parameters
    ----------
    y_true : list[str]
    y_pred : list[str]
    y_conf : list[float]
        Confidence of each predicted label.
    y_probs : list[dict[str, float]]
        Per-sample label probability vectors.

    Returns
    -------
    dict with keys:
        ``n``, ``labels``, ``accuracy``,
        ``precision_macro``, ``recall_macro``, ``f1_macro``,
        ``precision_micro``, ``recall_micro``, ``f1_micro``,
        ``per_label`` (dict of per-label P/R/F1 + counts),
        ``confusion_matrix`` (list of lists),
        ``brier`` (float | None),
        ``ece``   (float | None),
        ``reliability`` (dict | None),
        ``sklearn_macro`` (optional cross-check, only if sklearn importable).
    """
    n = len(y_true)
    if not (len(y_pred) == len(y_conf) == len(y_probs) == n):
        raise ValueError("y_true, y_pred, y_conf, y_probs must have equal length.")

    labels = sorted(set(y_true) | set(y_pred))
    label2idx = {label: i for i, label in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t in label2idx and p in label2idx:
            cm[label2idx[t], label2idx[p]] += 1

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n if n else 0.0

    per_label: Dict[str, Dict] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    tp_total = fp_total = fn_total = 0
    for i, lab in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - cm[i, i])
        fn = int(cm[i, :].sum() - cm[i, i])
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_label[lab] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        tp_total += tp
        fp_total += fp
        fn_total += fn
    macro_p = float(np.mean(precisions)) if precisions else 0.0
    macro_r = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0

    out: Dict = {
        "n": n,
        "labels": labels,
        "accuracy": accuracy,
        "precision_macro": macro_p,
        "recall_macro": macro_r,
        "f1_macro": macro_f1,
        "precision_micro": micro_p,
        "recall_micro": micro_r,
        "f1_micro": micro_f1,
        "per_label": per_label,
        "confusion_matrix": cm.tolist(),
    }

    if _HAS_CALIBRATION:
        try:
            out["brier"] = brier_score(y_true, y_probs, labels)
            out["ece"] = expected_calibration_error(y_true, y_pred, y_conf, n_bins=10)
            y_correct = [1 if t == p else 0 for t, p in zip(y_true, y_pred)]
            out["reliability"] = reliability_data(y_conf, y_correct, n_bins=10)
        except Exception as exc:  # pragma: no cover - defensive
            out["brier"] = None
            out["ece"] = None
            out["reliability"] = None
            out["calibration_error"] = str(exc)
    else:  # pragma: no cover
        out["brier"] = None
        out["ece"] = None
        out["reliability"] = None

    # Optional sklearn cross-check (only if available).
    try:
        from sklearn.metrics import (  # type: ignore
            precision_recall_fscore_support,
        )

        sk_p, sk_r, sk_f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )
        out["sklearn_macro"] = {
            "precision": float(sk_p),
            "recall": float(sk_r),
            "f1": float(sk_f),
        }
    except Exception:
        pass

    return out


# ---------------------------------------------------------------------------
# Stratified analysis
# ---------------------------------------------------------------------------
def stratified_analysis(records: List[Dict], by: str = "code_mix_bin") -> Dict:
    """Group records by a stratification key and compute per-group metrics.

    Parameters
    ----------
    records : list[dict]
        Each record must include at least ``y_true``, ``y_pred``,
        ``y_conf``, ``y_probs``. The grouping key (``code_mix_ratio``,
        ``sector``, ``length``) is read from each record.
    by : str
        One of ``"code_mix_bin"``, ``"sector"``, ``"length_bin"``.

    Returns
    -------
    dict[str, dict]  — keyed by group label, with ``n``, ``accuracy``,
    ``f1_macro``, ``ece``, ``brier``.
    """
    if by not in ("code_mix_bin", "sector", "length_bin"):
        raise ValueError(
            f"Unsupported stratification key {by!r}; "
            "expected 'code_mix_bin', 'sector', or 'length_bin'."
        )

    def key_of(rec: Dict) -> str:
        if by == "code_mix_bin":
            cm = float(rec.get("code_mix_ratio", 0.0))
            if cm < 0.1:
                return "low(<0.1)"
            if cm <= 0.5:
                return "mid(0.1-0.5)"
            return "high(>0.5)"
        if by == "sector":
            return str(rec.get("sector", "unknown"))
        L = int(rec.get("length", 0))
        if L < 8:
            return "short(<8)"
        if L <= 15:
            return "mid(8-15)"
        return "long(>15)"

    groups: Dict[str, List[Dict]] = {}
    for rec in records:
        groups.setdefault(key_of(rec), []).append(rec)

    out: Dict[str, Dict] = {}
    for g, recs in groups.items():
        yt = [r["y_true"] for r in recs]
        yp = [r["y_pred"] for r in recs]
        yc = [r["y_conf"] for r in recs]
        ypr = [r["y_probs"] for r in recs]
        m = benchmark(yt, yp, yc, ypr)
        out[g] = {
            "n": len(recs),
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "ece": m.get("ece"),
            "brier": m.get("brier"),
        }
    return out
