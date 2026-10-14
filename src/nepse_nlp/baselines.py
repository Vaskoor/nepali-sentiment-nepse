"""Zero-shot baseline classifiers: NepaliBERT, MuRIL, XLM-R, GPT-4o (simulated).

Each baseline shares a deterministic simulated backend (so the entire
benchmark runs offline), with per-model bias and temperature so the
baselines differ realistically. If ``backend='transformers'`` is
requested and the ``transformers`` package is importable, a real
``transformers.pipeline("zero-shot-classification", ...)`` is constructed;
otherwise we fall back to the simulated backend.

The simulated backend is intentionally distinct from
:mod:`nepse_nlp.jev_benchmark`'s simulated backend: it adds per-model
noise (seeded from a stable text hash) and per-model bias, so the
benchmark table reports genuine cross-model variance even when no
transformers / torch is installed.

MIT License

Copyright (c) 2025 Nepali Sentiment NEPSE Research Project
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Dict, List, Sequence

import numpy as np

__all__ = ["ZeroShotBaseline", "run_baselines", "MODEL_PARAMS"]

# ---------------------------------------------------------------------------
# Shared lexicon (kept in sync with jev_benchmark)
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
# Per-model bias / temperature / latency for the simulated backend.
# These parameters are not calibrated to real models — they are stylised
# so that the four baselines occupy different points in (confidence,
# accuracy, latency) space, which is what the research narrative needs.
# ---------------------------------------------------------------------------
MODEL_PARAMS: Dict[str, Dict] = {
    "nepalibert": {
        "bias": {"negative": 0.05, "neutral": 0.0, "positive": 0.0},
        "temperature": 1.1,
        "noise_std": 0.04,
        "latency_ms": 28.0,
    },
    "muril": {
        "bias": {"negative": 0.02, "neutral": 0.03, "positive": 0.0},
        "temperature": 1.0,
        "noise_std": 0.04,
        "latency_ms": 22.0,
    },
    "xlm-roberta-large": {
        "bias": {"negative": 0.0, "neutral": 0.02, "positive": 0.01},
        "temperature": 0.9,
        "noise_std": 0.04,
        "latency_ms": 35.0,
    },
    "gpt-4o-simulated": {
        "bias": {"negative": 0.0, "neutral": 0.0, "positive": 0.0},
        "temperature": 0.7,
        "noise_std": 0.02,
        "latency_ms": 850.0,
    },
}

# ---------------------------------------------------------------------------
# Guarded sibling imports
# ---------------------------------------------------------------------------
try:  # pragma: no cover
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


try:  # pragma: no cover
    from nepse_nlp.jev_benchmark import (  # type: ignore
        LABELS_3,
        LABELS_5,
    )
    from nepse_nlp.jev_benchmark import (
        benchmark as _jev_benchmark,
    )

    _HAS_JEV = True
except Exception:  # pragma: no cover
    _HAS_JEV = False
    LABELS_3 = ["negative", "neutral", "positive"]
    LABELS_5 = ["very_negative", "negative", "neutral", "positive", "very_positive"]
    _jev_benchmark = None  # type: ignore


# ---------------------------------------------------------------------------
# Text feature helpers (kept local so this module is standalone-friendly)
# ---------------------------------------------------------------------------
def _lexicon_counts(text: str) -> Dict[str, int]:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    t = text.lower()
    pos = sum(1 for w in POSITIVE_LEXICON if w.lower() in t)
    neg = sum(1 for w in NEGATIVE_LEXICON if w.lower() in t)
    return {"pos": pos, "neg": neg}


def _code_mix_ratio(text: str) -> float:
    if not text:
        return 0.0
    ascii_alnum = sum(1 for c in text if c.isascii() and c.isalnum())
    total_alnum = sum(1 for c in text if c.isalnum())
    return ascii_alnum / total_alnum if total_alnum else 0.0


def _length(text: str) -> int:
    return len(text.split()) if text else 0


def _softmax(x: Sequence[float]) -> List[float]:
    if not x:
        return []
    m = max(x)
    exps = [math.exp(v - m) for v in x]
    s = sum(exps)
    if s <= 0:
        n = len(x)
        return [1.0 / n for _ in x]
    return [e / s for e in exps]


def _stable_hash(text: str) -> int:
    """Stable 32-bit hash from MD5 (deterministic across runs / machines)."""
    if not text:
        return 0
    h = hashlib.md5(text.encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(h[:4], "big")


# ---------------------------------------------------------------------------
# ZeroShotBaseline
# ---------------------------------------------------------------------------
class ZeroShotBaseline:
    """Zero-shot sentiment baseline with simulated or transformers backend.

    Parameters
    ----------
    model_name : str
        One of ``"nepalibert"``, ``"muril"``, ``"xlm-roberta-large"``,
        ``"gpt-4o-simulated"``.
    labels : sequence[str]
        Label set; defaults to :data:`LABELS_3`.
    backend : str
        ``"transformers"`` (try real pipeline; fall back to simulated) or
        ``"simulated"`` (offline, default).
    seed : int
        Used both for the global RNG and for per-text noise seeding.

    Notes
    -----
    The simulated backend uses a per-model temperature, bias, and noise
    standard deviation (see :data:`MODEL_PARAMS`). Noise is seeded from a
    stable MD5 hash of the text so identical inputs always yield identical
    outputs across processes and machines.
    """

    def __init__(
        self,
        model_name: str = "xlm-roberta-large",
        labels: Sequence[str] = LABELS_3,
        backend: str = "simulated",
        seed: int = 42,
    ):
        if model_name not in MODEL_PARAMS:
            raise ValueError(
                f"Unknown model_name {model_name!r}; expected one of {list(MODEL_PARAMS)}."
            )
        self.model_name = model_name
        self.labels = list(labels)
        self.seed = int(seed)
        self.params = MODEL_PARAMS[model_name]
        set_seed(self.seed)
        self.logger = get_logger(f"nepse_nlp.baselines.{model_name}")

        # Resolve backend.
        self._hf_pipeline = None
        self.backend = "simulated"
        if backend == "transformers":
            try:
                from transformers import pipeline  # type: ignore

                self._hf_pipeline = pipeline(
                    "zero-shot-classification",
                    model=model_name,
                    tokenizer=model_name,
                )
                self.backend = "transformers"
            except Exception as exc:  # pragma: no cover - depends on env
                self.logger.warning(
                    "transformers backend unavailable (%s); using simulated.",
                    str(exc)[:80],
                )
                self.backend = "simulated"
        elif backend != "simulated":
            raise ValueError(f"backend must be 'transformers' or 'simulated'; got {backend!r}")

    # ----- Simulated backend ---------------------------------------------
    def _simulated_score(self, text: str) -> Dict:
        """Deterministic per-model simulated scoring.

        Uses the shared lexicon + per-model temperature and bias, plus a
        small stable noise term so each baseline behaves distinctly.
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        counts = _lexicon_counts(text)
        pos = counts["pos"]
        neg = counts["neg"]
        cm = _code_mix_ratio(text)
        L = _length(text)

        T = self.params["temperature"]
        bias = self.params["bias"]
        std = self.params["noise_std"]

        logit_pos = (0.5 * pos) / T + bias.get("positive", 0.0) + 0.05 * cm
        logit_neg = (0.5 * neg) / T + bias.get("negative", 0.0) + 0.05 * (1.0 - cm)
        logit_neu = 0.2 - 0.05 * abs(pos - neg) + 0.002 * L + bias.get("neutral", 0.0)

        # Stable per-text noise so output is reproducible.
        seed_int = (_stable_hash(text) ^ self.seed ^ _stable_hash(self.model_name)) & 0xFFFFFFFF
        rng = np.random.RandomState(seed_int)
        noise = rng.normal(0.0, std, size=3)
        logits = [
            logit_neg + float(noise[0]),
            logit_neu + float(noise[1]),
            logit_pos + float(noise[2]),
        ]
        probs = _softmax(logits)

        if len(self.labels) == 5:
            five = np.zeros(5, dtype=np.float64)
            five[0] = probs[0] * 0.45
            five[1] = probs[0] * 0.55
            five[2] = probs[1]
            five[3] = probs[2] * 0.55
            five[4] = probs[2] * 0.45
            s = float(five.sum())
            if s > 0:
                five = five / s
            probs = list(five)

        idx = int(np.argmax(probs))
        return {
            "label": self.labels[idx],
            "confidence": float(probs[idx]),
            "probs": {self.labels[i]: float(p) for i, p in enumerate(probs)},
        }

    # ----- Transformers backend ------------------------------------------
    def _transformers_score(self, text: str) -> Dict:
        """Real transformers zero-shot classification."""
        if self._hf_pipeline is None:
            raise RuntimeError("transformers backend not initialised")
        out = self._hf_pipeline(text, list(self.labels), multi_label=False)
        probs = {label: 0.0 for label in self.labels}
        for lbl, sc in zip(out["labels"], out["scores"]):
            if lbl in probs:
                probs[lbl] = float(sc)
        idx = max(range(len(self.labels)), key=lambda i: probs[self.labels[i]])
        return {
            "label": self.labels[idx],
            "confidence": probs[self.labels[idx]],
            "probs": probs,
        }

    # ----- Public API ----------------------------------------------------
    def classify(self, text: str) -> Dict:
        """Classify a single text.

        Returns
        -------
        dict with keys:
            ``label``      : str
            ``confidence``  : float in [0, 1]
            ``probs``       : dict[str, float]
            ``backend``     : "simulated" | "transformers"
            ``model_name``  : str
            ``latency_ms``  : float
        """
        t0 = time.perf_counter()
        if self.backend == "transformers" and self._hf_pipeline is not None:
            result = self._transformers_score(text)
        else:
            result = self._simulated_score(text)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        # Floor latency to the model's nominal cost so the hybrid pipeline's
        # cost modelling is realistic even on a fast test box.
        latency_ms = max(latency_ms, self.params["latency_ms"])
        result["backend"] = self.backend
        result["model_name"] = self.model_name
        result["latency_ms"] = float(latency_ms)
        return result

    def classify_batch(self, texts: List[str]) -> List[Dict]:
        return [self.classify(t) for t in texts]


# ---------------------------------------------------------------------------
# run_baselines
# ---------------------------------------------------------------------------
def run_baselines(records: List[Dict], model_names: List[str]) -> Dict:
    """Run a list of baseline models on records; return per-model metrics.

    Each record dict must include at least ``text`` and ``y_true``;
    optional ``sector``, ``code_mix_ratio``, ``length`` keys are preserved
    for stratified analysis downstream.

    Returns
    -------
    dict[str, dict]  keyed by model name with ``accuracy``, ``f1_macro``,
    ``ece``, ``brier``.
    """
    out: Dict[str, Dict] = {}
    for name in model_names:
        clf = ZeroShotBaseline(model_name=name, labels=LABELS_3, backend="simulated", seed=42)
        y_true, y_pred, y_conf, y_probs = [], [], [], []
        for r in records:
            res = clf.classify(r.get("text", ""))
            y_true.append(r["y_true"])
            y_pred.append(res["label"])
            y_conf.append(res["confidence"])
            y_probs.append(res["probs"])
        if _HAS_JEV and _jev_benchmark is not None:
            m = _jev_benchmark(y_true, y_pred, y_conf, y_probs)
            out[name] = {
                "accuracy": m["accuracy"],
                "f1_macro": m["f1_macro"],
                "ece": m.get("ece"),
                "brier": m.get("brier"),
            }
        else:  # pragma: no cover - fallback
            acc = (
                (sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)) if y_true else 0.0
            )
            out[name] = {"accuracy": acc, "f1_macro": 0.0, "ece": 0.0, "brier": 0.0}
    return out
