"""Confidence-gated hybrid pipeline (Jev fast -> slower escalation model).

A System-One fast model (:class:`~nepse_nlp.jev_benchmark.JevClassifier`)
handles high-confidence cases. Low-confidence cases, or cases flagged by
an optional conformal gate, are escalated to a slower System-Two model
(default: a heavier :class:`~nepse_nlp.baselines.ZeroShotBaseline`).

The pipeline tracks cost and latency per request and exposes a Pareto
frontier over the confidence threshold, allowing the user to pick the
operating point on the cost-accuracy trade-off.

MIT License

Copyright (c) 2025 Nepali Sentiment NEPSE Research Project
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = ["HybridPipeline"]

# ---------------------------------------------------------------------------
# Guarded sibling imports
# ---------------------------------------------------------------------------
try:  # pragma: no cover
    from nepse_nlp.utils import set_seed, get_logger  # type: ignore
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
    from nepse_nlp.jev_benchmark import JevClassifier, benchmark as _bench  # type: ignore
    _HAS_JEV = True
except Exception:  # pragma: no cover
    _HAS_JEV = False
    JevClassifier = None  # type: ignore

try:  # pragma: no cover
    from nepse_nlp.baselines import ZeroShotBaseline  # type: ignore
    _HAS_BASELINES = True
except Exception:  # pragma: no cover
    _HAS_BASELINES = False
    ZeroShotBaseline = None  # type: ignore

try:  # pragma: no cover
    from nepse_nlp.conformal import ConformalGate  # type: ignore
    _HAS_CONFORMAL = True
except Exception:  # pragma: no cover
    _HAS_CONFORMAL = False
    ConformalGate = None  # type: ignore


# ---------------------------------------------------------------------------
# Default threshold sweep grid
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS: List[float] = [
    round(0.30 + 0.05 * i, 2) for i in range(14)  # 0.30, 0.35, ..., 0.95
]


class HybridPipeline:
    """Confidence-gated hybrid pipeline.

    Parameters
    ----------
    fast_model
        A classifier exposing ``classify(text) -> dict`` with at least the
        keys ``label``, ``confidence``, ``probs``. Typically a
        :class:`~nepse_nlp.jev_benchmark.JevClassifier`.
    escalation_model
        Same interface; defaults to a heavier
        :class:`~nepse_nlp.baselines.ZeroShotBaseline` (xlm-roberta-large,
        simulated) if available.
    threshold : float
        Minimum fast-model confidence below which escalation is forced.
    cost_fast, cost_slow : float
        USD per call for the fast and slow models, used for cost accounting.
    """

    def __init__(self,
                 fast_model: Any,
                 escalation_model: Any = None,
                 threshold: float = 0.7,
                 cost_fast: float = 0.0002,
                 cost_slow: float = 0.005):
        if fast_model is None:
            raise ValueError("fast_model must be provided")
        self.fast_model = fast_model
        if escalation_model is None:
            if _HAS_BASELINES and ZeroShotBaseline is not None:
                escalation_model = ZeroShotBaseline(
                    model_name="xlm-roberta-large",
                    labels=getattr(fast_model, "labels",
                                    ["negative", "neutral", "positive"]),
                    backend="simulated",
                )
            else:
                raise RuntimeError(
                    "No escalation_model provided and the baselines "
                    "module is unavailable; pass an explicit "
                    "escalation_model."
                )
        self.escalation_model = escalation_model
        self.threshold = float(threshold)
        self.cost_fast = float(cost_fast)
        self.cost_slow = float(cost_slow)
        self.gate: Optional["ConformalGate"] = None
        self.logger = get_logger("nepse_nlp.hybrid_pipeline")

    # ----- Conformal gate ------------------------------------------------
    def set_conformal_gate(self,
                           val_scores: "numpy.ndarray",
                           alpha: float = 0.1) -> "HybridPipeline":
        """Calibrate a conformal escalation gate on validation scores.

        Convention: ``val_scores`` are fast-model *confidences* (higher =
        better). The gate's threshold becomes the ``(1-alpha)`` quantile
        of those confidences; at test time a fast confidence below the
        threshold escalates. This effectively makes the conformal gate a
        learned confidence threshold calibrated on the held-out split.

        Returns
        -------
        HybridPipeline
            ``self`` for fluent chaining.
        """
        if not _HAS_CONFORMAL or ConformalGate is None:
            raise RuntimeError(
                "conformal module unavailable; cannot set a gate."
            )
        self.gate = ConformalGate(alpha=alpha)
        self.gate.fit(val_scores)
        self.logger.info("Conformal gate fit; threshold=%.4f",
                         self.gate.threshold())
        return self

    # ----- Prediction ----------------------------------------------------
    def predict(self, text: str) -> Dict:
        """Run a single prediction through the hybrid pipeline.

        Returns
        -------
        dict with keys:
            ``label``           : str
            ``confidence``       : float in [0, 1]
            ``probs``            : dict[str, float] | None
            ``source``           : "fast" | "escalated"
            ``cost``            : float (USD)
            ``latency_ms``       : float
            ``fast_result``      : dict (always present)
            ``escalation_result``: dict (only when escalated)
        """
        t0 = time.perf_counter()
        fast = self.fast_model.classify(text)
        conf = float(fast["confidence"])
        escalate = conf < self.threshold
        if self.gate is not None:
            # gate.should_escalate uses the same "higher = better"
            # convention as the simple threshold.
            try:
                if self.gate.should_escalate(conf):
                    escalate = True
            except RuntimeError:
                # Gate not fit yet; fall through to the simple threshold.
                pass

        if escalate:
            esc = self.escalation_model.classify(text)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "label": esc["label"],
                "confidence": esc["confidence"],
                "probs": esc.get("probs"),
                "source": "escalated",
                "cost": self.cost_fast + self.cost_slow,
                "latency_ms": latency_ms,
                "fast_result": fast,
                "escalation_result": esc,
            }
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "label": fast["label"],
            "confidence": fast["confidence"],
            "probs": fast.get("probs"),
            "source": "fast",
            "cost": self.cost_fast,
            "latency_ms": latency_ms,
        }

    # ----- Aggregate evaluation at the current threshold ---------------
    def _aggregate(self, records: List[Dict]) -> Dict:
        """Compute aggregate metrics at the *current* threshold."""
        y_true: List[str] = []
        y_pred: List[str] = []
        y_conf: List[float] = []
        y_probs: List[Dict] = []
        n_esc = 0
        total_cost = 0.0
        total_latency = 0.0
        for r in records:
            res = self.predict(r.get("text", ""))
            y_true.append(r["y_true"])
            y_pred.append(res["label"])
            y_conf.append(res["confidence"])
            y_probs.append(res.get("probs") or {})
            if res["source"] == "escalated":
                n_esc += 1
            total_cost += res["cost"]
            total_latency += res["latency_ms"]
        n = len(records) or 1
        metrics: Dict[str, Any] = {
            "n": len(records),
            "threshold": float(self.threshold),
            "accuracy": (sum(1 for t, p in zip(y_true, y_pred) if t == p)
                         / n) if n else 0.0,
            "escalation_rate": n_esc / n,
            "avg_cost_per_1k": (total_cost / n) * 1000.0,
            "avg_latency_ms": total_latency / n,
        }
        if _HAS_JEV and _bench is not None:
            try:
                m = _bench(y_true, y_pred, y_conf, y_probs)
                metrics["f1_macro"] = m["f1_macro"]
                metrics["ece"] = m.get("ece")
                metrics["brier"] = m.get("brier")
            except Exception as exc:  # pragma: no cover
                metrics["benchmark_error"] = str(exc)
        return metrics

    # ----- Public evaluate (with sweep) --------------------------------
    def evaluate(self, records: List[Dict]) -> Dict:
        """Evaluate the pipeline on records.

        Returns the aggregate metrics at the *current* threshold, plus a
        ``sweep`` key holding cost/accuracy points for every threshold in
        ``[0.30, 0.35, ..., 0.95]`` (the cost-accuracy trade-off curve).

        Returns
        -------
        dict with aggregate keys (``accuracy``, ``escalation_rate``,
        ``avg_cost_per_1k``, ``avg_latency_ms``, ``f1_macro``, ``ece``,
        ``brier``) and a ``sweep`` list of
        ``{threshold, accuracy, cost_per_1k, escalation_rate}`` points.
        """
        metrics = self._aggregate(records)
        # Sweep the threshold to populate the cost-accuracy trade-off.
        original = self.threshold
        sweep: List[Dict] = []
        for th in DEFAULT_THRESHOLDS:
            self.threshold = float(th)
            ev = self._aggregate(records)
            sweep.append({
                "threshold": float(th),
                "accuracy": ev["accuracy"],
                "cost_per_1k": ev["avg_cost_per_1k"],
                "escalation_rate": ev["escalation_rate"],
            })
        self.threshold = original
        metrics["sweep"] = sweep
        return metrics

    # ----- Pareto frontier ---------------------------------------------
    def pareto_frontier(self,
                        records: List[Dict],
                        thresholds: Optional[List[float]] = None
                        ) -> List[Dict]:
        """Sweep confidence thresholds and return the Pareto-dominated filter.

        Parameters
        ----------
        records : list[dict]
        thresholds : list[float] | None
            Optional override (default: ``[0.30, 0.35, ..., 0.95]``).

        Returns
        -------
        list[dict] of ``{threshold, accuracy, cost_per_1k,
        escalation_rate}`` sorted by threshold ascending. A point is
        removed if another point has at least as high accuracy and at
        most as high cost, with at least one strict improvement.
        """
        if thresholds is None:
            thresholds = list(DEFAULT_THRESHOLDS)
        original = self.threshold
        points: List[Dict] = []
        for th in thresholds:
            self.threshold = float(th)
            ev = self._aggregate(records)
            points.append({
                "threshold": float(th),
                "accuracy": ev["accuracy"],
                "cost_per_1k": ev["avg_cost_per_1k"],
                "escalation_rate": ev["escalation_rate"],
            })
        self.threshold = original

        # Pareto filter (maximise accuracy, minimise cost).
        pareto: List[Dict] = []
        for p in points:
            dominated = False
            for q in points:
                if q is p:
                    continue
                if (q["accuracy"] >= p["accuracy"]
                        and q["cost_per_1k"] <= p["cost_per_1k"]
                        and (q["accuracy"] > p["accuracy"]
                             or q["cost_per_1k"] < p["cost_per_1k"])):
                    dominated = True
                    break
            if not dominated:
                pareto.append(p)
        pareto.sort(key=lambda x: x["threshold"])
        return pareto
