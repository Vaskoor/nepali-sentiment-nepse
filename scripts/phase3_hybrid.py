#!/usr/bin/env python3
"""Phase 3 hybrid pipeline CLI: threshold sweep + Pareto frontier.

Builds a :class:`HybridPipeline` with a simulated Jev fast model and a
simulated XLM-RoBERTa-large escalation model, sweeps confidence
thresholds to produce a Pareto frontier, calibrates a conformal
escalation gate on a held-out validation split, prints a markdown table,
and saves all results to ``results/phase3_pareto.json``.

Offline by design; no network calls.

MIT License

Copyright (c) 2025 Nepali Sentiment NEPSE Research Project
"""

from __future__ import annotations

import json
import os
import sys

# Make `nepse_nlp` importable when running this script directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402

from nepse_nlp.baselines import ZeroShotBaseline  # noqa: E402
from nepse_nlp.hybrid_pipeline import HybridPipeline  # noqa: E402
from nepse_nlp.jev_benchmark import JevClassifier  # noqa: E402

try:  # pragma: no cover - depends on scraper module presence
    from nepse_nlp.scraper import simulated_headlines  # type: ignore

    _HAS_SCRAPER = True
except Exception:  # pragma: no cover
    _HAS_SCRAPER = False


def _fallback_headlines():
    return [
        {
            "text": "नेपाली बैंकले नाफा बढ्यो रेकर्ड सकारात्मक",
            "y_true": "positive",
            "sector": "banking",
            "code_mix_ratio": 0.0,
            "length": 7,
        },
        {
            "text": "नेप्से गिरावट घट्यो चिन्ता",
            "y_true": "negative",
            "sector": "index",
            "code_mix_ratio": 0.0,
            "length": 5,
        },
        {
            "text": "profit surge growth strong record high",
            "y_true": "positive",
            "sector": "index",
            "code_mix_ratio": 1.0,
            "length": 6,
        },
        {
            "text": "loss fall decline weak drop crash worry",
            "y_true": "negative",
            "sector": "index",
            "code_mix_ratio": 1.0,
            "length": 7,
        },
        {
            "text": "बजार स्थिर रह्यो कुनै ठूलो परिवर्तन छैन",
            "y_true": "neutral",
            "sector": "index",
            "code_mix_ratio": 0.0,
            "length": 7,
        },
        {
            "text": "hydropower company shares profit growth",
            "y_true": "positive",
            "sector": "hydropower",
            "code_mix_ratio": 1.0,
            "length": 5,
        },
        {
            "text": "घरजगाना कमजोर नोक्सान चिन्ता",
            "y_true": "negative",
            "sector": "finance",
            "code_mix_ratio": 0.0,
            "length": 4,
        },
        {
            "text": "insurance sector mixed performance today",
            "y_true": "neutral",
            "sector": "insurance",
            "code_mix_ratio": 1.0,
            "length": 5,
        },
    ]


def _load_headlines():
    if _HAS_SCRAPER:
        raw = simulated_headlines
        if callable(raw):
            raw = raw()
        records = list(raw)
    else:
        records = _fallback_headlines()
    norm = []
    for r in records:
        if isinstance(r, str):
            r = {"text": r, "y_true": "neutral"}
        # Field-name compatibility shim (see phase2_benchmark.py).
        text = r.get("text") or r.get("headline") or r.get("title") or r.get("summary") or ""
        summary = r.get("summary")
        if summary and summary != text:
            text = f"{text}. {summary}"
        y_true = r.get("y_true") or r.get("true_label") or r.get("label") or "neutral"
        norm.append(
            {
                "text": text,
                "y_true": y_true,
                "sector": r.get("sector", "unknown"),
                "code_mix_ratio": r.get("code_mix_ratio", 0.0),
                "length": r.get("length", len(text.split())),
            }
        )
    return norm


def main():
    records = _load_headlines()

    # Split: 80% train+test, 20% validation (used only to calibrate the
    # conformal gate). The Pareto sweep is evaluated on the remaining
    # records.
    n = len(records)
    n_val = max(1, n // 5)
    val_records = records[:n_val]
    test_records = records[n_val:]

    fast = JevClassifier(backend="simulated", seed=42)
    esc = ZeroShotBaseline(model_name="xlm-roberta-large", backend="simulated", seed=42)

    pipe = HybridPipeline(
        fast_model=fast, escalation_model=esc, threshold=0.7, cost_fast=0.0002, cost_slow=0.005
    )

    # ----- Pareto frontier (gate-free sweep) ---------------------------
    pareto = pipe.pareto_frontier(test_records)

    # ----- Full sweep for transparency --------------------------------
    full_eval = pipe.evaluate(test_records)
    sweep = full_eval.get("sweep", [])

    # ----- Conformal gate calibration ---------------------------------
    # Use the fast model's confidence on the validation split as the
    # calibration score (higher = better). Threshold = (1-alpha) quantile.
    val_conf = np.array([fast.classify(r["text"])["confidence"] for r in val_records])
    conformal_threshold = None
    try:
        pipe.set_conformal_gate(val_conf, alpha=0.1)
        conformal_threshold = float(pipe.gate.threshold())
    except Exception as exc:  # pragma: no cover
        print(f"(conformal gate calibration skipped: {exc})")

    # ----- Aggregate evaluation at default threshold -----------------
    pipe.threshold = 0.7
    eval_default = pipe._aggregate(test_records)

    # ----- Save results -----------------------------------------------
    results = {
        "phase": 3,
        "n_records": n,
        "n_val": n_val,
        "n_test": len(test_records),
        "cost_fast_usd": 0.0002,
        "cost_slow_usd": 0.005,
        "conformal_alpha": 0.1,
        "conformal_threshold": conformal_threshold,
        "default_threshold": 0.7,
        "default_eval": eval_default,
        "full_sweep": sweep,
        "pareto_frontier": pareto,
    }
    results_dir = os.path.normpath(os.path.join(_HERE, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "phase3_pareto.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ----- Markdown --------------------------------------------------
    print("# Phase 3 - Confidence-Gated Hybrid Pipeline\n")
    print(f"Records: total={n} val={n_val} test={len(test_records)}\n")
    if conformal_threshold is not None:
        print(
            f"Conformal gate threshold (alpha=0.1, on val confidences): {conformal_threshold:.4f}\n"
        )
    print("Default @ threshold=0.7:")
    print(f"  accuracy        = {eval_default['accuracy']:.4f}")
    print(f"  escalation rate = {eval_default['escalation_rate']:.4f}")
    print(f"  avg cost / 1k   = ${eval_default['avg_cost_per_1k']:.4f}")
    print(f"  avg latency ms  = {eval_default['avg_latency_ms']:.2f}\n")

    print("## Pareto frontier (threshold sweep, Pareto-dominated filter)\n")
    print("| Threshold | Accuracy | Cost/1k ($) | Escalation rate |")
    print("|---|---|---|---|")
    for p in pareto:
        print(
            f"| {p['threshold']:.2f} | {p['accuracy']:.4f} | "
            f"{p['cost_per_1k']:.4f} | {p['escalation_rate']:.4f} |"
        )

    print("\n## Full threshold sweep\n")
    print("| Threshold | Accuracy | Cost/1k ($) | Escalation rate |")
    print("|---|---|---|---|")
    for p in sweep:
        print(
            f"| {p['threshold']:.2f} | {p['accuracy']:.4f} | "
            f"{p['cost_per_1k']:.4f} | {p['escalation_rate']:.4f} |"
        )

    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
