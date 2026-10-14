#!/usr/bin/env python3
"""Phase 2 benchmark CLI: run Jev + baselines on simulated headlines.

Loads simulated Nepali financial headlines (from
``nepse_nlp.scraper.simulated_headlines``), runs the JevClassifier and
zero-shot baselines (``xlm-roberta-large`` and ``gpt-4o-simulated``),
computes ``benchmark()`` metrics for Jev, calls :func:`run_baselines`
for the baselines, runs stratified analysis by code-mix bin and sector,
prints a markdown table, and saves all results to
``results/phase2_benchmark.json``.

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


from nepse_nlp.baselines import run_baselines  # noqa: E402
from nepse_nlp.jev_benchmark import (  # noqa: E402
    JevClassifier,
    benchmark,
    stratified_analysis,
)

# Optional: prefer the scraper's simulated headlines; fall back to a small
# local set if the scraper module is not yet available.
try:  # pragma: no cover - depends on scraper module presence
    from nepse_nlp.scraper import simulated_headlines  # type: ignore

    _HAS_SCRAPER = True
except Exception:  # pragma: no cover
    _HAS_SCRAPER = False


# ---------------------------------------------------------------------------
# Fallback simulated headlines (used if scraper.simulated_headlines
# is not yet available). Eight short Nepali + code-mixed headlines with
# known ground-truth sentiment covering 3 classes, 3 code-mix bins, and
# 4 sectors.
# ---------------------------------------------------------------------------
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
    """Return a normalised list of headline records."""
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
        # Field-name compatibility shim: the scraper's simulated headlines
        # use ``title``/``summary``/``true_label``, other generators use
        # ``text``/``headline``/``y_true``/``label``.
        text = r.get("text") or r.get("headline") or r.get("title") or r.get("summary") or ""
        # If a separate summary exists, append it so the classifier sees
        # both the headline and the summary body.
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

    # ----- Jev ---------------------------------------------------------
    jev = JevClassifier(backend="simulated", seed=42)
    y_true, y_pred, y_conf, y_probs = [], [], [], []
    for r in records:
        res = jev.classify(r["text"])
        y_true.append(r["y_true"])
        y_pred.append(res["label"])
        y_conf.append(res["confidence"])
        y_probs.append(res["probs"])
    jev_metrics = benchmark(y_true, y_pred, y_conf, y_probs)

    # ----- Baselines ---------------------------------------------------
    baseline_names = ["xlm-roberta-large", "gpt-4o-simulated"]
    baseline_metrics = run_baselines(records, baseline_names)

    # ----- Stratified analysis ----------------------------------------
    strat_records = [
        {**r, "y_pred": y_pred[i], "y_conf": y_conf[i], "y_probs": y_probs[i]}
        for i, r in enumerate(records)
    ]
    strat_cm = stratified_analysis(strat_records, by="code_mix_bin")
    strat_sec = stratified_analysis(strat_records, by="sector")

    # ----- Save results -----------------------------------------------
    results = {
        "phase": 2,
        "n_records": len(records),
        "jev": {
            "accuracy": jev_metrics["accuracy"],
            "f1_macro": jev_metrics["f1_macro"],
            "precision_macro": jev_metrics["precision_macro"],
            "recall_macro": jev_metrics["recall_macro"],
            "f1_micro": jev_metrics["f1_micro"],
            "ece": jev_metrics.get("ece"),
            "brier": jev_metrics.get("brier"),
            "labels": jev_metrics["labels"],
            "confusion_matrix": jev_metrics["confusion_matrix"],
        },
        "baselines": baseline_metrics,
        "stratified_code_mix": strat_cm,
        "stratified_sector": strat_sec,
    }
    results_dir = os.path.normpath(os.path.join(_HERE, "..", "results"))
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "phase2_benchmark.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ----- Markdown table ---------------------------------------------
    print("# Phase 2 - Jev Zero-Shot Benchmark\n")
    print(f"Records: {len(records)} (simulated Nepali financial headlines)\n")

    rows = [
        (
            "jev (simulated)",
            results["jev"]["accuracy"],
            results["jev"]["f1_macro"],
            results["jev"]["ece"],
            results["jev"]["brier"],
        )
    ]
    for name, m in baseline_metrics.items():
        rows.append((name, m["accuracy"], m["f1_macro"], m["ece"], m["brier"]))
    print("| Model | Accuracy | F1 (macro) | ECE | Brier |")
    print("|---|---|---|---|---|")
    for r in rows:
        ece = "n/a" if r[3] is None else f"{r[3]:.4f}"
        brier = "n/a" if r[4] is None else f"{r[4]:.4f}"
        print(f"| {r[0]} | {r[1]:.4f} | {r[2]:.4f} | {ece} | {brier} |")

    # ----- Stratified summary -----------------------------------------
    print("\n## Stratified by code-mix bin (Jev)\n")
    print("| Bin | n | Accuracy | F1 (macro) | ECE | Brier |")
    print("|---|---|---|---|---|---|")
    for k, m in strat_cm.items():
        ece = "n/a" if m["ece"] is None else f"{m['ece']:.4f}"
        brier = "n/a" if m["brier"] is None else f"{m['brier']:.4f}"
        print(f"| {k} | {m['n']} | {m['accuracy']:.4f} | {m['f1_macro']:.4f} | {ece} | {brier} |")

    print("\n## Stratified by sector (Jev)\n")
    print("| Sector | n | Accuracy | F1 (macro) | ECE | Brier |")
    print("|---|---|---|---|---|---|")
    for k, m in strat_sec.items():
        ece = "n/a" if m["ece"] is None else f"{m['ece']:.4f}"
        brier = "n/a" if m["brier"] is None else f"{m['brier']:.4f}"
        print(f"| {k} | {m['n']} | {m['accuracy']:.4f} | {m['f1_macro']:.4f} | {ece} | {brier} |")

    print(f"\nResults written to: {out_path}")


if __name__ == "__main__":
    main()
