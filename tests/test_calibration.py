"""Tests for calibration, conformal, and hybrid escalation.

Numpy-only by design; the tests do not require torch, transformers,
sklearn, or network access. They cover:

    * Brier score perfect/worst bounds.
    * ECE in [0, 1].
    * Reliability-data key set and equal-length lists.
    * Conformal threshold is a valid score quantile and the escalation
      logic is consistent with that threshold.
    * Hybrid pipeline: a low-confidence input triggers escalation while a
      high-confidence input does not.

MIT License

Copyright (c) 2025 Nepali Sentiment NEPSE Research Project
"""

import os
import sys

# Put `src/` on sys.path so `nepse_nlp` is importable when running these
# tests directly via ``python -m pytest tests/test_calibration.py`` from
# the repo root, without requiring an editable install.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np  # noqa: E402

from nepse_nlp.calibration import (  # noqa: E402
    brier_score,
    expected_calibration_error,
    reliability_data,
)
from nepse_nlp.conformal import ConformalGate, calibrate_conformal  # noqa: E402

LABELS = ["negative", "neutral", "positive"]


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------
def test_brier_perfect_zero():
    """Perfect deterministic predictions -> Brier ~0."""
    y_true = ["positive", "negative", "neutral"]
    y_probs = [
        {"negative": 0.0, "neutral": 0.0, "positive": 1.0},
        {"negative": 1.0, "neutral": 0.0, "positive": 0.0},
        {"negative": 0.0, "neutral": 1.0, "positive": 0.0},
    ]
    score = brier_score(y_true, y_probs, LABELS)
    assert abs(score) < 1e-9, f"expected ~0, got {score}"


def test_brier_worst():
    """All-wrong max-probability predictions -> ~2 for 3-class."""
    y_true = ["positive", "negative", "neutral"]
    y_probs = [
        {"negative": 1.0, "neutral": 0.0, "positive": 0.0},  # true=pos
        {"negative": 0.0, "neutral": 1.0, "positive": 0.0},  # true=neg
        {"negative": 1.0, "neutral": 0.0, "positive": 0.0},  # true=neu
    ]
    score = brier_score(y_true, y_probs, LABELS)
    # Each row contributes (1-0)^2 + (0-1)^2 = 2 -> mean = 2.0
    assert 1.5 < score <= 2.0 + 1e-9, f"expected ~2.0, got {score}"


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------
def test_ece_range():
    """ECE must lie in [0, 1]."""
    y_true = ["positive", "negative", "neutral", "positive", "negative"]
    y_pred = ["positive", "positive", "neutral", "positive", "negative"]
    y_conf = [0.9, 0.6, 0.5, 0.8, 0.7]
    ece = expected_calibration_error(y_true, y_pred, y_conf, n_bins=10)
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0, f"expected 0<=ece<=1, got {ece}"


# ---------------------------------------------------------------------------
# Reliability data
# ---------------------------------------------------------------------------
def test_reliability_data_keys():
    """Reliability data has the four keys, all same length, equal to n_bins."""
    y_conf = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
    y_correct = [0, 1, 1, 0, 1, 1]
    d = reliability_data(y_conf, y_correct, n_bins=10)
    for k in ("bin_centers", "accs", "confs", "counts"):
        assert k in d, f"missing key {k}"
    n = len(d["bin_centers"])
    assert n == 10
    assert len(d["accs"]) == n
    assert len(d["confs"]) == n
    assert len(d["counts"]) == n
    assert sum(d["counts"]) == len(y_conf)


# ---------------------------------------------------------------------------
# Conformal
# ---------------------------------------------------------------------------
def test_conformal_threshold():
    """Threshold is a valid score quantile; should_escalate is consistent."""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    thr = calibrate_conformal(scores, alpha=0.1)
    # The threshold should lie within the data range.
    assert float(scores.min()) <= thr <= float(scores.max()), f"threshold {thr} not in score range"
    # ConformalGate reproduces the threshold and applies the rule
    # "score < threshold -> escalate" consistently.
    gate = ConformalGate(alpha=0.1)
    gate.fit(scores)
    assert abs(gate.threshold() - thr) < 1e-12
    assert gate.should_escalate(thr - 0.05) is True
    assert gate.should_escalate(thr + 0.05) is False


# ---------------------------------------------------------------------------
# Hybrid pipeline escalation
# ---------------------------------------------------------------------------
def test_hybrid_escalation():
    """Low-confidence input triggers escalation; high-confidence doesn't."""
    from nepse_nlp.hybrid_pipeline import HybridPipeline
    from nepse_nlp.jev_benchmark import JevClassifier

    fast = JevClassifier(backend="simulated", seed=42)
    pipe = HybridPipeline(fast_model=fast, threshold=0.7)

    # A neutral text with no lexicon hits -> low confidence -> escalate.
    res_low = pipe.predict("the meeting was held at noon today")
    assert res_low["source"] == "escalated", res_low
    assert "escalation_result" in res_low

    # A clearly positive text with multiple lexicon hits -> high confidence
    # -> stays on the fast path.
    res_high = pipe.predict("profit growth record surge strong high gain rally")
    assert res_high["source"] == "fast", res_high
    assert "escalation_result" not in res_high
