"""Calibration metrics for Nepali sentiment classifiers (numpy only).

This module is dependency-light: it relies only on numpy so that the
calibration tooling can be exercised in CI environments without torch,
transformers, or scikit-learn. It is imported by ``jev_benchmark`` and the
hybrid pipeline.

Implemented:
    - ``brier_score``         : multiclass Brier score (one-hot vs predicted).
    - ``brier_multiclass``   : alias of ``brier_score`` (API parity).
    - ``expected_calibration_error`` : standard ECE over uniform bins.
    - ``reliability_data``    : binned (acc, conf, count) for reliability plots.
    - ``ordinal_bias``        : 3-class vs 5-class agreement diagnostic.

MIT License

Copyright (c) 2025 Nepali Sentiment NEPSE Research Project
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

__all__ = [
    "brier_score",
    "brier_multiclass",
    "expected_calibration_error",
    "reliability_data",
    "ordinal_bias",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _label_to_index(labels: Sequence[str]) -> Dict[str, int]:
    """Return a stable label->index map."""
    return {l: i for i, l in enumerate(labels)}


def _one_hot(idx: int, n: int) -> np.ndarray:
    """Return a one-hot float vector of length ``n``."""
    v = np.zeros(n, dtype=np.float64)
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


def _bin_mask(conf: np.ndarray, lo: float, hi: float, last: bool) -> np.ndarray:
    """Inclusive-on-the-right bin mask."""
    if last:
        return (conf >= lo) & (conf <= hi)
    return (conf >= lo) & (conf < hi)


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------
def brier_score(y_true: List[str],
                y_probs: List[Dict[str, float]],
                labels: List[str]) -> float:
    """Multiclass Brier score.

    For each sample ``i`` with true label ``y_i`` and predicted probabilities
    ``p_i`` over the same label set:

        brier_i = sum_k (p_i[k] - 1{y_i == k})^2

    The final score is the mean over samples. A perfect deterministic
    classifier scores 0. The maximum for a 3-class problem (all wrong, all
    probability on the wrong label) is 2.

    Parameters
    ----------
    y_true : list[str]
        Ground-truth labels.
    y_probs : list[dict[str, float]]
        Per-sample predicted probabilities keyed by label. Missing labels
        are treated as 0.0.
    labels : list[str]
        The full label set (defines the dimension of the one-hot vector).

    Returns
    -------
    float
        Mean Brier score over the samples.
    """
    if not y_true:
        return 0.0
    if len(y_true) != len(y_probs):
        raise ValueError(
            f"y_true ({len(y_true)}) and y_probs ({len(y_probs)}) must align."
        )
    label2idx = _label_to_index(labels)
    n = len(labels)
    total = 0.0
    for yt, probs in zip(y_true, y_probs):
        if yt not in label2idx:
            # Skip unknown true labels rather than crash.
            continue
        v = _one_hot(label2idx[yt], n)
        p = np.array([probs.get(l, 0.0) for l in labels], dtype=np.float64)
        total += float(np.sum((p - v) ** 2))
    return total / len(y_true)


def brier_multiclass(y_true: List[str],
                     y_probs: List[Dict[str, float]],
                     labels: List[str]) -> float:
    """Alias of :func:`brier_score` kept for API parity."""
    return brier_score(y_true, y_probs, labels)


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------
def expected_calibration_error(y_true: List[str],
                                y_pred: List[str],
                                y_conf: List[float],
                                n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE).

    Bins predictions by confidence into ``n_bins`` equal-width bins on
    [0, 1]. For each bin with at least one sample:

        ECE += (n_b / N) * |acc(b) - avg_conf(b)|

    Returns a float in [0, 1].
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    y_conf = list(y_conf)
    n = len(y_true)
    if n == 0:
        return 0.0
    if not (len(y_pred) == len(y_conf) == n):
        raise ValueError("y_true, y_pred, y_conf must have equal length.")
    correct = np.array([1.0 if t == p else 0.0 for t, p in zip(y_true, y_pred)],
                       dtype=np.float64)
    conf = np.asarray(y_conf, dtype=np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = _bin_mask(conf, bins[i], bins[i + 1], last=(i == n_bins - 1))
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        acc_b = float(correct[mask].mean())
        conf_b = float(conf[mask].mean())
        ece += (cnt / n) * abs(acc_b - conf_b)
    return float(ece)


# ---------------------------------------------------------------------------
# Reliability data (for reliability diagrams)
# ---------------------------------------------------------------------------
def reliability_data(y_conf: List[float],
                     y_correct: List[int],
                     n_bins: int = 10) -> Dict[str, List[float]]:
    """Return binned reliability-diagram data.

    Parameters
    ----------
    y_conf : list[float]
        Confidence of each prediction in [0, 1].
    y_correct : list[int]
        1 if the prediction was correct, 0 otherwise.
    n_bins : int
        Number of equal-width bins on [0, 1].

    Returns
    -------
    dict with keys:
        - ``bin_centers`` : list[float] of length n_bins (midpoint of each bin)
        - ``accs``        : list[float] of length n_bins (per-bin accuracy,
          NaN if empty)
        - ``confs``       : list[float] of length n_bins (per-bin avg confidence,
          NaN if empty)
        - ``counts``      : list[int] of length n_bins (per-bin sample counts)
    """
    conf = np.asarray(y_conf, dtype=np.float64)
    correct = np.asarray(y_correct, dtype=np.float64)
    if conf.size != correct.size:
        raise ValueError("y_conf and y_correct must have equal length.")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers: List[float] = []
    accs: List[float] = []
    confs: List[float] = []
    counts: List[int] = []
    for i in range(n_bins):
        mask = _bin_mask(conf, bins[i], bins[i + 1], last=(i == n_bins - 1))
        cnt = int(mask.sum())
        centers.append(float((bins[i] + bins[i + 1]) / 2.0))
        if cnt == 0:
            accs.append(float("nan"))
            confs.append(float("nan"))
        else:
            accs.append(float(correct[mask].mean()))
            confs.append(float(conf[mask].mean()))
        counts.append(cnt)
    return {"bin_centers": centers, "accs": accs,
            "confs": confs, "counts": counts}


# ---------------------------------------------------------------------------
# Ordinal bias (3-class vs 5-class agreement)
# ---------------------------------------------------------------------------
# Mapping from the 5-class ordinal scale back to the 3-class scale.
_FIVE_TO_THREE = {
    "very_negative": "negative",
    "negative": "negative",
    "neutral": "neutral",
    "positive": "positive",
    "very_positive": "positive",
}
_THREE_INT = {"negative": 0, "neutral": 1, "positive": 2}


def ordinal_bias(y_true: List[str],
                 y_pred_3: List[str],
                 y_pred_5: List[str]) -> Dict[str, float]:
    """Compare 3-class predictions vs 5-class predictions collapsed to 3.

    Useful for diagnosing whether the 5-class ordinal scale introduces a
    systematic shift relative to the coarser 3-class scale (an "ordinal
    bias").

    Parameters
    ----------
    y_true : list[str]
        Ground-truth labels (any of the 3-class or 5-class names).
    y_pred_3 : list[str]
        3-class predictions, e.g. ``["negative","neutral","positive"]``.
    y_pred_5 : list[str]
        5-class predictions, e.g. ``["very_negative", ..., "very_positive"]``.

    Returns
    -------
    dict with keys:
        - ``agreement``     : fraction where the collapsed 5-class prediction
          equals the 3-class prediction.
        - ``disagreement``  : 1 - ``agreement``.
        - ``mean_distance`` : mean absolute distance (in 3-class ordinal
          units 0/1/2) over the disagreement subset.
        - ``n``             : number of samples compared.
    """
    if len(y_pred_3) != len(y_pred_5):
        raise ValueError("y_pred_3 and y_pred_5 must have equal length.")
    n = len(y_pred_3)
    if n == 0:
        return {"agreement": 0.0, "disagreement": 0.0,
                "mean_distance": 0.0, "n": 0}
    agree = 0
    dist_sum = 0.0
    for p3, p5 in zip(y_pred_3, y_pred_5):
        collapse = _FIVE_TO_THREE.get(p5, "neutral")
        if collapse == p3:
            agree += 1
        else:
            dist_sum += abs(_THREE_INT.get(p3, 1) - _THREE_INT.get(collapse, 1))
    agreement = agree / n
    return {
        "agreement": agreement,
        "disagreement": 1.0 - agreement,
        "mean_distance": dist_sum / n,
        "n": n,
    }
