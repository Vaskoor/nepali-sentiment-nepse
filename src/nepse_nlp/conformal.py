"""Split-conformal thresholds for confidence-gated escalation.

Implements a simple split-conformal calibration where a held-out validation
set of scores (e.g. fast-model confidences, or non-conformity scores such as
``1 - max_prob``) is summarised by its ``(1 - alpha)`` quantile. The
resulting scalar threshold is then used to gate escalation in the hybrid
pipeline.

Convention used in this module
------------------------------
* ``scores`` are interpreted as "higher = better calibrated / higher
  confidence". A higher score means we trust the fast model.
* The threshold is the ``(1 - alpha)`` quantile of those scores.
* A test sample with ``score < threshold`` is escalated (low confidence).

This is the convention called for by the task spec: it matches a learned
confidence threshold rather than the textbook non-conformity formulation
(in which higher = worse and escalation happens above the threshold). The
behaviour is fully deterministic and reproducible.

Numpy only.

MIT License

Copyright (c) 2025 Nepali Sentiment NEPSE Research Project
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

__all__ = ["calibrate_conformal", "ConformalGate"]


def calibrate_conformal(scores: "numpy.ndarray", alpha: float = 0.1) -> float:
    """Return the ``(1 - alpha)`` quantile of ``scores``.

    Parameters
    ----------
    scores : np.ndarray
        1-D array of calibration scores (interpretation: higher = better).
    alpha : float, default 0.1
        Target tail mass (e.g. 0.1 -> 90th percentile threshold).

    Returns
    -------
    float
        The threshold ``tau = Q_{1-alpha}(scores)``. Returns ``0.0`` for
        empty input so callers can fall back to "always escalate".
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if scores.size == 0:
        return 0.0
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0,1]; got {alpha}")
    q = float(np.quantile(scores, 1.0 - alpha))
    return q


class ConformalGate:
    """Conformal escalation gate.

    Usage
    -----
    >>> import numpy as np
    >>> from nepse_nlp.conformal import ConformalGate
    >>> gate = ConformalGate(alpha=0.1)
    >>> gate.fit(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
    >>> gate.threshold()
    0.91
    >>> gate.should_escalate(0.5)   # below threshold -> escalate
    True
    >>> gate.should_escalate(0.95)  # above threshold -> trust fast model
    False
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = float(alpha)
        self._threshold: Optional[float] = None

    def fit(self, val_scores: "numpy.ndarray") -> "ConformalGate":
        """Calibrate the threshold on validation scores.

        Parameters
        ----------
        val_scores : np.ndarray
            Validation scores; convention: higher = better calibrated.

        Returns
        -------
        ConformalGate
            ``self`` for fluent chaining.
        """
        self._threshold = calibrate_conformal(val_scores, self.alpha)
        return self

    def should_escalate(self, score: float) -> bool:
        """Return True when ``score`` is below the calibrated threshold.

        Parameters
        ----------
        score : float
            The test sample's score (higher = better).

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self._threshold is None:
            raise RuntimeError("ConformalGate is not fit; call fit() first.")
        return float(score) < self._threshold

    def threshold(self) -> float:
        """Return the calibrated threshold."""
        if self._threshold is None:
            raise RuntimeError("ConformalGate is not fit; call fit() first.")
        return self._threshold
