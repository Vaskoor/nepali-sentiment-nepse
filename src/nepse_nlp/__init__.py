"""nepse_nlp — Nepali financial sentiment analysis for NEPSE stock prediction.

Subpackages / modules
---------------------
- data_preprocessing : Devanagari normalization, cleaning, tokenization, code-mix detection.
- scraper             : News + NEPSE data collection with deterministic simulated fallback.
- jev_benchmark       : Jev (System-One) zero-shot classifier + benchmark harness.
- baselines           : NepaliBERT / MuRIL / XLM-R / GPT-4o-simulated zero-shot baselines.
- calibration         : Brier score, Expected Calibration Error (ECE), reliability data.
- conformal           : Split-conformal threshold calibration + ConformalGate.
- hybrid_pipeline     : Confidence-gated hybrid (fast Jev -> escalate) with Pareto analysis.
- features            : Sentiment aggregation + technical indicators + feature matrix.
- prediction          : NEPSE next-day return / direction predictors + walk-forward.
- backtest            : Walk-forward trading simulation with NEPSE transaction costs.
- utils               : Seeds, logging, config loading, path constants.

License: MIT
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("nepse-nlp")
except PackageNotFoundError:  # not installed (dev mode)
    __version__ = "0.2.0"

__all__ = ["__version__"]
