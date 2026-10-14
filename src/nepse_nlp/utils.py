"""Shared utilities: reproducibility, logging, configuration, path constants.

All path constants are anchored to the repository layout so that the package
works both when installed (``pip install -e .``) and when used via ``sys.path``
insertion from scripts/tests.
"""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
#  Path constants (anchored to this file: src/nepse_nlp/utils.py)             #
# --------------------------------------------------------------------------- #
# utils.py -> nepse_nlp/ -> src/ -> repo root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
EXTERNAL_DIR: Path = DATA_DIR / "external"
MODELS_DIR: Path = PROJECT_ROOT / "models"
MLRUNS_DIR: Path = PROJECT_ROOT / "mlruns"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
CONFIG_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"


# --------------------------------------------------------------------------- #
#  Reproducibility                                                            #
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> int:
    """Seed Python, NumPy, and (if available) PyTorch / TF / transformers RNG.

    Returns the seed so callers can log it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # optional
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:  # optional
        import tensorflow as tf  # type: ignore

        tf.random.set_seed(seed)
    except Exception:
        pass
    return seed


# --------------------------------------------------------------------------- #
#  Logging                                                                    #
# --------------------------------------------------------------------------- #
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str = "nepse_nlp", level: int | str | None = None) -> logging.Logger:
    """Return a configured logger. Idempotent (no duplicate handlers)."""
    logger = logging.getLogger(name)
    if logger.handlers:
        if level is not None:
            logger.setLevel(level)
        return logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level if level is not None else logging.INFO)
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
#  Configuration                                                              #
# --------------------------------------------------------------------------- #
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config file. Falls back to ``CONFIG_PATH`` if ``path`` is None.

    If PyYAML is unavailable or the file is missing, returns an empty dict so
    callers can use ``.get(key, default)`` without crashing.
    """
    target = Path(path) if path is not None else CONFIG_PATH
    if not target.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    with open(target, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


# --------------------------------------------------------------------------- #
#  Misc helpers                                                               #
# --------------------------------------------------------------------------- #
def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed, return ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "PROCESSED_DIR",
    "EXTERNAL_DIR",
    "MODELS_DIR",
    "MLRUNS_DIR",
    "RESULTS_DIR",
    "CONFIG_PATH",
    "set_seed",
    "get_logger",
    "load_config",
    "ensure_dir",
]
