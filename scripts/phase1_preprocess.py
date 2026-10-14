# MIT License
#
# Copyright (c) 2024 Nepali Sentiment NEPSE Research Project
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Phase-1 CLI entry: simulate data + preprocess headlines + write CSVs.

Runs fully offline (no network).  Produces:

* ``data/processed/news_sentiment.csv``
    columns: date, title, sector, true_label, code_mix_ratio, n_tokens, language
* ``data/processed/nepse_daily.csv``
    columns: date, open, high, low, close, volume

Usage::

    python scripts/phase1_preprocess.py
"""

from __future__ import annotations

import csv
import sys
import pathlib
from typing import Any, Dict, List

# Make ``nepse_nlp`` importable from the repo's ``src/`` directory without an
# install step.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Shared utilities (written by the lead in parallel).  We import the directed
# symbols; if the lead's module is not yet present we fall back to a minimal
# shim so this script remains runnable in isolation.
try:
    from nepse_nlp.utils import set_seed, get_logger, PROCESSED_DIR  # type: ignore
except Exception:  # pragma: no cover - defensive fallback
    import logging
    import random as _random

    def set_seed(seed: int = 42) -> None:  # type: ignore[override]
        _random.seed(seed)
        try:
            import numpy as _np  # type: ignore
            _np.random.seed(seed)
        except Exception:
            pass

    def get_logger(name: str):  # type: ignore[override]
        return logging.getLogger(name)

    PROCESSED_DIR = _REPO_ROOT / "data" / "processed"  # type: ignore[assignment]

from nepse_nlp.scraper import simulated_headlines, simulate_nepse_series  # noqa: E402
from nepse_nlp.data_preprocessing import preprocess_pipeline  # noqa: E402


NEWS_CSV_FIELDS = [
    "date",
    "title",
    "sector",
    "true_label",
    "code_mix_ratio",
    "n_tokens",
    "language",
]
NEPSE_CSV_FIELDS = ["date", "open", "high", "low", "close", "volume"]


def _write_news_csv(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    """Write news rows as a CSV using the stdlib csv module (pandas-free)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=NEWS_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_nepse_csv(path: pathlib.Path, data: Any) -> int:
    """Write the NEPSE series to CSV, accepting either a DataFrame or list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # DataFrame path (pandas present).
    if hasattr(data, "to_csv"):
        data.to_csv(path, index=False)
        # Best-effort row count via the object's length.
        try:
            return int(len(data))
        except Exception:
            return 0
    # list-of-dicts path (pandas absent).
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=NEPSE_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    return len(data)


def main() -> None:
    log = get_logger("phase1")
    set_seed(42)

    # ---- 1. Simulate raw Phase-1 data (offline, deterministic) ----
    headlines = simulated_headlines(n=50, seed=42)
    nepse = simulate_nepse_series(n_days=250, start_price=2100.0, seed=42)

    # ---- 2. Preprocess each headline ----
    news_rows: List[Dict[str, Any]] = []
    code_mix_sum = 0.0
    n_tokens_sum = 0
    for h in headlines:
        out = preprocess_pipeline(h["title"])
        n_tokens = len(out["tokens"])  # total token count after normalization
        news_rows.append(
            {
                "date": h["date"],
                "title": h["title"],
                "sector": h["sector"],
                "true_label": h["true_label"],
                "code_mix_ratio": round(out["code_mix_ratio"], 4),
                "n_tokens": n_tokens,
                "language": out["language"],
            }
        )
        code_mix_sum += out["code_mix_ratio"]
        n_tokens_sum += n_tokens

    # ---- 3. Write processed CSVs ----
    processed_dir = pathlib.Path(PROCESSED_DIR)
    news_path = processed_dir / "news_sentiment.csv"
    nepse_path = processed_dir / "nepse_daily.csv"

    _write_news_csv(news_path, news_rows)
    nepse_rows = _write_nepse_csv(nepse_path, nepse)

    # ---- 4. Summary ----
    n_articles = len(news_rows)
    mean_code_mix = (code_mix_sum / n_articles) if n_articles else 0.0
    mean_n_tokens = (n_tokens_sum / n_articles) if n_articles else 0.0

    log.info("Phase-1 preprocessing complete.")
    print("=" * 60)
    print("Phase 1 — Data Collection & Preprocessing Summary")
    print("=" * 60)
    print(f"News articles processed : {n_articles}")
    print(f"Mean code-mix ratio     : {mean_code_mix:.4f}")
    print(f"Mean tokens / headline  : {mean_n_tokens:.2f}")
    print(f"NEPSE daily rows         : {nepse_rows}")
    print(f"  -> {news_path}")
    print(f"  -> {nepse_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
