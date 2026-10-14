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
"""Unit tests for nepse_nlp.data_preprocessing (stdlib + package only).

Tests run without installing the package by inserting ``src/`` onto
``sys.path`` at collection time.
"""

import pathlib
import sys

# Make ``nepse_nlp`` importable from the repo's ``src/`` directory without an
# install step.  This mirrors the pattern used by scripts/phase1_preprocess.py.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from nepse_nlp.data_preprocessing import (  # noqa: E402
    STOPWORDS_NE,
    clean_text,
    detect_code_mix,
    normalize_devanagari,
    preprocess_pipeline,
    remove_stopwords,
    tokenize_nepali,
)


def test_normalize_idempotent_and_strips_zwsp() -> None:
    """Normalizing twice == normalizing once; zero-width chars are removed."""
    raw = "नेपाल\u200b शेयर\u200c बजार\u200d"
    once = normalize_devanagari(raw)
    twice = normalize_devanagari(once)
    assert once == twice  # idempotent
    assert "\u200b" not in once
    assert "\u200c" not in once
    assert "\u200d" not in once


def test_clean_text_removes_url_html_emoji() -> None:
    """clean_text strips URLs, HTML, @mentions, emoji but keeps words."""
    raw = "Visit <b>https://example.com</b> @user #NEPSE 🚀"
    out = clean_text(raw)
    assert "http" not in out
    assert "www." not in out
    assert "<" not in out
    assert ">" not in out
    assert "@" not in out
    assert "🚀" not in out
    assert "NEPSE" in out  # hashtag word kept (leading '#' dropped)


def test_tokenize_splits_devanagari_and_latin() -> None:
    """Tokenizer yields the expected sequence of Devanagari/Latin/punct tokens."""
    text = "NEPSE आज बढ्यो। NABIL +2.5%"
    toks = tokenize_nepali(text)
    # '।' (danda) must be a separate token, not glued to "बढ्यो".
    # '+' and '%' are not in the punctuation set -> ignored.
    # '.' IS in the punct set -> emitted as its own token between '2' and '5'.
    assert toks == ["NEPSE", "आज", "बढ्यो", "।", "NABIL", "2", ".", "5"]


def test_code_mix_ratio_in_range() -> None:
    """Ratio is bounded in [0, 1]; ~0 for pure Nepali, ~1 for pure English."""
    assert 0.0 <= detect_code_mix("NEPSE आज बढ्यो") <= 1.0
    assert abs(detect_code_mix("यो एक नेपाली वाक्य हो") - 0.0) < 1e-9
    assert abs(detect_code_mix("this is pure english text") - 1.0) < 1e-9


def test_preprocess_pipeline_keys() -> None:
    """Pipeline dict contains all expected keys."""
    out = preprocess_pipeline("NEPSE आज बढ्यो। https://x.io")
    expected = {
        "raw",
        "clean",
        "normalized",
        "tokens",
        "tokens_no_stop",
        "code_mix_ratio",
        "language",
    }
    assert expected.issubset(out.keys())


def test_stopwords_removed() -> None:
    """No stopword survives remove_stopwords."""
    text = "यो शेयर बजार आज बढ्यो र NABIL पनि मजबुत"
    toks = tokenize_nepali(text)
    out = remove_stopwords(toks)
    for tok in out:
        assert tok not in STOPWORDS_NE
    # Sanity: at least the stopwords that were present got removed.
    assert "यो" not in out
    assert "र" not in out
    assert "पनि" not in out
