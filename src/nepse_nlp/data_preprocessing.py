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
"""Devanagari-aware text preprocessing for Nepali financial news.

This module is intentionally stdlib-only (``re``, ``unicodedata``, ``json``)
so it imports cleanly even in minimal environments. It provides Unicode NFC
normalization tailored to Devanagari, robust text cleaning, a syllabic
regex tokenizer, a Nepali stopword list, code-mix detection, and an end-to-end
``preprocess_pipeline`` that returns a structured dict for downstream
sentiment modeling (Jev / baseline classifiers).

Design notes
------------
* Consonants ``ब`` (U+092C) and ``व`` (U+0935) are intentionally kept
  distinct — they are phonemically different in Nepali and unifying them
  would harm downstream classifiers.
* Zero-width characters (ZWSP/ZWNJ/ZWJ/BOM) and BIDI control marks are
  stripped because they frequently survive copy-paste from news portals
  and break tokenization.
* The tokenizer is a single compiled regex that emits Devanagari clusters,
  Latin/digit runs, and a small set of sentence/clause punctuation
  (including the Devanagari danda ``।``).
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Dict, List

__all__ = [
    "normalize_devanagari",
    "clean_text",
    "tokenize_nepali",
    "STOPWORDS_NE",
    "remove_stopwords",
    "detect_code_mix",
    "preprocess_pipeline",
]

# ---------------------------------------------------------------------------
# Compiled regexes (compiled once at import for speed)
# ---------------------------------------------------------------------------

# Zero-width and BOM-style characters that survive copy-paste and break
# tokenization.  U+200B ZWSP, U+200C ZWNJ, U+200D ZWJ, U+FEFF ZWNBSP/BOM.
_ZW_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF]")

# BIDI control marks: LRM/RLM (U+200E/F), embedding/override (U+202A-E),
# and the isolates (U+2066-2069).
_BIDI_RE = re.compile(r"[\u200E\u200F\u202A-\u202E\u2066-\u2069]")

# Spurious whitespace inserted before Devanagari combining diacritics.
# anusvara ं (U+0902), visarga ः (U+0903), chandrabindu ँ (U+0901),
# nukta ़ (U+093C).  E.g. "अ ं" -> "अं".
_COMBINING_SPACING_RE = re.compile(r"\s+([\u0901\u0902\u0903\u093C])")

# General whitespace collapse.
_WS_RE = re.compile(r"\s+")

# HTML tag stripper.
_HTML_TAG_RE = re.compile(r"<[^>]+>")

# URL stripper (http/https/www).
_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)

# @-mention stripper.
_MENTION_RE = re.compile(r"@\w+")

# Hashtag -> keep the bare word (drop the leading '#').
_HASHTAG_RE = re.compile(r"#(\w+)")

# Emoji stripper covering the common pictographic/symbol ranges.
_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f680-\U0001f6ff"  # transport & map
    "\U0001f700-\U0001f77f"  # alchemical symbols
    "\U0001f780-\U0001f7ff"  # geometric shapes ext
    "\U0001f800-\U0001f8ff"  # supplemental arrows-C
    "\U0001f900-\U0001f9ff"  # supplemental symbols & pictographs
    "\U0001fa00-\U0001fa6f"  # chess symbols / extended-A
    "\U0001fa70-\U0001faff"  # symbols & pictographs extended-A/B
    "\U00002600-\U000026ff"  # misc symbols
    "\U00002700-\U000027bf"  # dingbats
    "\U0001f000-\U0001f02f"  # mahjong / dominoes
    "\U0001f0a0-\U0001f0ff"  # playing cards
    "]+",
    flags=re.UNICODE,
)

# Tokenizer: punctuation first (so danda । and ॥ are captured alone), then
# Devanagari clusters, then Latin/digit runs.  The danda U+0964 (।) and the
# double-danda U+0965 (॥) live *inside* the Devanagari block (U+0900–U+097F);
# if we let the cluster class match them, `+` greedily glues them onto the
# preceding word (e.g. "बढ्यो।" becomes one token).  We therefore restrict
# the cluster class to [\u0900-\u0963\u0966-\u097F] (skipping U+0964/U+0965)
# so the leading punctuation alternative can match the dandas as standalone
# tokens.
_TOKEN_RE = re.compile(r"[।?!.,;:]|[\u0900-\u0963\u0966-\u097F]+|[A-Za-z0-9]+")

# Pure-Latin-alphabetic token (used for code-mix detection).
_LATIN_WORD_RE = re.compile(r"^[A-Za-z]+$")

# Devanagari presence test (used for language heuristic).
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")

# ---------------------------------------------------------------------------
# Stopwords (~40 common Nepali function words).  Kept as a frozenset for O(1)
# membership tests.  Note: ब/व kept distinct per project convention.
# ---------------------------------------------------------------------------
STOPWORDS_NE: frozenset = frozenset(
    {
        "र",
        "यो",
        "यस",
        "त्यो",
        "त्यस",
        "भने",
        "गरेको",
        "छ",
        "थियो",
        "गर्न",
        "गर्दा",
        "लाई",
        "ले",
        "को",
        "मा",
        "पनि",
        "नि",
        "हो",
        "हुन",
        "गरौं",
        "जो",
        "जुन",
        "तर",
        "अनि",
        "अथवा",
        "वा",
        "हुन्",
        "गरें",
        "गर्छ",
        "गर्थे",
        "हुन्छ",
        "थिए",
        "छन्",
        "यसरी",
        "त्यसरी",
        "कसरी",
        "किन",
        "कहिले",
        "कहाँ",
        "अब",
    }
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_devanagari(text: str) -> str:
    """NFC-normalize a Devanagari string and strip noise.

    Operations:
      1. NFC composition (combines Devanagari matras with their consonants).
      2. Remove zero-width characters (ZWSP/ZWNJ/ZWJ/BOM).
      3. Remove BIDI control marks.
      4. Collapse spurious whitespace before anusvara/visarga/chandrabindu/nukta
         (e.g. ``"अ ं"`` -> ``"अं"``).
      5. Collapse all remaining whitespace to single spaces.

    Note: ``ब`` (ba) and ``व`` (va) are kept distinct — they are
    phonemically different in Nepali.
    """
    if not isinstance(text, str):
        text = str(text)
    # 1. NFC composition.
    text = unicodedata.normalize("NFC", text)
    # 2. Strip zero-width / BOM chars.
    text = _ZW_RE.sub("", text)
    # 3. Strip BIDI control marks.
    text = _BIDI_RE.sub("", text)
    # 4. Pull combining diacritics back onto their base consonant.
    text = _COMBINING_SPACING_RE.sub(r"\1", text)
    # 5. Collapse whitespace.
    text = _WS_RE.sub(" ", text).strip()
    return text


def clean_text(text: str) -> str:
    """Clean noisy news text while preserving Devanagari + Latin content.

    Removes HTML tags, URLs, @mentions, emoji, and the leading ``#`` of
    hashtags (the word itself is retained).  Collapses whitespace.
    The Devanagari danda and basic Latin/Devanagari punctuation survive.
    """
    if not isinstance(text, str):
        text = str(text)
    text = _HTML_TAG_RE.sub(" ", text)  # drop HTML tags
    text = _URL_RE.sub(" ", text)  # drop URLs
    text = _MENTION_RE.sub(" ", text)  # drop @mentions
    text = _HASHTAG_RE.sub(r"\1", text)  # keep hashtag word, drop '#'
    text = _EMOJI_RE.sub(" ", text)  # drop emoji
    text = _WS_RE.sub(" ", text).strip()  # collapse whitespace
    return text


def tokenize_nepali(text: str) -> List[str]:
    """Tokenize Nepali / code-mixed text with a single regex.

    Emits, in priority order:
      * clause/sentence punctuation: ``। ? ! . , ; :``
      * Devanagari syllabic clusters ``[\\u0900-\\u097F]+``
      * Latin/digit runs ``[A-Za-z0-9]+``
    """
    if not isinstance(text, str):
        text = str(text)
    return _TOKEN_RE.findall(text)


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Drop Nepali stopword tokens.  Non-Nepali tokens pass through."""
    return [t for t in tokens if t not in STOPWORDS_NE]


def detect_code_mix(text: str) -> float:
    """Return the fraction of Latin-alphabetic tokens over all tokens.

    Pure Nepali text -> ~0.0 ; pure English text -> ~1.0.
    Returns 0.0 for empty input.
    """
    tokens = tokenize_nepali(text)
    if not tokens:
        return 0.0
    latin = sum(1 for t in tokens if _LATIN_WORD_RE.match(t))
    return latin / len(tokens)


def preprocess_pipeline(text: str) -> Dict:
    """Run the full Phase-1 preprocessing pipeline on a single headline.

    Returns a dict with keys:
      ``raw``              -- original text
      ``clean``            -- cleaned text (HTML/URL/mention/emoji stripped)
      ``normalized``       -- Devanagari-NFC-normalized, noise-free text
      ``tokens``           -- token list from the normalized text
      ``tokens_no_stop``   -- tokens with Nepali stopwords removed
      ``code_mix_ratio``   -- Latin alphabetic token fraction (0..1)
      ``language``         -- heuristic tag: 'ne' | 'en' | 'code-mixed'
    """
    raw = text if isinstance(text, str) else str(text)
    cleaned = clean_text(raw)
    normalized = normalize_devanagari(cleaned)
    tokens = tokenize_nepali(normalized)
    tokens_no_stop = remove_stopwords(tokens)
    ratio = detect_code_mix(normalized)

    has_devanagari = bool(_DEVANAGARI_RE.search(normalized))
    if 0.1 <= ratio <= 0.9:
        language = "code-mixed"
    elif has_devanagari:
        language = "ne"
    else:
        language = "en"

    return {
        "raw": raw,
        "clean": cleaned,
        "normalized": normalized,
        "tokens": tokens,
        "tokens_no_stop": tokens_no_stop,
        "code_mix_ratio": ratio,
        "language": language,
    }


# ---------------------------------------------------------------------------
# Module self-test / demo (no external deps)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo = (
        "NEPSE आज बढ्यो। NABIL +2.5%  👉 "
        "https://example.com/news #NEPSE @trader  "
        "निवौ निवेश बैंकले लाभांश घोषणा गर्यो।"
    )
    out = preprocess_pipeline(demo)
    print(json.dumps(out, ensure_ascii=False, indent=2))
