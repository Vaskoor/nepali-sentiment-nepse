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
"""Phase-1 data collection for Nepali financial news + NEPSE OHLCV.

This module provides:

* :data:`NEWS_SOURCES` -- declarative catalog of Nepali financial portals.
* :func:`fetch_page`   -- async HTTP fetch guarded behind an ``aiohttp``
  import so the module loads without it.
* :func:`parse_articles` -- BeautifulSoup-based extraction that degrades
  gracefully when ``bs4`` is missing or the HTML is unparseable.
* :func:`scrape_news`  -- end-to-end orchestration with a deterministic
  simulated-headline fallback for offline / sandboxed runs.
* :func:`simulated_headlines` -- a seeded, realistic Nepali financial
  headline generator (>=30 templates, 4 sectors x 3 sentiment classes).
* :func:`simulate_nepse_series` -- a NEPSE-like daily OHLCV generator using
  a random-walk-with-drift plus a simple GARCH-ish volatility clustering
  (EWMA of squared returns).  Returns a ``pandas.DataFrame`` if pandas is
  available, otherwise a list of dicts.
* :func:`fetch_nepse_ohlcv` -- async wrapper that tries a real fetch and
  falls back to :func:`simulate_nepse_series` on any failure.

The whole module is importable and runnable offline: no network access,
no optional third-party packages required at import time.
"""

from __future__ import annotations

import asyncio
import random
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

__all__ = [
    "NEWS_SOURCES",
    "fetch_page",
    "parse_articles",
    "scrape_news",
    "simulated_headlines",
    "simulate_nepse_series",
    "fetch_nepse_ohlcv",
    "main",
]


# ---------------------------------------------------------------------------
# News source catalog (declarative; parser_hint is a CSS-ish selector intent)
# ---------------------------------------------------------------------------
NEWS_SOURCES: Dict[str, Dict[str, str]] = {
    "MeroLagani": {
        "base_url": "https://merolagani.com",
        "parser_hint": "div.news-list a.news-title",
    },
    "ShareSansar": {
        "base_url": "https://sharesansar.com",
        "parser_hint": "div.news-item h4 a",
    },
    "NepaliPaisa": {
        "base_url": "https://www.nepalipaisa.com",
        "parser_hint": "article.featured-news a",
    },
    "BizMandu": {
        "base_url": "https://bizmandu.com",
        "parser_hint": "div.list-item a.title",
    },
}


# ---------------------------------------------------------------------------
# Async HTTP fetch (guarded against missing aiohttp)
# ---------------------------------------------------------------------------
async def fetch_page(session: Any, url: str, timeout: float = 10.0) -> Optional[str]:
    """GET ``url`` via the provided aiohttp session; return HTML text or None.

    ``aiohttp`` is imported inside a try/except so this module remains
    importable without it.  Any request-level failure returns ``None``
    (callers fall back to the simulator).
    """
    try:
        import aiohttp  # noqa: F401  (import is the guarded step)
    except ImportError:
        return None
    try:
        # aiohttp is available; perform the actual request.
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            if resp.status != 200:
                return None
            return await resp.text()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HTML -> article dicts (guarded against missing bs4)
# ---------------------------------------------------------------------------
def parse_articles(html: str, source: str) -> List[Dict[str, str]]:
    """Extract ``{title, date, url, summary}`` dicts from raw HTML.

    Degrades gracefully: returns ``[]`` when ``bs4`` is unavailable or the
    parse path raises.  The extraction here is intentionally generic
    (anchor-driven) because portal-specific selectors live in
    :data:`NEWS_SOURCES` and would be wired up per-source in production.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return []
    try:
        soup = BeautifulSoup(html, "html.parser")
        out: List[Dict[str, str]] = []
        for a in soup.find_all("a"):
            title = a.get_text(strip=True)
            href = a.get("href", "") or ""
            if title and len(title) > 15 and href:
                out.append(
                    {
                        "title": title,
                        "date": "",
                        "url": (
                            href
                            if href.startswith("http")
                            else f"{NEWS_SOURCES.get(source, {}).get('base_url', '')}{href}"
                        ),
                        "summary": "",
                    }
                )
        return out
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Orchestration: real scrape -> simulated fallback
# ---------------------------------------------------------------------------
async def scrape_news(source: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Fetch + parse news for ``source``; fall back to simulation on failure.

    The fallback path is the default in offline sandboxes, ensuring the
    pipeline always yields data for downstream feature engineering.
    """
    cfg = NEWS_SOURCES.get(source)
    if not cfg:
        return simulated_headlines(limit)
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        return simulated_headlines(limit)
    try:
        async with aiohttp.ClientSession() as session:
            html = await fetch_page(session, cfg["base_url"])
            if not html:
                return simulated_headlines(limit)
            articles = parse_articles(html, source)
            if not articles:
                return simulated_headlines(limit)
            return articles[:limit]
    except Exception:
        return simulated_headlines(limit)


# ---------------------------------------------------------------------------
# Simulated Nepali financial headlines
# ---------------------------------------------------------------------------
# Each tuple: (title, summary, sector, true_label, source_name).
# 4 sectors (banking, hydropower, insurance, manufacturing) x 3 sentiment
# classes (positive, negative, neutral) x 3 templates = 36 distinct templates.
_HEADLINE_TEMPLATES: List[Dict[str, str]] = [
    # --- Banking / positive ---
    {
        "title": "Nepal Investment Bank announces 15% dividend; share price surges",
        "summary": "NIB announces strong dividend on the back of record Q4 profit.",
        "sector": "banking",
        "true_label": "positive",
        "source": "MeroLagani",
    },
    {
        "title": "निवौ निवेश बैंकले रु. १५ लाभांश वितरण गर्ने घोषणा",
        "summary": "निवौ निवेश बैंकले रेकर्ड नाफासहित लाभांश घोषणा गरेको छ।",
        "sector": "banking",
        "true_label": "positive",
        "source": "ShareSansar",
    },
    {
        "title": "NABIL Bank posts record profit in Q4; deposits up 12% YoY",
        "summary": "NABIL reports record quarterly profit and double-digit deposit growth.",
        "sector": "banking",
        "true_label": "positive",
        "source": "BizMandu",
    },
    # --- Banking / negative ---
    {
        "title": "Nepal Bank Federation warns of rising NPLs across commercial banks",
        "summary": "Non-performing loans climb as credit growth slows.",
        "sector": "banking",
        "true_label": "negative",
        "source": "ShareSansar",
    },
    {
        "title": "बैंकिङ क्षेत्रमा नकारात्मक ऋण बढ्दै; NPL ५ प्रतिशत पुग्यो",
        "summary": "वाणिज्य बैंकहरूको नकारात्मक ऋण बढ्दै, तरलता थिचाइमा।",
        "sector": "banking",
        "true_label": "negative",
        "source": "NepaliPaisa",
    },
    {
        "title": "RBB reports drop in net profit; liquidity tightens",
        "summary": "Rastriya Banijya Bank sees profit decline amid liquidity stress.",
        "sector": "banking",
        "true_label": "negative",
        "source": "MeroLagani",
    },
    # --- Banking / neutral ---
    {
        "title": "Nepal Rastra Bank reviews monetary policy; rates unchanged",
        "summary": "NRB keeps policy rates steady in the quarterly review.",
        "sector": "banking",
        "true_label": "neutral",
        "source": "BizMandu",
    },
    {
        "title": "वाणिज्य बैंकहरूको त्रैमासिक प्रतिवेदन सार्वजनिक",
        "summary": "वाणिज्य बैंकहरूले त्रैमासिक प्रतिवेदन सार्वजनिक गरेका छन्।",
        "sector": "banking",
        "true_label": "neutral",
        "source": "ShareSansar",
    },
    {
        "title": "ADB approves fresh credit line to Nepali banking sector",
        "summary": "Asian Development Bank extends a credit line to support SME lending.",
        "sector": "banking",
        "true_label": "neutral",
        "source": "NepaliPaisa",
    },
    # --- Hydropower / positive ---
    {
        "title": "Chilime Hydropower profit jumps 20%; generation above target",
        "summary": "Chilime reports above-target generation and strong profit growth.",
        "sector": "hydropower",
        "true_label": "positive",
        "source": "MeroLagani",
    },
    {
        "title": "जलविद्युत कम्पनीहरूको नाफा बढ्यो; उत्पादन लक्ष्यभन्दा बढी",
        "summary": "जलविद्युत कम्पनीहरूको नाफा बढ्यो, उत्पादन लक्ष्यभन्दा बढी रहेको छ।",
        "sector": "hydropower",
        "true_label": "positive",
        "source": "ShareSansar",
    },
    {
        "title": "Upper Tamakoshi resumes full generation; revenue outlook strong",
        "summary": "Upper Tamakoshi returns to full capacity; revenue outlook upgraded.",
        "sector": "hydropower",
        "true_label": "positive",
        "source": "BizMandu",
    },
    # --- Hydropower / negative ---
    {
        "title": "Drought cuts hydropower output; NEA warns of winter shortages",
        "summary": "Reduced river flow threatens winter generation capacity.",
        "sector": "hydropower",
        "true_label": "negative",
        "source": "NepaliPaisa",
    },
    {
        "title": "जलविद्युत आयोजना निर्माणमा ढिलाइ; लक्ष्य गुजारा",
        "summary": "जलविद्युत आयोजना निर्माणमा ढिलाइ, लक्ष्य गुजारा हुने देखिएको छ।",
        "sector": "hydropower",
        "true_label": "negative",
        "source": "MeroLagani",
    },
    {
        "title": "Cost overrun deepens at key hydropower project; commissioning slips",
        "summary": "A flagship project sees further cost overrun and schedule slip.",
        "sector": "hydropower",
        "true_label": "negative",
        "source": "ShareSansar",
    },
    # --- Hydropower / neutral ---
    {
        "title": "NEA signs PPA with new 50MW project; tariff under review",
        "summary": "NEA and developer sign a power purchase agreement; tariff pending.",
        "sector": "hydropower",
        "true_label": "neutral",
        "source": "BizMandu",
    },
    {
        "title": "जलविद्युत लगानी नीतिमा समीक्षा सुरु",
        "summary": "सरकारले जलविद्युत लगानी नीतिमा समीक्षा सुरु गरेको छ।",
        "sector": "hydropower",
        "true_label": "neutral",
        "source": "NepaliPaisa",
    },
    {
        "title": "Department of Electricity Development invites project proposals",
        "summary": "DoED invites expressions of interest for reservoir-type projects.",
        "sector": "hydropower",
        "true_label": "neutral",
        "source": "MeroLagani",
    },
    # --- Insurance / positive ---
    {
        "title": "Nepal Life Insurance premium income grows 18% YoY",
        "summary": "Nepal Life reports robust premium growth and improved persistency.",
        "sector": "insurance",
        "true_label": "positive",
        "source": "ShareSansar",
    },
    {
        "title": "बीमा कम्पनीको प्रिमियम आयमा वृद्धि",
        "summary": "जीवन बीमा कम्पनीहरूको प्रिमियम आयमा वृद्धि भएको छ।",
        "sector": "insurance",
        "true_label": "positive",
        "source": "BizMandu",
    },
    {
        "title": "Shikhar Insurance declares bonus shares; outlook upgraded",
        "summary": "Shikhar Insurance announces bonus shares; analysts upgrade rating.",
        "sector": "insurance",
        "true_label": "positive",
        "source": "MeroLagani",
    },
    # --- Insurance / negative ---
    {
        "title": "Flood claims surge hits non-life insurers' margins",
        "summary": "Monsoon-related claims pressure underwriting margins.",
        "sector": "insurance",
        "true_label": "negative",
        "source": "NepaliPaisa",
    },
    {
        "title": "गैर-जीवन बीमामा दावी भरपुर्ता बढ्यो; नाफामा गिरावट",
        "summary": "गैर-जीवन बीमा कम्पनीको दावी भरपुर्ता बढ्दा नाफामा गिरावट आएको छ।",
        "sector": "insurance",
        "true_label": "negative",
        "source": "ShareSansar",
    },
    {
        "title": "Regulator fines three insurers for solvency shortfall",
        "summary": "Beema Samiti imposes fines for solvency-margin breaches.",
        "sector": "insurance",
        "true_label": "negative",
        "source": "BizMandu",
    },
    # --- Insurance / neutral ---
    {
        "title": "Insurance Board drafts new provisioning guideline",
        "summary": "Beema Samiti circulates a draft provisioning guideline for feedback.",
        "sector": "insurance",
        "true_label": "neutral",
        "source": "MeroLagani",
    },
    {
        "title": "बीमा ऐन संशोधनमा सार्वजनिक सुझाव माग्दै",
        "summary": "बीमा ऐन संशोधनका लागि सार्वजनिक सुझाव मागिएको छ।",
        "sector": "insurance",
        "true_label": "neutral",
        "source": "NepaliPaisa",
    },
    {
        "title": "Beema Samiti reviews micro-insurance framework",
        "summary": "Beema Samiti reviews the micro-insurance product framework.",
        "sector": "insurance",
        "true_label": "neutral",
        "source": "ShareSansar",
    },
    # --- Manufacturing / positive ---
    {
        "title": "Unilever Nepal posts strong Q3; demand recovery on track",
        "summary": "Unilever Nepal reports strong Q3 on demand recovery.",
        "sector": "manufacturing",
        "true_label": "positive",
        "source": "BizMandu",
    },
    {
        "title": "उत्पादन उद्योगहरूको बिक्री बढ्यो; आयात प्रतिस्थापनमा टेवा",
        "summary": "उत्पादन उद्योगहरूको बिक्री बढ्दा आयात प्रतिस्थापनमा टेवा परेको छ।",
        "sector": "manufacturing",
        "true_label": "positive",
        "source": "MeroLagani",
    },
    {
        "title": "Chaudhary Group expands cement capacity; new plant online",
        "summary": "Chaudhary Group commissions a new cement plant, lifting capacity.",
        "sector": "manufacturing",
        "true_label": "positive",
        "source": "NepaliPaisa",
    },
    # --- Manufacturing / negative ---
    {
        "title": "Import curbs bite raw-material supply for manufacturers",
        "summary": "Tighter import controls squeeze raw-material availability.",
        "sector": "manufacturing",
        "true_label": "negative",
        "source": "ShareSansar",
    },
    {
        "title": "उद्योगहरूमा कच्चा पदार्थ अभाव; उत्पादन घट्यो",
        "summary": "कच्चा पदार्थ अभावले गर्दा उद्योगहरूको उत्पादन घटेको छ।",
        "sector": "manufacturing",
        "true_label": "negative",
        "source": "BizMandu",
    },
    {
        "title": "Festive season sales disappoint two-wheeler dealers",
        "summary": "Weak festive demand hits two-wheeler retail sales.",
        "sector": "manufacturing",
        "true_label": "negative",
        "source": "MeroLagani",
    },
    # --- Manufacturing / neutral ---
    {
        "title": "FNCCI meets on new industrial policy framework",
        "summary": "Federation of Nepalese Chambers of Commerce consults on policy.",
        "sector": "manufacturing",
        "true_label": "neutral",
        "source": "NepaliPaisa",
    },
    {
        "title": "उद्योग वाणिज्य महासंघले नीतिमा छलफल",
        "summary": "उद्योग वाणिज्य महासंघले औद्योगिक नीतिमा छलफल गरेको छ।",
        "sector": "manufacturing",
        "true_label": "neutral",
        "source": "ShareSansar",
    },
    {
        "title": "Department of Industry publishes quarterly manufacturing index",
        "summary": "DoI releases the quarterly index of manufacturing production.",
        "sector": "manufacturing",
        "true_label": "neutral",
        "source": "BizMandu",
    },
]
# assert len(_HEADLINE_TEMPLATES) >= 30  # design invariant


def simulated_headlines(n: int = 20, seed: int = 42) -> List[Dict[str, Any]]:
    """Return ``n`` deterministic, realistic Nepali financial headlines.

    Each dict has keys: ``title``, ``date`` (ISO ``YYYY-MM-DD``), ``url``,
    ``summary``, ``source``, ``true_label`` (positive/neutral/negative),
    ``sector`` (banking/hydropower/insurance/manufacturing).

    Determinism: a seeded shuffle of the 36-template pool, then cycling so
    that requests for ``n > 36`` (up to ~100) still yield stable, varied
    output.  Dates advance one calendar day per article starting 2024-01-01.
    """
    rng = random.Random(seed)
    order = list(range(len(_HEADLINE_TEMPLATES)))
    rng.shuffle(order)

    start = date(2024, 1, 1)
    out: List[Dict[str, Any]] = []
    for i in range(n):
        tpl = _HEADLINE_TEMPLATES[order[i % len(order)]]
        out.append(
            {
                "title": tpl["title"],
                "date": (start + timedelta(days=i)).isoformat(),
                "url": f"https://simulated.nepse.nlp/news/{i + 1:04d}",
                "summary": tpl["summary"],
                "source": tpl["source"],
                "true_label": tpl["true_label"],
                "sector": tpl["sector"],
            }
        )
    return out


# ---------------------------------------------------------------------------
# Simulated NEPSE OHLCV series
# ---------------------------------------------------------------------------
def simulate_nepse_series(
    n_days: int = 250,
    start_price: float = 2100.0,
    seed: int = 42,
) -> "Any":
    """Generate a NEPSE-like daily OHLCV series.

    Model: a random walk with small positive drift, where the conditional
    variance follows a simple GARCH(1,1)-ish recursion
    ``sigma2_{t} = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}``
    (i.e. an EWMA of squared returns producing volatility clustering).

    Columns: ``date, open, high, low, close, volume``.
    Dates are ``YYYY-MM-DD`` strings starting 2024-01-01.

    Returns a ``pandas.DataFrame`` if pandas is importable, else a list of
    dicts (so the function is usable in pandas-free environments).
    """
    rng = random.Random(seed)

    mu = 0.0002  # tiny daily drift (~5% annualized)
    omega = 1.0e-6  # long-run variance floor
    alpha = 0.08  # reaction to yesterday's shock
    beta = 0.90  # persistence of variance
    sigma2 = 4.0e-4  # initial variance (~2% daily vol)

    prev_close = float(start_price)
    start = date(2024, 1, 1)
    rows: List[Dict[str, Any]] = []

    for i in range(n_days):
        eps = rng.gauss(0.0, 1.0)
        sigma = sigma2**0.5
        ret = mu + sigma * eps
        close = prev_close * (1.0 + ret)

        open_ = prev_close * (1.0 + rng.gauss(0.0, sigma * 0.3))
        intraday_sigma = sigma * 0.5
        high = max(open_, close) * (1.0 + abs(rng.gauss(0.0, intraday_sigma)))
        low = min(open_, close) * (1.0 - abs(rng.gauss(0.0, intraday_sigma)))
        volume = int(rng.uniform(1_000_000, 8_000_000))

        rows.append(
            {
                "date": (start + timedelta(days=i)).isoformat(),
                "open": round(open_, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": volume,
            }
        )

        # Update conditional variance (EWMA of squared shocks).
        sigma2 = omega + alpha * (ret - mu) ** 2 + beta * sigma2
        prev_close = close

    # Prefer a DataFrame when pandas is available; otherwise return rows.
    try:
        import pandas as pd  # type: ignore

        return pd.DataFrame(rows)
    except ImportError:
        return rows


async def fetch_nepse_ohlcv(
    start: str = "2024-01-01",
    end: str = "2024-12-31",
) -> "Any":
    """Fetch real NEPSE OHLCV; fall back to simulation on any failure.

    The offline sandbox has no live endpoint configured, so the real-fetch
    step intentionally raises and we drop to :func:`simulate_nepse_series`.
    The day span is derived from ``start``/``end`` when both are ISO dates.
    """
    try:
        # No live NEPSE endpoint is configured for the offline sandbox.
        raise RuntimeError("live NEPSE endpoint unavailable in offline sandbox")
    except Exception:
        df = None

    if df is None:
        try:
            s = date.fromisoformat(start)
            e = date.fromisoformat(end)
            n_days = max(1, (e - s).days + 1)
        except Exception:
            n_days = 250
        return simulate_nepse_series(n_days=n_days, start_price=2100.0, seed=42)
    return df


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
async def main() -> None:
    """Demo: print 5 simulated headlines + the head of the NEPSE sim."""
    print("=== Simulated Nepali financial headlines (5) ===")
    for h in simulated_headlines(5):
        print(f"- [{h['date']}] ({h['sector']}/{h['true_label']}) {h['title']}")

    print("\n=== NEPSE simulated series (head) ===")
    df = simulate_nepse_series(250)
    try:
        import pandas as pd  # type: ignore

        if isinstance(df, pd.DataFrame):
            print(df.head().to_string(index=False))
        else:
            print(df[:5])
    except ImportError:
        # list-of-dicts fallback
        print(df[:5])


if __name__ == "__main__":
    asyncio.run(main())
