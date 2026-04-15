"""
News Fetcher — Recent headlines and sentiment for trading symbols.

Tries Finnhub first (better financial news coverage, generous free tier),
then falls back to the existing NewsAPI integration from config.py.
Returns empty results silently if neither key is configured — the rest of
the pipeline continues without news context.

Env vars:
    FINNHUB_API_KEY   — Finnhub.io free tier (preferred)
    NEWS_API_KEY      — NewsAPI.org (fallback, already in config.py)

Caches results for 15 minutes per symbol to avoid burning rate limits.
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_CACHE: Dict[str, tuple] = {}   # symbol -> (fetch_ts, results)
_CACHE_TTL_SECS = 900           # 15 minutes

# Simple keyword lists for rule-based sentiment scoring
_POSITIVE_WORDS = {
    "beat", "beats", "record", "growth", "profit", "gain", "gains", "surge",
    "rally", "upgrade", "upgrades", "raised", "raise", "exceed", "exceeds",
    "strong", "bullish", "outperform", "outperforms", "buyback", "dividend",
    "revenue growth", "earnings beat", "approved", "launch", "launches",
    "partnership", "deal", "breakthrough",
}
_NEGATIVE_WORDS = {
    "miss", "misses", "decline", "loss", "losses", "drop", "drops", "fall",
    "falls", "plunge", "plunges", "cut", "cuts", "downgrade", "downgrades",
    "lowered", "disappoint", "disappoints", "weak", "bearish", "underperform",
    "underperforms", "earnings miss", "lawsuit", "recall", "fraud", "warning",
    "investigation", "fine", "fined", "bankruptcy", "layoffs", "layoff",
    "shortfall", "concern", "concerns",
}


class NewsFetcher:
    """
    Fetch recent news headlines and derive simple sentiment for symbols.

    Usage:
        fetcher = NewsFetcher()
        headlines = fetcher.fetch_news("AAPL")
        # -> [{"headline": "...", "sentiment": "positive", "published_at": "...", "source": "..."}]

        batch = fetcher.fetch_news_batch(["AAPL", "MSFT", "NVDA"])
        # -> {"AAPL": [...], "NVDA": [...]}  (empty symbols omitted)
    """

    def __init__(self):
        # Finnhub: env var first, then config.py hardcoded fallback
        self.finnhub_key = os.getenv("FINNHUB_API_KEY") or ""
        if not self.finnhub_key.strip():
            try:
                from config import FINNHUB_API_KEY as cfg_fh
                self.finnhub_key = cfg_fh or ""
            except (ImportError, AttributeError):
                pass
        self.finnhub_key = self.finnhub_key.strip()

        # NewsAPI: config.py first, then env var
        self.newsapi_key = ""
        try:
            from config import NEWS_API_KEY
            if NEWS_API_KEY and NEWS_API_KEY != "YOUR_NEWS_API_KEY":
                self.newsapi_key = NEWS_API_KEY
        except (ImportError, AttributeError):
            self.newsapi_key = os.getenv("NEWS_API_KEY", "")

        if self.finnhub_key:
            logger.info("NewsFetcher: Finnhub enabled (primary)")
        elif self.newsapi_key:
            logger.info("NewsFetcher: NewsAPI enabled (Finnhub not configured)")
        else:
            logger.info("NewsFetcher: No news API configured — news features disabled")

    @property
    def is_available(self) -> bool:
        return bool(self.finnhub_key or self.newsapi_key)

    # ── Public API ───────────────────────────────────────────────────────

    def fetch_news(self, symbol: str, days_back: int = 2) -> List[Dict]:
        """
        Return recent news for one symbol.

        Results are cached for _CACHE_TTL_SECS to avoid rate-limit issues.
        Returns empty list if no API is configured or the request fails.
        """
        cache_entry = _CACHE.get(symbol)
        if cache_entry and (time.time() - cache_entry[0]) < _CACHE_TTL_SECS:
            return cache_entry[1]

        result: List[Dict] = []
        if self.finnhub_key:
            result = self._fetch_finnhub(symbol, days_back)
        elif self.newsapi_key:
            result = self._fetch_newsapi(symbol, days_back)

        _CACHE[symbol] = (time.time(), result)
        return result

    def fetch_news_batch(
        self, symbols: List[str], days_back: int = 2
    ) -> Dict[str, List[Dict]]:
        """
        Fetch news for multiple symbols.
        Returns {symbol: [headlines]} — symbols with no news are omitted.
        """
        results: Dict[str, List[Dict]] = {}
        for symbol in symbols:
            try:
                headlines = self.fetch_news(symbol, days_back)
                if headlines:
                    results[symbol] = headlines
            except Exception as e:
                logger.debug(f"NewsFetcher: Error fetching {symbol}: {e}")
        return results

    # ── Finnhub ──────────────────────────────────────────────────────────

    def _fetch_finnhub(self, symbol: str, days_back: int) -> List[Dict]:
        """Fetch company news from Finnhub API."""
        to_dt = datetime.now()
        from_dt = to_dt - timedelta(days=days_back)
        try:
            resp = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={
                    "symbol": symbol,
                    "from": from_dt.strftime("%Y-%m-%d"),
                    "to": to_dt.strftime("%Y-%m-%d"),
                    "token": self.finnhub_key,
                },
                timeout=6,
            )
            if resp.status_code == 429:
                logger.warning("NewsFetcher: Finnhub rate limit hit — skipping")
                return []
            if resp.status_code != 200:
                logger.debug(f"NewsFetcher: Finnhub returned {resp.status_code} for {symbol}")
                return []
            items = resp.json()
            if not isinstance(items, list):
                return []
            output = []
            for item in items[:5]:  # Cap at 5 headlines per symbol
                headline = (item.get("headline") or "").strip()
                summary = (item.get("summary") or "").strip()
                if not headline:
                    continue
                sentiment = self._score_sentiment(headline + " " + summary)
                output.append({
                    "headline": headline,
                    "sentiment": sentiment,
                    "published_at": str(item.get("datetime", "")),
                    "source": item.get("source", ""),
                })
            return output
        except requests.exceptions.Timeout:
            logger.debug(f"NewsFetcher: Finnhub timeout for {symbol}")
            return []
        except Exception as e:
            logger.debug(f"NewsFetcher: Finnhub error for {symbol}: {e}")
            return []

    # ── NewsAPI ──────────────────────────────────────────────────────────

    def _fetch_newsapi(self, symbol: str, days_back: int) -> List[Dict]:
        """Fetch from NewsAPI.org."""
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": symbol,
                    "from": from_date,
                    "sortBy": "publishedAt",
                    "pageSize": 5,
                    "language": "en",
                    "apiKey": self.newsapi_key,
                },
                timeout=6,
            )
            if resp.status_code == 429:
                logger.warning("NewsFetcher: NewsAPI rate limit hit — skipping")
                return []
            if resp.status_code != 200:
                logger.debug(f"NewsFetcher: NewsAPI returned {resp.status_code} for {symbol}")
                return []
            data = resp.json()
            articles = data.get("articles", [])
            output = []
            for a in articles[:5]:
                title = (a.get("title") or "").strip()
                description = (a.get("description") or "").strip()
                if not title or title == "[Removed]":
                    continue
                sentiment = self._score_sentiment(title + " " + description)
                output.append({
                    "headline": title,
                    "sentiment": sentiment,
                    "published_at": a.get("publishedAt", ""),
                    "source": (a.get("source") or {}).get("name", ""),
                })
            return output
        except requests.exceptions.Timeout:
            logger.debug(f"NewsFetcher: NewsAPI timeout for {symbol}")
            return []
        except Exception as e:
            logger.debug(f"NewsFetcher: NewsAPI error for {symbol}: {e}")
            return []

    # ── Sentiment Scoring ────────────────────────────────────────────────

    @staticmethod
    def _score_sentiment(text: str) -> str:
        """
        Keyword-based sentiment: 'positive', 'negative', or 'neutral'.
        Simple but effective for financial news headlines.
        """
        text_lower = text.lower()
        pos = sum(1 for w in _POSITIVE_WORDS if w in text_lower)
        neg = sum(1 for w in _NEGATIVE_WORDS if w in text_lower)
        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"
