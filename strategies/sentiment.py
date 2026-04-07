"""
News / Social Sentiment Analysis Strategy.

Inspired by:
  - Two Sigma's alternative data (news, satellite, social) pipelines
  - Citadel's real-time news analysis and sentiment extraction
  - DE Shaw's NLP-driven alpha signals

Concept: Market prices are driven by information flow. By analyzing
the sentiment of recent news, we can anticipate short-term price
movements before the market fully digests the news.

Uses a keyword-based sentiment scorer (no external ML dependency)
with plans to upgrade to transformer-based NLP.
"""

import re
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from strategies.mean_reversion import Signal


# ── Sentiment Lexicon ────────────────────────────────────────────────────
# Simplified financial sentiment dictionary
POSITIVE_WORDS = {
    "beat", "beats", "exceeded", "exceeds", "surpass", "surge", "surged",
    "soar", "soared", "rally", "rallied", "gain", "gained", "gains",
    "profit", "profitable", "upgrade", "upgraded", "outperform", "bullish",
    "growth", "growing", "record", "high", "boost", "boosted", "strong",
    "positive", "optimistic", "innovation", "breakthrough", "recovery",
    "recovered", "expansion", "dividend", "buyback", "acquisition",
    "partnership", "approved", "approval", "revenue", "earnings",
}

NEGATIVE_WORDS = {
    "miss", "missed", "misses", "decline", "declined", "drop", "dropped",
    "fall", "fell", "crash", "crashed", "plunge", "plunged", "loss",
    "losses", "downgrade", "downgraded", "underperform", "bearish",
    "recession", "slowdown", "weak", "warning", "risk", "risks", "lawsuit",
    "investigation", "fraud", "scandal", "bankruptcy", "default", "debt",
    "layoff", "layoffs", "shutdown", "recall", "negative", "cut", "cuts",
    "fear", "uncertainty", "volatility", "sell-off", "selloff",
}

INTENSITY_MODIFIERS = {
    "very": 1.5, "extremely": 2.0, "slightly": 0.5, "significantly": 1.8,
    "massive": 2.0, "huge": 1.8, "major": 1.5, "minor": 0.5,
}


class SentimentStrategy:
    """
    News sentiment analysis strategy.

    Signals:
      BUY  when aggregate sentiment is positive
      SELL when aggregate sentiment is negative
    """

    name = "sentiment"

    def __init__(self, decay_hours: float = 24.0):
        self.decay_hours = decay_hours
        self.sentiment_cache: Dict[str, float] = {}

    def generate_signal(
        self, symbol: str, articles: List[Dict]
    ) -> Signal:
        """
        Score sentiment from a list of news articles.

        Each article dict should have keys: 'title', 'description'.
        """
        if not articles:
            return Signal(
                symbol, "HOLD", 0.0, self.name, "No news data available"
            )

        scores = []
        for article in articles:
            text = (
                article.get("title", "") + " " + article.get("description", "")
            )
            score = self._score_text(text)
            scores.append(score)

        if not scores:
            return Signal(symbol, "HOLD", 0.0, self.name, "No scoreable news")

        # Aggregate: recent articles weighted higher
        # (assumes articles are sorted newest first)
        weights = np.array(
            [0.95**i for i in range(len(scores))]
        )
        weights /= weights.sum()
        composite = float(np.dot(scores, weights))
        composite = np.clip(composite, -1.0, 1.0)

        self.sentiment_cache[symbol] = composite

        # ── Decision ─────────────────────────────────────────────────
        n_articles = len(articles)
        if composite > 0.2:
            action = "BUY"
            reason = (
                f"Positive sentiment ({n_articles} articles): "
                f"score={composite:.3f}"
            )
        elif composite < -0.2:
            action = "SELL"
            reason = (
                f"Negative sentiment ({n_articles} articles): "
                f"score={composite:.3f}"
            )
        else:
            action = "HOLD"
            reason = f"Neutral sentiment: score={composite:.3f}"

        return Signal(symbol, action, composite, self.name, reason)

    def _score_text(self, text: str) -> float:
        """Score a single piece of text. Returns -1 to +1."""
        if not text:
            return 0.0

        words = re.findall(r"\w+", text.lower())
        score = 0.0
        modifier = 1.0

        for word in words:
            if word in INTENSITY_MODIFIERS:
                modifier = INTENSITY_MODIFIERS[word]
                continue

            if word in POSITIVE_WORDS:
                score += 1.0 * modifier
            elif word in NEGATIVE_WORDS:
                score -= 1.0 * modifier

            modifier = 1.0  # reset after use

        # Normalize by text length to avoid long-article bias
        max_score = max(len(words) * 0.1, 1.0)
        return np.clip(score / max_score, -1.0, 1.0)

    def get_cached_sentiment(self, symbol: str) -> float:
        """Return last computed sentiment for a symbol."""
        return self.sentiment_cache.get(symbol, 0.0)
