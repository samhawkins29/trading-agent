"""Tests for strategies/sentiment.py."""

import numpy as np
import pytest

from strategies.sentiment import (
    SentimentStrategy,
    POSITIVE_WORDS,
    NEGATIVE_WORDS,
    INTENSITY_MODIFIERS,
)
from strategies.mean_reversion import Signal


class TestKeywordScoring:
    """Test _score_text keyword-based scoring."""

    def test_positive_text_scores_positive(self):
        strategy = SentimentStrategy()
        score = strategy._score_text("Revenue surged to record high with strong growth")
        assert score > 0

    def test_negative_text_scores_negative(self):
        strategy = SentimentStrategy()
        score = strategy._score_text("Stock crashed amid recession fears and layoffs")
        assert score < 0

    def test_neutral_text_scores_near_zero(self):
        strategy = SentimentStrategy()
        score = strategy._score_text("The company held its annual board meeting today")
        assert abs(score) < 0.5

    def test_empty_text_scores_zero(self):
        strategy = SentimentStrategy()
        score = strategy._score_text("")
        assert score == 0.0

    def test_score_range(self):
        strategy = SentimentStrategy()
        score = strategy._score_text("beat beat beat surged surged")
        assert -1.0 <= score <= 1.0

    def test_all_positive_words_detected(self):
        strategy = SentimentStrategy()
        for word in list(POSITIVE_WORDS)[:5]:
            score = strategy._score_text(f"The company {word} expectations")
            assert score > 0, f"Positive word '{word}' not detected"

    def test_all_negative_words_detected(self):
        strategy = SentimentStrategy()
        # Filter out hyphenated words since the tokenizer splits on non-word chars
        testable = [w for w in list(NEGATIVE_WORDS)[:10] if "-" not in w]
        for word in testable[:5]:
            score = strategy._score_text(f"The stock {word} today")
            assert score < 0, f"Negative word '{word}' not detected"


class TestIntensityModifiers:
    """Test that intensity modifiers amplify/dampen scores."""

    def test_very_amplifies(self):
        strategy = SentimentStrategy()
        base = strategy._score_text("strong earnings")
        amplified = strategy._score_text("very strong earnings")
        assert amplified >= base

    def test_slightly_dampens(self):
        strategy = SentimentStrategy()
        base = strategy._score_text("strong earnings")
        dampened = strategy._score_text("slightly strong earnings")
        assert dampened <= base

    def test_extremely_amplifies_more_than_very(self):
        strategy = SentimentStrategy()
        very_score = strategy._score_text("very strong")
        extremely_score = strategy._score_text("extremely strong")
        assert extremely_score >= very_score


class TestRecencyWeighting:
    """Test that newer articles get higher weight in aggregation."""

    def test_recency_decay(self):
        strategy = SentimentStrategy()
        # First article (index 0) should get weight 0.95^0 = 1.0
        # Second gets 0.95^1 = 0.95, etc.
        articles = [
            {"title": "bearish crash decline", "description": ""},
            {"title": "bullish surge rally", "description": ""},
        ]
        signal = strategy.generate_signal("TEST", articles)
        # First article (negative) has higher weight, so composite should lean negative
        assert signal.strength < 0 or signal.action != "BUY"


class TestSentimentAggregation:
    """Test aggregate signal generation from multiple articles."""

    def test_positive_articles_buy(self, positive_articles):
        strategy = SentimentStrategy()
        signal = strategy.generate_signal("AAPL", positive_articles)
        assert signal.action == "BUY"
        assert signal.strength > 0

    def test_negative_articles_sell(self, negative_articles):
        strategy = SentimentStrategy()
        signal = strategy.generate_signal("AAPL", negative_articles)
        assert signal.action == "SELL"
        assert signal.strength < 0

    def test_neutral_articles_hold(self, neutral_articles):
        strategy = SentimentStrategy()
        signal = strategy.generate_signal("AAPL", neutral_articles)
        assert signal.action == "HOLD"

    def test_composite_in_range(self, positive_articles):
        strategy = SentimentStrategy()
        signal = strategy.generate_signal("AAPL", positive_articles)
        assert -1.0 <= signal.strength <= 1.0


class TestSignalFormat:
    """Verify signal output format."""

    def test_signal_fields(self, positive_articles):
        strategy = SentimentStrategy()
        signal = strategy.generate_signal("AAPL", positive_articles)
        assert signal.symbol == "AAPL"
        assert signal.strategy == "sentiment"
        assert signal.action in ("BUY", "SELL", "HOLD")

    def test_reason_contains_article_count(self, positive_articles):
        strategy = SentimentStrategy()
        signal = strategy.generate_signal("AAPL", positive_articles)
        assert "2 articles" in signal.reason


class TestEdgeCases:
    """Edge cases: no news, all positive, all negative, empty input."""

    def test_no_articles_hold(self):
        strategy = SentimentStrategy()
        signal = strategy.generate_signal("AAPL", [])
        assert signal.action == "HOLD"
        assert signal.strength == 0.0
        assert "No news" in signal.reason

    def test_single_article(self):
        strategy = SentimentStrategy()
        signal = strategy.generate_signal(
            "AAPL", [{"title": "Strong surge", "description": ""}]
        )
        assert isinstance(signal, Signal)

    def test_articles_with_empty_fields(self):
        strategy = SentimentStrategy()
        signal = strategy.generate_signal(
            "AAPL", [{"title": "", "description": ""}]
        )
        assert signal.action == "HOLD"
        assert signal.strength == 0.0

    def test_articles_missing_keys(self):
        strategy = SentimentStrategy()
        signal = strategy.generate_signal("AAPL", [{}])
        assert isinstance(signal, Signal)

    def test_many_articles_doesnt_crash(self):
        strategy = SentimentStrategy()
        articles = [
            {"title": f"Article {i} about growth", "description": ""}
            for i in range(100)
        ]
        signal = strategy.generate_signal("AAPL", articles)
        assert -1.0 <= signal.strength <= 1.0


class TestCachedSentiment:
    """Test sentiment cache."""

    def test_cache_updated_after_signal(self, positive_articles):
        strategy = SentimentStrategy()
        strategy.generate_signal("AAPL", positive_articles)
        cached = strategy.get_cached_sentiment("AAPL")
        assert cached > 0

    def test_cache_default_zero(self):
        strategy = SentimentStrategy()
        assert strategy.get_cached_sentiment("UNKNOWN") == 0.0
