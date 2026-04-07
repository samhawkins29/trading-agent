"""Tests for strategies/pattern_recognition.py."""

import numpy as np
import pandas as pd
import pytest

from strategies.pattern_recognition import (
    PatternRecognitionStrategy,
    MarketRegime,
)
from strategies.mean_reversion import Signal


class TestCandlestickPatterns:
    """Test candlestick pattern detection."""

    def test_doji_detection(self):
        """Doji: body is tiny relative to total range."""
        strategy = PatternRecognitionStrategy()
        n = 50
        dates = pd.bdate_range("2024-01-02", periods=n)
        df = pd.DataFrame(
            {
                "Open": [100.0] * n,
                "High": [102.0] * n,
                "Low": [98.0] * n,
                "Close": [100.01] * n,  # tiny body
                "Volume": [1e6] * n,
            },
            index=dates,
        )
        sig = strategy._candlestick_signal(df)
        # Doji should reduce conviction → signal should be moderate
        assert isinstance(sig, float)
        assert -1.0 <= sig <= 1.0

    def test_hammer_detection(self):
        """Hammer: long lower shadow, small body, little upper shadow."""
        strategy = PatternRecognitionStrategy()
        n = 10
        dates = pd.bdate_range("2024-01-02", periods=n)
        # Build a downtrend then a hammer
        closes = list(range(110, 100, -1))
        opens = [c + 0.5 for c in closes]
        highs = [max(o, c) + 0.1 for o, c in zip(opens, closes)]
        lows = [min(o, c) - 0.2 for o, c in zip(opens, closes)]
        # Make the last candle a hammer
        opens[-1] = 101.0
        closes[-1] = 101.5
        highs[-1] = 101.7
        lows[-1] = 98.0  # long lower shadow
        df = pd.DataFrame(
            {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": [1e6] * n},
            index=dates,
        )
        sig = strategy._candlestick_signal(df)
        assert isinstance(sig, float)

    def test_bullish_engulfing(self):
        """Bullish engulfing: prev red, current green & bigger."""
        strategy = PatternRecognitionStrategy()
        n = 5
        dates = pd.bdate_range("2024-01-02", periods=n)
        opens = [100, 101, 102, 101, 99]   # prev candle: O=101, C=99 (red)
        closes = [101, 102, 101, 99, 103]   # curr candle: O=99, C=103 (green, bigger)
        highs = [102, 103, 103, 102, 104]
        lows = [99, 100, 100, 98, 98]
        df = pd.DataFrame(
            {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": [1e6] * n},
            index=dates,
        )
        sig = strategy._candlestick_signal(df)
        assert sig > 0  # bullish engulfing → positive

    def test_bearish_engulfing(self):
        """Bearish engulfing: prev green, current red & bigger."""
        strategy = PatternRecognitionStrategy()
        n = 5
        dates = pd.bdate_range("2024-01-02", periods=n)
        # Prev candle: O=100, C=102 (green, body=2)
        # Curr candle: O=105, C=99 (red, body=6, much bigger than prev)
        # Total range = 106-98 = 8, body=6 → body/range = 0.75 > 0.1, so not doji
        opens = [100, 99, 100, 100, 105]
        closes = [101, 100, 101, 102, 99]
        highs = [102, 101, 102, 103, 106]
        lows = [99, 98, 99, 99, 98]
        df = pd.DataFrame(
            {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": [1e6] * n},
            index=dates,
        )
        sig = strategy._candlestick_signal(df)
        assert sig < 0  # bearish engulfing → negative

    def test_too_few_candles(self):
        strategy = PatternRecognitionStrategy()
        dates = pd.bdate_range("2024-01-02", periods=2)
        df = pd.DataFrame(
            {"Open": [100, 101], "High": [101, 102], "Low": [99, 100], "Close": [100.5, 101.5], "Volume": [1e6, 1e6]},
            index=dates,
        )
        sig = strategy._candlestick_signal(df)
        assert isinstance(sig, float)


class TestSupportResistance:
    """Test support/resistance identification."""

    def test_near_support_bullish(self, sample_ohlcv):
        strategy = PatternRecognitionStrategy()
        df = sample_ohlcv.copy()
        # Set close near the low of the range
        recent_low = df["Low"].iloc[-20:].min()
        df.iloc[-1, df.columns.get_loc("Close")] = recent_low + 0.01
        sig = strategy._support_resistance_signal(df, window=20)
        assert sig > 0  # near support → bullish

    def test_near_resistance_bearish(self, sample_ohlcv):
        strategy = PatternRecognitionStrategy()
        df = sample_ohlcv.copy()
        recent_high = df["High"].iloc[-20:].max()
        df.iloc[-1, df.columns.get_loc("Close")] = recent_high - 0.01
        sig = strategy._support_resistance_signal(df, window=20)
        assert sig < 0  # near resistance → bearish

    def test_mid_range_neutral(self, sample_ohlcv):
        strategy = PatternRecognitionStrategy()
        df = sample_ohlcv.copy()
        mid = (df["High"].iloc[-20:].max() + df["Low"].iloc[-20:].min()) / 2
        df.iloc[-1, df.columns.get_loc("Close")] = mid
        sig = strategy._support_resistance_signal(df, window=20)
        assert sig == 0.0

    def test_insufficient_data(self, short_ohlcv):
        strategy = PatternRecognitionStrategy()
        sig = strategy._support_resistance_signal(short_ohlcv, window=20)
        assert sig == 0.0

    def test_flat_range_zero(self, flat_ohlcv):
        strategy = PatternRecognitionStrategy()
        sig = strategy._support_resistance_signal(flat_ohlcv, window=20)
        assert sig == 0.0


class TestRegimeClassification:
    """Test market regime detection."""

    def test_trending_up(self, trending_up_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(trending_up_ohlcv)
        strategy = PatternRecognitionStrategy()
        regime = strategy._detect_regime(df)
        assert regime == MarketRegime.TRENDING_UP

    def test_trending_down(self, trending_down_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(trending_down_ohlcv)
        strategy = PatternRecognitionStrategy()
        regime = strategy._detect_regime(df)
        assert regime == MarketRegime.TRENDING_DOWN

    def test_ranging_flat(self, flat_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(flat_ohlcv)
        strategy = PatternRecognitionStrategy()
        regime = strategy._detect_regime(df)
        assert regime == MarketRegime.RANGING

    def test_regime_signal_trending_up(self, trending_up_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(trending_up_ohlcv)
        strategy = PatternRecognitionStrategy()
        sig = strategy._regime_signal(df, MarketRegime.TRENDING_UP)
        assert sig == 0.5

    def test_regime_signal_trending_down(self, trending_down_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(trending_down_ohlcv)
        strategy = PatternRecognitionStrategy()
        sig = strategy._regime_signal(df, MarketRegime.TRENDING_DOWN)
        assert sig == -0.5

    def test_regime_signal_ranging(self, flat_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(flat_ohlcv)
        strategy = PatternRecognitionStrategy()
        sig = strategy._regime_signal(df, MarketRegime.RANGING)
        assert sig == 0.0


class TestVolatilitySignal:
    """Test volatility signal."""

    def test_low_atr_slight_bullish(self, sample_ohlcv_with_indicators):
        df = sample_ohlcv_with_indicators.copy()
        strategy = PatternRecognitionStrategy()
        # Force low ATR relative to mean
        mean_atr = df["ATR"].iloc[-20:].mean()
        df.iloc[-1, df.columns.get_loc("ATR")] = mean_atr * 0.5
        sig = strategy._volatility_signal(df)
        assert sig == pytest.approx(0.2)

    def test_missing_atr_returns_zero(self, sample_ohlcv):
        strategy = PatternRecognitionStrategy()
        sig = strategy._volatility_signal(sample_ohlcv)
        assert sig == 0.0


class TestGenerateSignalFull:
    """Test full generate_signal pipeline."""

    def test_normal_data(self, sample_ohlcv_with_indicators):
        strategy = PatternRecognitionStrategy()
        signal = strategy.generate_signal("AAPL", sample_ohlcv_with_indicators)
        assert signal.action in ("BUY", "SELL", "HOLD")
        assert -1.0 <= signal.strength <= 1.0
        assert signal.strategy == "pattern_recognition"

    def test_insufficient_data(self, short_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(short_ohlcv)
        strategy = PatternRecognitionStrategy()
        signal = strategy.generate_signal("TEST", df)
        assert signal.action == "HOLD"
        assert signal.strength == 0.0

    def test_flat_data(self, flat_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(flat_ohlcv)
        strategy = PatternRecognitionStrategy()
        signal = strategy.generate_signal("TEST", df)
        assert signal.action == "HOLD"

    def test_gap_data(self, extreme_move_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(extreme_move_ohlcv)
        strategy = PatternRecognitionStrategy()
        signal = strategy.generate_signal("TEST", df)
        assert -1.0 <= signal.strength <= 1.0


class TestGetKeyLevels:
    """Test support/resistance key level retrieval."""

    def test_key_levels_returns_tuple(self, sample_ohlcv):
        strategy = PatternRecognitionStrategy()
        support, resistance = strategy.get_key_levels(sample_ohlcv)
        assert isinstance(support, float)
        assert isinstance(resistance, float)
        assert support <= resistance

    def test_key_levels_short_data(self, short_ohlcv):
        strategy = PatternRecognitionStrategy()
        support, resistance = strategy.get_key_levels(short_ohlcv, window=50)
        assert support <= resistance
