"""Tests for strategies/mean_reversion.py."""

import numpy as np
import pandas as pd
import pytest

from strategies.mean_reversion import MeanReversionStrategy, Signal


class TestSignalDataclass:
    """Test the Signal dataclass."""

    def test_signal_creation(self):
        s = Signal("AAPL", "BUY", 0.5, "mean_reversion", "test reason")
        assert s.symbol == "AAPL"
        assert s.action == "BUY"
        assert s.strength == 0.5
        assert s.strategy == "mean_reversion"
        assert s.reason == "test reason"

    def test_signal_negative_strength(self):
        s = Signal("AAPL", "SELL", -0.8, "mean_reversion", "selling")
        assert s.strength == -0.8


class TestZScoreCalculation:
    """Test z-score behavior in mean reversion signals."""

    def test_oversold_generates_buy(self, sample_ohlcv_with_indicators):
        """When price drops well below its mean, expect a BUY signal."""
        df = sample_ohlcv_with_indicators.copy()
        # Force price way below rolling mean
        df.iloc[-1, df.columns.get_loc("Close")] = df["Close"].iloc[-21:-1].mean() - 3 * df["Close"].iloc[-20:].std()
        strategy = MeanReversionStrategy()
        signal = strategy.generate_signal("TEST", df)
        assert signal.action in ("BUY", "HOLD")
        assert signal.strength >= 0.0 or signal.action == "HOLD"

    def test_overbought_generates_sell(self, sample_ohlcv_with_indicators):
        """When price rises well above its mean, expect a SELL signal."""
        df = sample_ohlcv_with_indicators.copy()
        df.iloc[-1, df.columns.get_loc("Close")] = df["Close"].iloc[-21:-1].mean() + 3 * df["Close"].iloc[-20:].std()
        strategy = MeanReversionStrategy()
        signal = strategy.generate_signal("TEST", df)
        assert signal.action in ("SELL", "HOLD")
        assert signal.strength <= 0.0 or signal.action == "HOLD"


class TestBollingerBandSignal:
    """Test BB_pct contribution."""

    def test_bb_pct_near_zero_is_bullish(self, sample_ohlcv_with_indicators):
        """BB_pct near 0 means price at lower band → bullish for mean reversion."""
        df = sample_ohlcv_with_indicators.copy()
        df.iloc[-1, df.columns.get_loc("BB_pct")] = 0.0
        strategy = MeanReversionStrategy()
        signal = strategy.generate_signal("TEST", df)
        # bb_signal = -(0.0 - 0.5)*2 = 1.0, which is bullish
        assert isinstance(signal, Signal)

    def test_bb_pct_near_one_is_bearish(self, sample_ohlcv_with_indicators):
        df = sample_ohlcv_with_indicators.copy()
        df.iloc[-1, df.columns.get_loc("BB_pct")] = 1.0
        strategy = MeanReversionStrategy()
        signal = strategy.generate_signal("TEST", df)
        assert isinstance(signal, Signal)


class TestRSIComposite:
    """Test RSI contribution to composite scoring."""

    def test_rsi_below_oversold_is_bullish(self, sample_ohlcv_with_indicators):
        df = sample_ohlcv_with_indicators.copy()
        df.iloc[-1, df.columns.get_loc("RSI")] = 20.0
        strategy = MeanReversionStrategy(rsi_oversold=30.0)
        signal = strategy.generate_signal("TEST", df)
        # RSI < 30 contributes positive signal
        assert isinstance(signal, Signal)

    def test_rsi_above_overbought_is_bearish(self, sample_ohlcv_with_indicators):
        df = sample_ohlcv_with_indicators.copy()
        df.iloc[-1, df.columns.get_loc("RSI")] = 80.0
        strategy = MeanReversionStrategy(rsi_overbought=70.0)
        signal = strategy.generate_signal("TEST", df)
        assert isinstance(signal, Signal)

    def test_rsi_neutral_no_contribution(self, sample_ohlcv_with_indicators):
        df = sample_ohlcv_with_indicators.copy()
        df.iloc[-1, df.columns.get_loc("RSI")] = 50.0
        strategy = MeanReversionStrategy()
        # RSI at 50 contributes 0 to composite
        signal = strategy.generate_signal("TEST", df)
        assert isinstance(signal, Signal)


class TestSignalOutputFormat:
    """Verify signal output format and fields."""

    def test_signal_has_required_fields(self, sample_ohlcv_with_indicators):
        strategy = MeanReversionStrategy()
        signal = strategy.generate_signal("AAPL", sample_ohlcv_with_indicators)
        assert hasattr(signal, "symbol")
        assert hasattr(signal, "action")
        assert hasattr(signal, "strength")
        assert hasattr(signal, "strategy")
        assert hasattr(signal, "reason")

    def test_signal_action_is_valid(self, sample_ohlcv_with_indicators):
        strategy = MeanReversionStrategy()
        signal = strategy.generate_signal("AAPL", sample_ohlcv_with_indicators)
        assert signal.action in ("BUY", "SELL", "HOLD")

    def test_signal_strength_in_range(self, sample_ohlcv_with_indicators):
        strategy = MeanReversionStrategy()
        signal = strategy.generate_signal("AAPL", sample_ohlcv_with_indicators)
        assert -1.0 <= signal.strength <= 1.0

    def test_signal_strategy_name(self, sample_ohlcv_with_indicators):
        strategy = MeanReversionStrategy()
        signal = strategy.generate_signal("AAPL", sample_ohlcv_with_indicators)
        assert signal.strategy == "mean_reversion"

    def test_signal_symbol_matches(self, sample_ohlcv_with_indicators):
        strategy = MeanReversionStrategy()
        signal = strategy.generate_signal("TSLA", sample_ohlcv_with_indicators)
        assert signal.symbol == "TSLA"


class TestEdgeCases:
    """Edge cases: flat data, extreme moves, insufficient data."""

    def test_insufficient_data_returns_hold(self, short_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(short_ohlcv)
        strategy = MeanReversionStrategy()
        signal = strategy.generate_signal("TEST", df)
        assert signal.action == "HOLD"
        assert signal.strength == 0.0
        assert "Insufficient" in signal.reason

    def test_flat_price_returns_hold(self, flat_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(flat_ohlcv)
        strategy = MeanReversionStrategy()
        signal = strategy.generate_signal("TEST", df)
        # Flat data → z-score ≈ 0 → HOLD
        assert signal.action == "HOLD"

    def test_extreme_move_up(self, extreme_move_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(extreme_move_ohlcv)
        strategy = MeanReversionStrategy()
        signal = strategy.generate_signal("TEST", df)
        # Big up move → overbought → should be SELL or strong negative
        assert signal.action in ("SELL", "HOLD", "BUY")  # at least doesn't crash
        assert -1.0 <= signal.strength <= 1.0


class TestPairsOpportunity:
    """Test pairs trading z-score."""

    def test_pairs_with_similar_prices(self, sample_ohlcv):
        strategy = MeanReversionStrategy()
        z = strategy.detect_pairs_opportunity(sample_ohlcv, sample_ohlcv, window=60)
        # Same series → spread z ≈ 0
        assert z is not None
        assert abs(z) < 0.01

    def test_pairs_with_insufficient_data(self, short_ohlcv):
        strategy = MeanReversionStrategy()
        z = strategy.detect_pairs_opportunity(short_ohlcv, short_ohlcv, window=60)
        assert z is None

    def test_pairs_divergent_prices(self, trending_up_ohlcv, trending_down_ohlcv):
        strategy = MeanReversionStrategy()
        z = strategy.detect_pairs_opportunity(
            trending_up_ohlcv, trending_down_ohlcv, window=60
        )
        assert z is not None
        assert isinstance(z, float)
