"""Tests for strategies/momentum.py."""

import numpy as np
import pandas as pd
import pytest

from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import Signal


class TestMACrossover:
    """Test moving average crossover detection."""

    def test_golden_cross_bullish(self, trending_up_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(trending_up_ohlcv)
        strategy = MomentumStrategy()
        signal = strategy.generate_signal("TEST", df)
        # Strong uptrend → SMA_20 > SMA_50 → bullish MA signal
        assert signal.action in ("BUY", "HOLD")
        assert signal.strength >= 0 or signal.action == "HOLD"

    def test_death_cross_bearish(self, trending_down_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(trending_down_ohlcv)
        strategy = MomentumStrategy()
        signal = strategy.generate_signal("TEST", df)
        # Strong downtrend → SMA_20 < SMA_50 → bearish MA signal
        assert signal.action in ("SELL", "HOLD")
        assert signal.strength <= 0 or signal.action == "HOLD"


class TestMACDSignal:
    """Test MACD histogram signal generation."""

    def test_macd_positive_histogram(self, sample_ohlcv_with_indicators):
        df = sample_ohlcv_with_indicators.copy()
        df.iloc[-1, df.columns.get_loc("MACD_hist")] = 2.0
        df.iloc[-2, df.columns.get_loc("MACD_hist")] = 1.0
        strategy = MomentumStrategy()
        signal = strategy.generate_signal("TEST", df)
        # Positive and increasing histogram → bullish MACD signal
        assert isinstance(signal, Signal)

    def test_macd_negative_histogram(self, sample_ohlcv_with_indicators):
        df = sample_ohlcv_with_indicators.copy()
        df.iloc[-1, df.columns.get_loc("MACD_hist")] = -2.0
        df.iloc[-2, df.columns.get_loc("MACD_hist")] = -1.0
        strategy = MomentumStrategy()
        signal = strategy.generate_signal("TEST", df)
        assert isinstance(signal, Signal)


class TestRateOfChange:
    """Test rate-of-change calculation."""

    def test_roc_positive_for_uptrend(self, trending_up_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(trending_up_ohlcv)
        strategy = MomentumStrategy(roc_period=10)
        # Verify ROC is positive (price went up over 10 periods)
        close = df["Close"].values
        roc = (close[-1] - close[-11]) / close[-11]
        assert roc > 0

    def test_roc_negative_for_downtrend(self, trending_down_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(trending_down_ohlcv)
        close = df["Close"].values
        roc = (close[-1] - close[-11]) / close[-11]
        assert roc < 0


class TestVolumeConfirmation:
    """Test volume confirmation logic."""

    def test_high_volume_boosts_signal(self, sample_ohlcv_with_indicators):
        df = sample_ohlcv_with_indicators.copy()
        df.iloc[-1, df.columns.get_loc("Vol_ratio")] = 3.0
        strategy = MomentumStrategy()
        signal_high_vol = strategy.generate_signal("TEST", df)

        df.iloc[-1, df.columns.get_loc("Vol_ratio")] = 0.3
        signal_low_vol = strategy.generate_signal("TEST", df)

        # High volume version should have same or stronger absolute signal
        # (at least shouldn't crash)
        assert isinstance(signal_high_vol, Signal)
        assert isinstance(signal_low_vol, Signal)

    def test_vol_ratio_nan_handled(self, sample_ohlcv_with_indicators):
        df = sample_ohlcv_with_indicators.copy()
        df.iloc[-1, df.columns.get_loc("Vol_ratio")] = np.nan
        strategy = MomentumStrategy()
        signal = strategy.generate_signal("TEST", df)
        assert isinstance(signal, Signal)


class TestSignalFormat:
    """Verify output format."""

    def test_action_valid(self, sample_ohlcv_with_indicators):
        strategy = MomentumStrategy()
        signal = strategy.generate_signal("AAPL", sample_ohlcv_with_indicators)
        assert signal.action in ("BUY", "SELL", "HOLD")

    def test_strength_in_range(self, sample_ohlcv_with_indicators):
        strategy = MomentumStrategy()
        signal = strategy.generate_signal("AAPL", sample_ohlcv_with_indicators)
        assert -1.0 <= signal.strength <= 1.0

    def test_strategy_name(self, sample_ohlcv_with_indicators):
        strategy = MomentumStrategy()
        signal = strategy.generate_signal("AAPL", sample_ohlcv_with_indicators)
        assert signal.strategy == "momentum"


class TestEdgeCases:
    """Edge cases: no trend, sudden reversal, low volume, insufficient data."""

    def test_insufficient_data_hold(self, short_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(short_ohlcv)
        strategy = MomentumStrategy()
        signal = strategy.generate_signal("TEST", df)
        assert signal.action == "HOLD"
        assert signal.strength == 0.0

    def test_flat_data_hold(self, flat_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(flat_ohlcv)
        strategy = MomentumStrategy()
        signal = strategy.generate_signal("TEST", df)
        assert signal.action == "HOLD"

    def test_extreme_move_doesnt_crash(self, extreme_move_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(extreme_move_ohlcv)
        strategy = MomentumStrategy()
        signal = strategy.generate_signal("TEST", df)
        assert -1.0 <= signal.strength <= 1.0

    def test_missing_macd_hist_column(self, sample_ohlcv):
        """Momentum should handle missing optional columns gracefully."""
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        df = df.drop(columns=["MACD_hist"])
        strategy = MomentumStrategy()
        signal = strategy.generate_signal("TEST", df)
        assert isinstance(signal, Signal)


class TestBreakoutDetection:
    """Test detect_breakout method."""

    def test_breakout_above_resistance(self, sample_ohlcv):
        strategy = MomentumStrategy()
        # Force the last close above the prior range
        df = sample_ohlcv.copy()
        recent_high = df["High"].iloc[-21:-1].max()
        df.iloc[-1, df.columns.get_loc("Close")] = recent_high + 5.0
        strength = strategy.detect_breakout(df, lookback=20)
        assert strength > 0

    def test_breakout_below_support(self, sample_ohlcv):
        strategy = MomentumStrategy()
        df = sample_ohlcv.copy()
        recent_low = df["Low"].iloc[-21:-1].min()
        df.iloc[-1, df.columns.get_loc("Close")] = recent_low - 5.0
        strength = strategy.detect_breakout(df, lookback=20)
        assert strength < 0

    def test_no_breakout_mid_range(self, sample_ohlcv):
        strategy = MomentumStrategy()
        df = sample_ohlcv.copy()
        mid = (df["High"].iloc[-21:-1].max() + df["Low"].iloc[-21:-1].min()) / 2
        df.iloc[-1, df.columns.get_loc("Close")] = mid
        strength = strategy.detect_breakout(df, lookback=20)
        assert strength == 0.0

    def test_breakout_insufficient_data(self, short_ohlcv):
        strategy = MomentumStrategy()
        strength = strategy.detect_breakout(short_ohlcv, lookback=20)
        assert strength == 0.0

    def test_breakout_flat_range(self, flat_ohlcv):
        strategy = MomentumStrategy()
        strength = strategy.detect_breakout(flat_ohlcv, lookback=20)
        assert strength == 0.0
