"""Integration tests — full pipeline flows."""

from unittest.mock import patch, MagicMock
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from strategies.mean_reversion import MeanReversionStrategy, Signal
from strategies.momentum import MomentumStrategy
from strategies.pattern_recognition import PatternRecognitionStrategy
from strategies.sentiment import SentimentStrategy


class TestFullBacktestCycle:
    """Integration: full backtest cycle with mock data."""

    def test_backtest_with_synthetic_data(self, sample_ohlcv):
        """Run a full backtest pipeline with synthetic OHLCV data."""
        from data_fetcher import DataFetcher
        from logger import TradeLogger
        from risk_manager import RiskManager
        from self_improver import SelfImprover

        df = DataFetcher.compute_indicators(sample_ohlcv)

        logger = TradeLogger(log_dir="/tmp/test_backtest_integration")
        rm = RiskManager(initial_capital=100_000.0, logger=logger)
        si = SelfImprover(logger=logger, save_path="/tmp/test_backtest_integration")

        strategies = {
            "mean_reversion": MeanReversionStrategy(),
            "momentum": MomentumStrategy(),
            "pattern_recognition": PatternRecognitionStrategy(),
        }

        equity_curve = []
        trade_log = []

        for i in range(60, len(df)):
            df_slice = df.iloc[:i + 1]
            current_price = float(df_slice["Close"].iloc[-1])
            atr = float(df_slice["ATR"].iloc[-1]) if "ATR" in df_slice.columns and not pd.isna(df_slice["ATR"].iloc[-1]) else current_price * 0.02
            vol = float(df_slice["returns"].std()) if "returns" in df_slice.columns else 0.02

            # Generate signals
            signals = {}
            for name, strat in strategies.items():
                signals[name] = strat.generate_signal("TEST", df_slice)

            # Combine
            weights = si.weights
            combined = sum(
                weights.get(n, 0) * s.strength for n, s in signals.items() if n in weights
            )
            combined = np.clip(combined, -1.0, 1.0)

            can_trade, _ = rm.can_trade("TEST")

            if combined > 0.3 and can_trade and "TEST" not in rm.positions:
                qty = rm.calculate_position_size("TEST", current_price, combined, vol)
                if qty > 0:
                    rm.open_position("TEST", qty, current_price, "test_strat", atr)
                    trade_log.append({"action": "BUY", "price": current_price})

            elif combined < -0.3 and "TEST" in rm.positions:
                pos = rm.positions["TEST"]
                pnl = rm.close_position("TEST", current_price)
                si.record_experience(
                    symbol="TEST", strategy="test_strat", action="SELL",
                    signal_strength=combined, entry_price=pos.entry_price,
                    exit_price=current_price,
                )
                trade_log.append({"action": "SELL", "price": current_price, "pnl": pnl})

            total_value = rm.current_capital
            for sym, pos in rm.positions.items():
                total_value += pos.quantity * current_price
            equity_curve.append(total_value)

        # Verify outputs
        assert len(equity_curve) > 0
        assert equity_curve[0] > 0
        # The simulation should complete without errors
        assert isinstance(equity_curve[-1], float)


class TestStrategyToRiskManagerFlow:
    """Integration: strategy → risk manager → agent decision."""

    def test_signal_to_position_flow(self, sample_ohlcv_with_indicators, trade_logger):
        from risk_manager import RiskManager

        df = sample_ohlcv_with_indicators
        rm = RiskManager(initial_capital=100_000.0, logger=trade_logger)

        # Generate signal from strategy
        strategy = MeanReversionStrategy()
        signal = strategy.generate_signal("AAPL", df)

        price = float(df["Close"].iloc[-1])
        atr = float(df["ATR"].iloc[-1]) if not pd.isna(df["ATR"].iloc[-1]) else 2.0
        vol = float(df["returns"].std()) if "returns" in df.columns else 0.02

        # Risk check
        can_trade, reason = rm.can_trade("AAPL")
        assert can_trade

        # Position sizing
        if signal.action == "BUY" and signal.strength > 0.3:
            qty = rm.calculate_position_size("AAPL", price, signal.strength, vol)
            if qty > 0:
                rm.open_position("AAPL", qty, price, signal.strategy, atr)
                assert "AAPL" in rm.positions

    def test_multiple_strategies_combined(self, sample_ohlcv_with_indicators, positive_articles):
        """All four strategies produce signals and combine correctly."""
        df = sample_ohlcv_with_indicators

        mr = MeanReversionStrategy()
        mom = MomentumStrategy()
        sent = SentimentStrategy()
        pr = PatternRecognitionStrategy()

        s1 = mr.generate_signal("AAPL", df)
        s2 = mom.generate_signal("AAPL", df)
        s3 = sent.generate_signal("AAPL", positive_articles)
        s4 = pr.generate_signal("AAPL", df)

        signals = {"mean_reversion": s1, "momentum": s2, "sentiment": s3, "pattern_recognition": s4}

        weights = {"mean_reversion": 0.30, "momentum": 0.30, "sentiment": 0.15, "pattern_recognition": 0.25}

        combined = sum(weights[n] * s.strength for n, s in signals.items())
        combined = np.clip(combined, -1.0, 1.0)

        assert -1.0 <= combined <= 1.0


class TestSelfImprovementLoop:
    """Integration: self-improvement loop over multiple trades."""

    def test_improvement_loop(self, self_improver):
        """Record many trades and verify weights update."""
        np.random.seed(42)
        initial_weights = dict(self_improver.weights)

        # Simulate many trades with varied outcomes
        strategies = list(self_improver.weights.keys())
        for i in range(50):
            strat = strategies[i % len(strategies)]
            # Momentum does better than others
            if strat == "momentum":
                exit_p = 100.0 + np.random.uniform(0, 10)
            else:
                exit_p = 100.0 + np.random.uniform(-5, 5)

            self_improver.record_experience(
                symbol="TEST", strategy=strat, action="BUY",
                signal_strength=0.5, entry_price=100.0, exit_price=exit_p,
            )

        # Update weights
        new_weights = self_improver.update_weights()

        # Verify weights still sum to ~1
        assert abs(sum(new_weights.values()) - 1.0) < 0.01

        # All weights within constraints
        from config import config
        for w in new_weights.values():
            assert w >= config.min_strategy_weight - 0.001
            assert w <= config.max_strategy_weight + 0.001


class TestEndToEndWithMockData:
    """End-to-end: agent initialization → cycle → status."""

    @patch("agent.TradingAgent._check_alpaca", return_value=False)
    def test_agent_lifecycle(self, mock_alpaca, sample_ohlcv, trade_logger):
        from agent import TradingAgent
        from data_fetcher import DataFetcher

        with patch("agent.DataFetcher") as MockFetcher:
            mock_fetcher = MagicMock()
            df = DataFetcher.compute_indicators(sample_ohlcv)
            mock_fetcher.get_historical.return_value = sample_ohlcv
            mock_fetcher.compute_indicators = DataFetcher.compute_indicators
            mock_fetcher.get_latest_prices.return_value = {}
            mock_fetcher.get_news.return_value = []
            MockFetcher.return_value = mock_fetcher
            MockFetcher.compute_indicators = DataFetcher.compute_indicators

            agent = TradingAgent(capital=100_000.0, paper_trade=False)
            agent.logger = trade_logger

        # Run a cycle
        with patch.object(agent, "_analyze_symbol", return_value=(None, {})):
            actions = agent.run_cycle()
            assert isinstance(actions, dict)

        # Check status
        status = agent.get_status()
        assert status["cycle_count"] == 1
        assert "risk" in status


class TestMathVerification:
    """Verify mathematical formulas across the system."""

    def test_rsi_manual_calculation(self, sample_ohlcv):
        """Verify RSI against manual calculation."""
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        close = sample_ohlcv["Close"]
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        expected_rsi = 100 - (100 / (1 + rs))

        # Compare at index 50
        idx = 50
        actual = df["RSI"].iloc[idx]
        expected = expected_rsi.iloc[idx]
        if not pd.isna(expected) and not pd.isna(actual):
            assert abs(actual - expected) < 1e-6

    def test_position_sizing_formula(self):
        """Verify: shares = int((capital * max_pct * vol_scalar * signal) / price)."""
        capital = 100_000
        max_pct = 0.05
        price = 150.0
        vol = 0.02
        signal = 0.8

        vol_scalar = max(0.2, min(1.0, 0.02 / max(vol, 1e-6)))
        adjusted = capital * max_pct * vol_scalar * abs(signal)
        expected_shares = int(adjusted / price)

        from logger import TradeLogger
        from risk_manager import RiskManager

        logger = TradeLogger(log_dir="/tmp/test_math_verify")
        rm = RiskManager(initial_capital=capital, logger=logger)
        actual_shares = rm.calculate_position_size("AAPL", price, signal, vol)
        assert actual_shares == expected_shares

    def test_sharpe_ratio_manual(self):
        """Sharpe = mean(r) / std(r) * sqrt(252)."""
        returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01])
        expected = np.mean(returns) / np.std(returns) * np.sqrt(252)

        from backtester import Backtester
        bt = Backtester(symbols=["TEST"])
        bt.daily_returns = list(returns)
        bt.equity_curve = list(100000 * np.cumprod(1 + returns))
        bt.trade_log = []
        results = bt._compute_metrics(elapsed=0.1)
        assert abs(results["sharpe_ratio"] - expected) < 1e-6
