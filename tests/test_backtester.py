"""Tests for backtester.py — simulation, metrics, trade logging."""

import numpy as np
import pandas as pd
import pytest

from backtester import Backtester


class TestComputeMetrics:
    """Test Sharpe ratio, max drawdown, win rate calculations."""

    def test_sharpe_ratio_formula(self):
        """Verify Sharpe = mean(returns)/std(returns) * sqrt(252)."""
        returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01, 0.008])
        expected_sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)

        bt = Backtester(symbols=["TEST"])
        bt.daily_returns = list(returns)
        bt.equity_curve = list(100000 * np.cumprod(1 + returns))
        bt.trade_log = []

        results = bt._compute_metrics(elapsed=0.1)
        assert abs(results["sharpe_ratio"] - expected_sharpe) < 1e-6

    def test_max_drawdown_calculation(self):
        """Max drawdown from known equity curve."""
        # Peak at 110k, valley at 90k → drawdown = (110-90)/110 ≈ 18.18%
        equity = [100000, 105000, 110000, 100000, 90000, 95000]
        bt = Backtester(symbols=["TEST"])
        bt.equity_curve = equity
        bt.daily_returns = [0] * len(equity)
        bt.trade_log = []

        results = bt._compute_metrics(elapsed=0.1)
        expected_dd = (110000 - 90000) / 110000
        assert abs(results["max_drawdown"] - expected_dd) < 1e-6

    def test_win_rate_all_winning(self):
        bt = Backtester(symbols=["TEST"])
        bt.equity_curve = [100000]
        bt.daily_returns = [0]
        bt.trade_log = [
            {"action": "SELL", "pnl": 100},
            {"action": "SELL", "pnl": 200},
            {"action": "SELL", "pnl": 50},
        ]
        results = bt._compute_metrics(elapsed=0.1)
        assert results["win_rate"] == 1.0

    def test_win_rate_all_losing(self):
        bt = Backtester(symbols=["TEST"])
        bt.equity_curve = [100000]
        bt.daily_returns = [0]
        bt.trade_log = [
            {"action": "SELL", "pnl": -100},
            {"action": "SELL", "pnl": -200},
        ]
        results = bt._compute_metrics(elapsed=0.1)
        assert results["win_rate"] == 0.0

    def test_win_rate_mixed(self):
        bt = Backtester(symbols=["TEST"])
        bt.equity_curve = [100000]
        bt.daily_returns = [0]
        bt.trade_log = [
            {"action": "SELL", "pnl": 100},
            {"action": "SELL", "pnl": -50},
            {"action": "BUY", "quantity": 10},  # buys don't count for win rate
        ]
        results = bt._compute_metrics(elapsed=0.1)
        assert results["win_rate"] == 0.5

    def test_no_trades_metrics(self):
        bt = Backtester(symbols=["TEST"])
        bt.equity_curve = [100000]
        bt.daily_returns = [0]
        bt.trade_log = []
        results = bt._compute_metrics(elapsed=0.1)
        assert results["total_trades"] == 0
        assert results["win_rate"] == 0


class TestTradeLogging:
    """Test that trades are properly logged."""

    def test_trade_log_starts_empty(self):
        bt = Backtester(symbols=["TEST"])
        assert bt.trade_log == []

    def test_equity_curve_starts_empty(self):
        bt = Backtester(symbols=["TEST"])
        assert bt.equity_curve == []


class TestEquityCurve:
    """Test equity curve DataFrame generation."""

    def test_get_equity_curve_empty(self):
        bt = Backtester(symbols=["TEST"])
        df = bt.get_equity_curve()
        assert isinstance(df, pd.DataFrame)
        assert "equity" in df.columns
        assert len(df) == 0

    def test_get_equity_curve_with_data(self):
        bt = Backtester(symbols=["TEST"])
        bt.equity_curve = [100000, 101000, 99000]
        df = bt.get_equity_curve()
        assert len(df) == 3
        assert df["equity"].iloc[0] == 100000


class TestMetricsEdgeCases:
    """Edge cases for metric computation."""

    def test_single_day_equity(self):
        bt = Backtester(symbols=["TEST"])
        bt.equity_curve = [100000]
        bt.daily_returns = [0]
        bt.trade_log = []
        results = bt._compute_metrics(elapsed=0.1)
        assert results["sharpe_ratio"] == 0.0
        assert results["max_drawdown"] == 0.0

    def test_constant_equity(self):
        bt = Backtester(symbols=["TEST"])
        bt.equity_curve = [100000] * 100
        bt.daily_returns = [0.0] * 100
        bt.trade_log = []
        results = bt._compute_metrics(elapsed=0.1)
        assert results["total_return"] == 0.0
        assert results["max_drawdown"] == 0.0

    def test_profit_factor_no_losses(self):
        bt = Backtester(symbols=["TEST"])
        bt.equity_curve = [110000]
        bt.daily_returns = [0.1]
        bt.trade_log = [{"action": "SELL", "pnl": 100}]
        results = bt._compute_metrics(elapsed=0.1)
        assert results["profit_factor"] == float("inf")

    def test_results_keys(self):
        bt = Backtester(symbols=["TEST"])
        bt.equity_curve = [100000]
        bt.daily_returns = [0]
        bt.trade_log = []
        results = bt._compute_metrics(elapsed=0.5)
        expected_keys = {
            "period", "initial_capital", "final_value", "total_return",
            "annualized_return", "sharpe_ratio", "max_drawdown",
            "total_trades", "win_rate", "avg_win", "avg_loss",
            "profit_factor", "elapsed_seconds", "final_weights",
        }
        assert expected_keys.issubset(set(results.keys()))


class TestMathVerification:
    """Verify mathematical formulas used in backtester."""

    def test_sharpe_ratio_known_values(self):
        """Sharpe ratio with known daily returns."""
        returns = np.array([0.01, 0.02, -0.005, 0.015, 0.01])
        mean_r = np.mean(returns)
        std_r = np.std(returns)
        expected = mean_r / std_r * np.sqrt(252)

        bt = Backtester(symbols=["TEST"])
        bt.daily_returns = list(returns)
        bt.equity_curve = list(100000 * np.cumprod(1 + returns))
        bt.trade_log = []
        results = bt._compute_metrics(elapsed=0.1)
        assert abs(results["sharpe_ratio"] - expected) < 1e-6

    def test_total_return_formula(self):
        bt = Backtester(symbols=["TEST"], initial_capital=100000)
        bt.equity_curve = [120000]
        bt.daily_returns = [0]
        bt.trade_log = []
        results = bt._compute_metrics(elapsed=0.1)
        assert abs(results["total_return"] - 0.20) < 1e-6

    def test_max_drawdown_peak_tracking(self):
        """Drawdown should track from the running peak, not just the start."""
        equity = [100, 120, 90, 110, 80]
        bt = Backtester(symbols=["TEST"])
        bt.equity_curve = equity
        bt.daily_returns = [0] * len(equity)
        bt.trade_log = []
        results = bt._compute_metrics(elapsed=0.1)
        # Peak is 120, lowest after peak is 80 → DD = (120-80)/120 = 0.3333
        assert abs(results["max_drawdown"] - (120 - 80) / 120) < 1e-6
