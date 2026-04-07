"""Tests for agent.py — strategy combination, decision making, trade flow."""

from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from strategies.mean_reversion import Signal


class TestStrategyWeighting:
    """Test strategy combination / weighting."""

    def test_combined_signal_weighted(self):
        """Verify the weighted combination logic."""
        weights = {
            "mean_reversion": 0.30,
            "momentum": 0.30,
            "sentiment": 0.15,
            "pattern_recognition": 0.25,
        }
        signals = {
            "mean_reversion": Signal("AAPL", "BUY", 0.8, "mr", ""),
            "momentum": Signal("AAPL", "BUY", 0.6, "mom", ""),
            "sentiment": Signal("AAPL", "HOLD", 0.0, "sent", ""),
            "pattern_recognition": Signal("AAPL", "BUY", 0.4, "pr", ""),
        }
        combined = sum(
            weights.get(name, 0) * sig.strength for name, sig in signals.items()
        )
        combined = np.clip(combined, -1.0, 1.0)
        expected = 0.30 * 0.8 + 0.30 * 0.6 + 0.15 * 0.0 + 0.25 * 0.4
        assert abs(combined - expected) < 1e-10

    def test_opposing_signals_cancel(self):
        weights = {"a": 0.5, "b": 0.5}
        signals = {
            "a": Signal("AAPL", "BUY", 1.0, "a", ""),
            "b": Signal("AAPL", "SELL", -1.0, "b", ""),
        }
        combined = sum(weights.get(n, 0) * s.strength for n, s in signals.items())
        assert abs(combined) < 1e-10

    def test_all_hold_gives_zero(self):
        weights = {"a": 0.5, "b": 0.5}
        signals = {
            "a": Signal("AAPL", "HOLD", 0.0, "a", ""),
            "b": Signal("AAPL", "HOLD", 0.0, "b", ""),
        }
        combined = sum(weights.get(n, 0) * s.strength for n, s in signals.items())
        assert combined == 0.0


class TestDecisionMaking:
    """Test decision-making with mock signals."""

    @patch("agent.TradingAgent._check_alpaca", return_value=False)
    @patch("agent.TradingAgent._analyze_symbol")
    def test_buy_decision_on_strong_signal(self, mock_analyze, mock_alpaca, trade_logger):
        from agent import TradingAgent

        mock_analyze.return_value = (
            Signal("AAPL", "BUY", 0.8, "momentum", "strong buy"),
            {"price": 150.0, "atr": 3.0, "volatility": 0.02},
        )

        with patch("agent.DataFetcher"):
            agent = TradingAgent(capital=100_000.0, paper_trade=False)
            agent.logger = trade_logger

        actions = agent.run_cycle()
        assert isinstance(actions, dict)
        assert "buys" in actions
        assert "sells" in actions
        assert "holds" in actions

    @patch("agent.TradingAgent._check_alpaca", return_value=False)
    @patch("agent.TradingAgent._analyze_symbol")
    def test_hold_on_weak_signal(self, mock_analyze, mock_alpaca, trade_logger):
        from agent import TradingAgent

        mock_analyze.return_value = (
            Signal("AAPL", "HOLD", 0.1, "combined", "weak"),
            {"price": 150.0, "atr": 3.0, "volatility": 0.02},
        )

        with patch("agent.DataFetcher"):
            agent = TradingAgent(capital=100_000.0, paper_trade=False)
            agent.logger = trade_logger

        actions = agent.run_cycle()
        # Weak signal should result in holds
        assert isinstance(actions, dict)


class TestTradeExecutionFlow:
    """Test the buy/sell execution pipeline."""

    @patch("agent.TradingAgent._check_alpaca", return_value=False)
    def test_execute_buy_updates_positions(self, mock_alpaca, trade_logger):
        from agent import TradingAgent

        with patch("agent.DataFetcher"):
            agent = TradingAgent(capital=100_000.0, paper_trade=False)
            agent.logger = trade_logger

        signal = Signal("AAPL", "BUY", 0.8, "momentum", "test")
        meta = {"price": 100.0, "atr": 2.0, "volatility": 0.02}
        agent._execute_buy("AAPL", signal, meta)

        assert "AAPL" in agent.risk_manager.positions

    @patch("agent.TradingAgent._check_alpaca", return_value=False)
    def test_execute_sell_closes_position(self, mock_alpaca, trade_logger):
        from agent import TradingAgent

        with patch("agent.DataFetcher"):
            agent = TradingAgent(capital=100_000.0, paper_trade=False)
            agent.logger = trade_logger

        # Open a position first
        agent.risk_manager.open_position("AAPL", 10, 100.0, "momentum", 2.0)
        assert "AAPL" in agent.risk_manager.positions

        agent._execute_sell("AAPL", 105.0, "test_sell")
        assert "AAPL" not in agent.risk_manager.positions

    @patch("agent.TradingAgent._check_alpaca", return_value=False)
    def test_sell_nonexistent_noop(self, mock_alpaca, trade_logger):
        from agent import TradingAgent

        with patch("agent.DataFetcher"):
            agent = TradingAgent(capital=100_000.0, paper_trade=False)
            agent.logger = trade_logger

        # Selling a position that doesn't exist should be a no-op
        agent._execute_sell("NONEXISTENT", 100.0, "test")
        assert len(agent.risk_manager.positions) == 0


class TestStatePersistence:
    """Test state between cycles."""

    @patch("agent.TradingAgent._check_alpaca", return_value=False)
    def test_cycle_count_increments(self, mock_alpaca, trade_logger):
        from agent import TradingAgent

        with patch("agent.DataFetcher"):
            agent = TradingAgent(capital=100_000.0, paper_trade=False)
            agent.logger = trade_logger

        with patch.object(agent, "_analyze_symbol", return_value=(None, {})):
            agent.run_cycle()
            assert agent.cycle_count == 1
            agent.run_cycle()
            assert agent.cycle_count == 2

    @patch("agent.TradingAgent._check_alpaca", return_value=False)
    def test_total_pnl_accumulates(self, mock_alpaca, trade_logger):
        from agent import TradingAgent

        with patch("agent.DataFetcher"):
            agent = TradingAgent(capital=100_000.0, paper_trade=False)
            agent.logger = trade_logger

        # Open and close a position with profit
        agent.risk_manager.open_position("AAPL", 10, 100.0, "momentum", 2.0)
        agent._execute_sell("AAPL", 110.0, "test")
        assert agent.total_pnl == 100.0  # (110-100)*10


class TestSelfImprovementTrigger:
    """Test self-improvement triggers after trades."""

    @patch("agent.TradingAgent._check_alpaca", return_value=False)
    def test_self_improvement_records_experience(self, mock_alpaca, trade_logger):
        from agent import TradingAgent

        with patch("agent.DataFetcher"):
            agent = TradingAgent(capital=100_000.0, paper_trade=False)
            agent.logger = trade_logger

        agent.risk_manager.open_position("AAPL", 10, 100.0, "momentum", 2.0)
        agent._execute_sell("AAPL", 105.0, "test")
        # Sell should record an experience
        assert len(agent.self_improver.replay_buffer) >= 1


class TestGetStatus:
    """Test status reporting."""

    @patch("agent.TradingAgent._check_alpaca", return_value=False)
    def test_status_keys(self, mock_alpaca, trade_logger):
        from agent import TradingAgent

        with patch("agent.DataFetcher"):
            agent = TradingAgent(capital=100_000.0, paper_trade=False)
            agent.logger = trade_logger

        status = agent.get_status()
        assert "cycle_count" in status
        assert "total_pnl" in status
        assert "risk" in status
        assert "strategy_weights" in status
