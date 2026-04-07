"""Tests for risk_manager.py."""

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pytest

from risk_manager import RiskManager, Position


class TestPositionSizing:
    """Test calculate_position_size with volatility adjustment."""

    def test_basic_sizing(self, risk_manager):
        shares = risk_manager.calculate_position_size(
            "AAPL", price=150.0, signal_strength=0.8, volatility=0.02
        )
        assert shares > 0
        # Max dollars = 100k * 0.05 = 5000
        # vol_scalar = max(0.2, min(1.0, 0.02/0.02)) = 1.0
        # signal_scalar = 0.8
        # adjusted = 5000 * 1.0 * 0.8 = 4000
        # shares = int(4000 / 150) = 26
        assert shares == 26

    def test_high_volatility_reduces_size(self, risk_manager):
        shares_normal = risk_manager.calculate_position_size(
            "AAPL", price=150.0, signal_strength=0.8, volatility=0.02
        )
        shares_high_vol = risk_manager.calculate_position_size(
            "AAPL", price=150.0, signal_strength=0.8, volatility=0.10
        )
        assert shares_high_vol < shares_normal

    def test_low_signal_reduces_size(self, risk_manager):
        shares_strong = risk_manager.calculate_position_size(
            "AAPL", price=150.0, signal_strength=0.9, volatility=0.02
        )
        shares_weak = risk_manager.calculate_position_size(
            "AAPL", price=150.0, signal_strength=0.3, volatility=0.02
        )
        assert shares_weak < shares_strong

    def test_zero_price_returns_zero(self, risk_manager):
        shares = risk_manager.calculate_position_size(
            "AAPL", price=0.0, signal_strength=0.5, volatility=0.02
        )
        assert shares == 0

    def test_zero_volatility_handled(self, risk_manager):
        shares = risk_manager.calculate_position_size(
            "AAPL", price=150.0, signal_strength=0.5, volatility=0.0
        )
        assert shares >= 0  # should not crash

    def test_extreme_volatility(self, risk_manager):
        shares = risk_manager.calculate_position_size(
            "AAPL", price=150.0, signal_strength=0.5, volatility=100.0
        )
        assert shares >= 0

    def test_halted_returns_zero(self, risk_manager):
        risk_manager.trading_halted = True
        shares = risk_manager.calculate_position_size(
            "AAPL", price=150.0, signal_strength=0.8, volatility=0.02
        )
        assert shares == 0


class TestStopLossTakeProfit:
    """Test ATR-based stop-loss and take-profit calculation."""

    def test_basic_stop_take(self, risk_manager):
        stop, take = risk_manager.compute_stop_take(entry_price=100.0, atr=2.0)
        # ATR stop = 100 - 2*2 = 96
        # Pct stop = 100 * (1 - 0.03) = 97
        # stop_loss = max(96, 97) = 97
        assert stop == 97.0
        # ATR TP = 100 + 3*2 = 106
        # Pct TP = 100 * (1 + 0.08) = 108
        # take_profit = min(106, 108) = 106
        assert take == 106.0

    def test_stop_below_entry(self, risk_manager):
        stop, take = risk_manager.compute_stop_take(100.0, 1.0)
        assert stop < 100.0

    def test_take_above_entry(self, risk_manager):
        stop, take = risk_manager.compute_stop_take(100.0, 1.0)
        assert take > 100.0

    def test_zero_atr(self, risk_manager):
        stop, take = risk_manager.compute_stop_take(100.0, 0.0)
        # ATR stop = 100 - 0 = 100
        # Pct stop = 97
        # stop = max(100, 97) = 100
        assert stop == 100.0

    def test_high_atr(self, risk_manager):
        stop, take = risk_manager.compute_stop_take(100.0, 50.0)
        # ATR stop = 100 - 100 = 0
        # Pct stop = 97
        # stop = max(0, 97) = 97
        assert stop == 97.0


class TestDrawdownCircuitBreaker:
    """Test drawdown monitoring and trading halt."""

    def test_no_drawdown_initially(self, risk_manager):
        assert not risk_manager.check_drawdown()
        assert not risk_manager.trading_halted

    def test_drawdown_triggers_halt(self, risk_manager):
        # Simulate 15% loss
        risk_manager.current_capital = 85_000.0
        risk_manager.peak_capital = 100_000.0
        halted = risk_manager.check_drawdown()
        assert halted
        assert risk_manager.trading_halted

    def test_drawdown_just_below_threshold(self, risk_manager):
        # 9% drawdown (below 10% threshold)
        risk_manager.current_capital = 91_000.0
        risk_manager.peak_capital = 100_000.0
        halted = risk_manager.check_drawdown()
        assert not halted

    def test_reset_halt(self, risk_manager):
        risk_manager.trading_halted = True
        risk_manager.reset_halt()
        assert not risk_manager.trading_halted


class TestMaxPositionLimits:
    """Test position limits."""

    def test_can_trade_initially(self, risk_manager):
        allowed, reason = risk_manager.can_trade("AAPL")
        assert allowed
        assert reason == "OK"

    def test_max_positions_reached(self, risk_manager):
        # Fill up to max positions
        for i in range(10):
            sym = f"SYM{i}"
            risk_manager.positions[sym] = Position(
                symbol=sym, quantity=1, entry_price=100.0,
                entry_time=datetime.now(), strategy="test",
                stop_loss=95.0, take_profit=110.0,
            )
        allowed, reason = risk_manager.can_trade("NEW")
        assert not allowed
        assert "Max open positions" in reason

    def test_can_trade_existing_position(self, risk_manager):
        for i in range(10):
            sym = f"SYM{i}"
            risk_manager.positions[sym] = Position(
                symbol=sym, quantity=1, entry_price=100.0,
                entry_time=datetime.now(), strategy="test",
                stop_loss=95.0, take_profit=110.0,
            )
        # Should still be able to trade an existing position
        allowed, _ = risk_manager.can_trade("SYM0")
        assert allowed

    def test_daily_trade_limit(self, risk_manager):
        risk_manager.daily_trade_date = datetime.now().strftime("%Y-%m-%d")
        risk_manager.daily_trades = 20
        allowed, reason = risk_manager.can_trade("AAPL")
        assert not allowed
        assert "Daily trade limit" in reason

    def test_halted_cannot_trade(self, risk_manager):
        risk_manager.trading_halted = True
        allowed, reason = risk_manager.can_trade("AAPL")
        assert not allowed
        assert "drawdown" in reason.lower()


class TestPositionManagement:
    """Test opening and closing positions."""

    def test_open_position(self, risk_manager):
        risk_manager.open_position("AAPL", 10, 150.0, "momentum", 3.0)
        assert "AAPL" in risk_manager.positions
        pos = risk_manager.positions["AAPL"]
        assert pos.quantity == 10
        assert pos.entry_price == 150.0
        assert risk_manager.current_capital == 100_000.0 - 10 * 150.0

    def test_close_position_profit(self, risk_manager):
        risk_manager.open_position("AAPL", 10, 100.0, "momentum", 2.0)
        pnl = risk_manager.close_position("AAPL", 110.0)
        assert pnl == 100.0  # (110-100)*10
        assert "AAPL" not in risk_manager.positions

    def test_close_position_loss(self, risk_manager):
        risk_manager.open_position("AAPL", 10, 100.0, "momentum", 2.0)
        pnl = risk_manager.close_position("AAPL", 90.0)
        assert pnl == -100.0

    def test_close_nonexistent_position(self, risk_manager):
        result = risk_manager.close_position("NONEXISTENT", 100.0)
        assert result is None

    def test_daily_trades_increment(self, risk_manager):
        initial = risk_manager.daily_trades
        risk_manager.open_position("AAPL", 10, 100.0, "test", 2.0)
        assert risk_manager.daily_trades == initial + 1


class TestStopLossTakeProfitCheck:
    """Test checking positions against current prices."""

    def test_stop_loss_triggered(self, risk_manager):
        risk_manager.open_position("AAPL", 10, 100.0, "test", 2.0)
        stop_price = risk_manager.positions["AAPL"].stop_loss
        to_close = risk_manager.check_stop_loss_take_profit(
            {"AAPL": stop_price - 1.0}
        )
        assert "AAPL" in to_close

    def test_take_profit_triggered(self, risk_manager):
        risk_manager.open_position("AAPL", 10, 100.0, "test", 2.0)
        tp_price = risk_manager.positions["AAPL"].take_profit
        to_close = risk_manager.check_stop_loss_take_profit(
            {"AAPL": tp_price + 1.0}
        )
        assert "AAPL" in to_close

    def test_no_trigger_mid_range(self, risk_manager):
        risk_manager.open_position("AAPL", 10, 100.0, "test", 2.0)
        to_close = risk_manager.check_stop_loss_take_profit({"AAPL": 100.0})
        assert len(to_close) == 0


class TestRiskStatus:
    """Test status reporting."""

    def test_status_keys(self, risk_manager):
        status = risk_manager.get_status()
        assert "capital" in status
        assert "total_value" in status
        assert "open_positions" in status
        assert "exposure_pct" in status
        assert "drawdown_pct" in status
        assert "trading_halted" in status
        assert "daily_trades" in status

    def test_initial_status(self, risk_manager):
        status = risk_manager.get_status()
        assert status["capital"] == 100_000.0
        assert status["open_positions"] == 0
        assert status["trading_halted"] is False
