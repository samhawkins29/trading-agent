"""
Backtest-integrity tests (REVIEW_AND_IMPROVEMENTS.md §2.2).

These cover the two structural fixes that make backtest P&L trustworthy:
  - execution moves to the NEXT bar's open (no same-bar look-ahead)
  - commission + slippage are charged on every fill

All offline: we drive the cost/fill helpers and _execute_pending_orders
directly with in-memory data, so no network or yfinance call is made.
"""

import pytest

from backtester import Backtester


@pytest.fixture
def bt():
    return Backtester(
        symbols=["X"],
        start_date="2020-01-01",
        end_date="2020-12-31",
        initial_capital=100_000.0,
        commission_per_trade=1.0,
        slippage_bps=5.0,
    )


class TestSlippage:
    def test_buy_fills_above_reference(self, bt):
        assert bt._fill_price(100.0, "buy") == pytest.approx(100.0 * 1.0005)

    def test_sell_fills_below_reference(self, bt):
        assert bt._fill_price(100.0, "sell") == pytest.approx(100.0 * 0.9995)


class TestNextBarExecution:
    def test_buy_fills_at_next_bar_open_not_signal_close(self, bt):
        # Order was decided on the prior bar; it must fill at THIS open.
        bt._pending_orders = [{
            "symbol": "X", "side": "buy", "strategy": "momentum",
            "combined": 0.8, "vol": 0.02, "atr": 0.0, "regime": "trending_up",
        }]
        day_actions = {"buys": [], "sells": []}
        cap_before = bt.risk_manager.current_capital

        bt._execute_pending_orders({"X": 100.0}, "2020-02-03", day_actions)

        assert "X" in bt.risk_manager.positions
        pos = bt.risk_manager.positions["X"]
        # Filled at the slipped open, never at the decision close.
        assert pos.entry_price == pytest.approx(100.0 * 1.0005)
        assert day_actions["buys"] == ["X"]
        # Commission left the account on top of the share cost.
        spent = cap_before - bt.risk_manager.current_capital
        assert spent == pytest.approx(pos.quantity * pos.entry_price + 1.0)
        # Order queue is drained after execution.
        assert bt._pending_orders == []

    def test_sell_realizes_pnl_net_of_costs(self, bt):
        # Open a position at 100, then queue a signal-exit filled next bar at 110.
        bt.risk_manager.open_position("X", quantity=10, price=100.0,
                                      strategy="momentum", atr=0.0)
        bt._pending_orders = [{
            "symbol": "X", "side": "sell", "strategy": "signal_exit",
            "combined": -0.8, "vol": 0.02, "atr": 0.0, "regime": "trending_down",
        }]
        day_actions = {"buys": [], "sells": []}
        bt._execute_pending_orders({"X": 110.0}, "2020-02-04", day_actions)

        assert "X" not in bt.risk_manager.positions
        sell = [t for t in bt.trade_log if t["action"] == "SELL"][-1]
        # Sell filled below 110 (slippage) and pnl is net of the $1 commission.
        assert sell["price"] == pytest.approx(110.0 * 0.9995)
        gross = (110.0 * 0.9995 - 100.0) * 10
        assert sell["pnl"] == pytest.approx(gross - 1.0)

    def test_order_dropped_when_no_open_price(self, bt):
        bt._pending_orders = [{
            "symbol": "X", "side": "buy", "strategy": "momentum",
            "combined": 0.8, "vol": 0.02, "atr": 0.0, "regime": "trending_up",
        }]
        day_actions = {"buys": [], "sells": []}
        bt._execute_pending_orders({}, "2020-02-05", day_actions)  # no open for X
        assert "X" not in bt.risk_manager.positions
        assert day_actions["buys"] == []

    def test_costs_make_a_round_trip_lose_at_flat_price(self, bt):
        # Buy then sell at the SAME open => slippage + 2x commission => a loss.
        bt._pending_orders = [{
            "symbol": "X", "side": "buy", "strategy": "momentum",
            "combined": 0.8, "vol": 0.02, "atr": 0.0, "regime": "trending_up",
        }]
        bt._execute_pending_orders({"X": 100.0}, "2020-02-03", {"buys": [], "sells": []})
        qty = bt.risk_manager.positions["X"].quantity
        bt._pending_orders = [{
            "symbol": "X", "side": "sell", "strategy": "signal_exit",
            "combined": -0.8, "vol": 0.02, "atr": 0.0, "regime": "trending_down",
        }]
        bt._execute_pending_orders({"X": 100.0}, "2020-02-04", {"buys": [], "sells": []})
        sell = [t for t in bt.trade_log if t["action"] == "SELL"][-1]
        assert sell["pnl"] < 0  # frictionless backtest would have shown ~0
