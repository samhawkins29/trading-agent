"""
Tests for the LIVE execution path (live_trader.py).

This is the code that actually touches money in paper trading, yet historically
had no test file (see REVIEW_AND_IMPROVEMENTS.md §2.7). Every test here runs
fully offline: the trader is constructed with ``dry_run=True`` (no Alpaca
connection) and the broker API is mocked. No order is ever sent to a live or
paper broker.
"""

import json
import os
from unittest.mock import MagicMock

import pytest

from config import config
from live_trader import LiveTrader


@pytest.fixture(autouse=True)
def restore_config():
    """Snapshot/restore the shared config singleton.

    Several behaviours under test mutate the module-level ``config`` (learned
    stop/take-profit application in particular). Without restoration those
    mutations would leak across tests and into the rest of the suite.
    """
    saved = {
        "stop_loss_pct": config.stop_loss_pct,
        "take_profit_pct": config.take_profit_pct,
    }
    yield
    config.stop_loss_pct = saved["stop_loss_pct"]
    config.take_profit_pct = saved["take_profit_pct"]


@pytest.fixture
def trader(tmp_path, monkeypatch):
    """A dry-run LiveTrader rooted in an isolated temp cwd.

    ``_apply_learned_params`` reads ``learned_params.json`` from cwd, so we
    chdir into a clean tmp dir to control exactly what it sees.
    """
    monkeypatch.chdir(tmp_path)
    return LiveTrader(dry_run=True)


def _write_learned_params(tmp_path, **overrides):
    params = {
        "strategy_weights": {
            "mean_reversion": 0.25,
            "momentum": 0.30,
            "sentiment": 0.20,
            "pattern_recognition": 0.25,
        },
        "buy_threshold": 0.30,
        "sell_threshold": -0.30,
        "updated_at": "2026-06-01",
    }
    params.update(overrides)
    with open(os.path.join(tmp_path, "learned_params.json"), "w") as f:
        json.dump(params, f)


class TestLearnedParamsApplication:
    """P1: learned stop/take-profit must actually take effect, not be dropped."""

    def test_learned_stop_and_take_are_applied_to_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_learned_params(tmp_path, stop_loss_pct=0.04, take_profit_pct=0.10)

        LiveTrader(dry_run=True)

        # risk_manager.compute_stop_take reads config.stop_loss_pct directly,
        # so these MUST be live on the shared config after construction.
        assert config.stop_loss_pct == 0.04
        assert config.take_profit_pct == 0.10

    def test_learned_stop_flows_into_risk_manager_levels(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_learned_params(tmp_path, stop_loss_pct=0.04, take_profit_pct=0.10)

        lt = LiveTrader(dry_run=True)
        # ATR=0 forces the percentage-based branch in compute_stop_take.
        stop, take = lt.risk_manager.compute_stop_take(
            entry_price=100.0, atr=0.0, strategy="momentum"
        )
        assert stop == pytest.approx(96.0)   # 100 * (1 - 0.04)
        assert take == pytest.approx(110.0)  # 100 * (1 + 0.10)

    def test_thresholds_applied(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_learned_params(tmp_path, buy_threshold=0.33, sell_threshold=-0.27)
        lt = LiveTrader(dry_run=True)
        assert lt._buy_threshold == 0.33
        assert lt._sell_threshold == -0.27

    def test_out_of_range_stop_is_ignored(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        default_sl = config.stop_loss_pct
        _write_learned_params(tmp_path, stop_loss_pct=0.90)  # absurd
        LiveTrader(dry_run=True)
        assert config.stop_loss_pct == default_sl  # unchanged

    def test_missing_file_is_safe(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        default_sl = config.stop_loss_pct
        # no learned_params.json written
        LiveTrader(dry_run=True)
        assert config.stop_loss_pct == default_sl


def _mock_connected(trader):
    """Attach a mock Alpaca API so order-submission paths are exercised."""
    trader.alpaca_connected = True
    trader.api = MagicMock()
    trader.dry_run = False
    return trader.api


class TestNativeRestingStops:
    """P2: every opened position rests a broker-native stop order."""

    def test_resting_stop_submitted_for_long(self, trader):
        api = _mock_connected(trader)
        api.submit_order.return_value = MagicMock(id="stop-1")
        trader.risk_manager.open_position(
            "AAPL", quantity=10, price=100.0, strategy="momentum", atr=0.0
        )
        trader._submit_resting_stop("AAPL")

        api.submit_order.assert_called_once()
        kwargs = api.submit_order.call_args.kwargs
        assert kwargs["type"] == "stop"
        assert kwargs["side"] == "sell"          # closing side for a long
        assert kwargs["time_in_force"] == "gtc"  # rests across sessions
        assert trader._resting_stops["AAPL"] == "stop-1"

    def test_resting_stop_cancelled_on_close(self, trader):
        api = _mock_connected(trader)
        api.submit_order.return_value = MagicMock(id="stop-2")
        trader.risk_manager.open_position(
            "MSFT", quantity=5, price=200.0, strategy="momentum", atr=0.0
        )
        trader._submit_resting_stop("MSFT")
        trader._cancel_resting_stop("MSFT")

        api.cancel_order.assert_called_once_with("stop-2")
        assert "MSFT" not in trader._resting_stops

    def test_dry_run_never_contacts_broker(self, trader):
        # trader fixture is dry_run=True with no api
        trader.risk_manager.open_position(
            "NVDA", quantity=3, price=300.0, strategy="momentum", atr=0.0
        )
        trader._submit_resting_stop("NVDA")  # must not raise
        assert trader._resting_stops == {}

    def test_disabled_when_use_native_stops_false(self, trader, monkeypatch):
        api = _mock_connected(trader)
        monkeypatch.setattr(config, "use_native_stops", False)
        trader.risk_manager.open_position(
            "TSLA", quantity=2, price=250.0, strategy="momentum", atr=0.0
        )
        trader._submit_resting_stop("TSLA")
        api.submit_order.assert_not_called()


class TestKillSwitch:
    """P2: real-time max-drawdown / daily-loss circuit breaker."""

    def test_trips_on_drawdown_and_liquidates(self, trader, monkeypatch):
        api = _mock_connected(trader)
        trader.risk_manager.peak_capital = 100_000.0
        trader.risk_manager.open_position(
            "AAPL", quantity=10, price=100.0, strategy="momentum", atr=0.0
        )
        # Real equity collapses to 75k => 25% drawdown (> 20% limit).
        monkeypatch.setattr(trader, "_get_actual_portfolio_value", lambda: 75_000.0)

        tripped = trader._check_kill_switch()

        assert tripped is True
        assert trader._kill_switch_tripped is True
        assert trader.risk_manager.trading_halted is True
        # A market order to flatten the long must have been submitted.
        sides = [c.kwargs.get("side") for c in api.submit_order.call_args_list]
        types = [c.kwargs.get("type") for c in api.submit_order.call_args_list]
        assert "sell" in sides and "market" in types
        api.cancel_all_orders.assert_called_once()
        # Persistent halt flag written.
        assert os.path.exists(trader._halt_flag_path)

    def test_trips_on_daily_loss(self, trader, monkeypatch):
        _mock_connected(trader)
        trader.risk_manager.peak_capital = 100_000.0
        # First check of the day sets the baseline at 100k...
        monkeypatch.setattr(trader, "_get_actual_portfolio_value", lambda: 100_000.0)
        assert trader._check_kill_switch() is False
        # ...then an intraday drop to 91k => 9% daily loss (> 8% limit).
        monkeypatch.setattr(trader, "_get_actual_portfolio_value", lambda: 91_000.0)
        assert trader._check_kill_switch() is True

    def test_does_not_trip_below_limits(self, trader, monkeypatch):
        _mock_connected(trader)
        trader.risk_manager.peak_capital = 100_000.0
        monkeypatch.setattr(trader, "_get_actual_portfolio_value", lambda: 95_000.0)
        assert trader._check_kill_switch() is False
        assert trader._kill_switch_tripped is False

    def test_zero_equity_read_does_not_liquidate(self, trader, monkeypatch):
        _mock_connected(trader)
        monkeypatch.setattr(trader, "_get_actual_portfolio_value", lambda: 0.0)
        assert trader._check_kill_switch() is False

    def test_persistent_flag_halts_on_restart(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Pre-create the halt flag where a fresh trader will look for it.
        flag = os.path.join(config.log_dir, "KILL_SWITCH.flag")
        os.makedirs(config.log_dir, exist_ok=True)
        with open(flag, "w") as f:
            f.write("prior trip\n")
        try:
            lt = LiveTrader(dry_run=True)
            assert lt._kill_switch_tripped is True
            assert lt.risk_manager.trading_halted is True
        finally:
            if os.path.exists(flag):
                os.remove(flag)
