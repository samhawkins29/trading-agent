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
