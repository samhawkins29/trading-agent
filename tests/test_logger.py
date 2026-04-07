"""Tests for logger.py — CSV trade logs, JSON snapshots, file operations."""

import csv
import json
import os

import pytest

from logger import TradeLogger


class TestCSVTradeLog:
    """Test CSV trade log format and operations."""

    def test_csv_file_created(self, tmp_log_dir):
        logger = TradeLogger(log_dir=tmp_log_dir)
        assert os.path.exists(logger.trades_file)

    def test_csv_header(self, tmp_log_dir):
        logger = TradeLogger(log_dir=tmp_log_dir)
        with open(logger.trades_file) as f:
            reader = csv.reader(f)
            header = next(reader)
        expected = [
            "timestamp", "symbol", "action", "quantity", "price",
            "strategy", "signal_strength", "portfolio_value", "reason",
        ]
        assert header == expected

    def test_log_trade_appends_row(self, tmp_log_dir):
        logger = TradeLogger(log_dir=tmp_log_dir)
        logger.log_trade(
            symbol="AAPL", action="BUY", quantity=10, price=150.0,
            strategy="momentum", signal_strength=0.75,
            portfolio_value=100000.0, reason="test buy",
        )
        with open(logger.trades_file) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 2  # header + 1 trade

    def test_multiple_trades(self, tmp_log_dir):
        logger = TradeLogger(log_dir=tmp_log_dir)
        for i in range(5):
            logger.log_trade(
                symbol=f"SYM{i}", action="BUY", quantity=i + 1,
                price=100.0 + i, strategy="test",
                signal_strength=0.5, portfolio_value=100000.0,
            )
        with open(logger.trades_file) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert len(rows) == 6  # header + 5 trades

    def test_csv_field_values(self, tmp_log_dir):
        logger = TradeLogger(log_dir=tmp_log_dir)
        logger.log_trade(
            symbol="TSLA", action="SELL", quantity=5, price=200.0,
            strategy="mean_reversion", signal_strength=-0.6,
            portfolio_value=95000.0, reason="take profit",
        )
        with open(logger.trades_file) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["symbol"] == "TSLA"
        assert row["action"] == "SELL"
        assert row["quantity"] == "5"
        assert row["strategy"] == "mean_reversion"


class TestJSONSnapshot:
    """Test JSON performance snapshot format."""

    def test_snapshot_creates_file(self, tmp_log_dir):
        logger = TradeLogger(log_dir=tmp_log_dir)
        logger.log_performance_snapshot({"test_key": "test_value"})
        assert os.path.exists(logger.perf_file)

    def test_snapshot_appends(self, tmp_log_dir):
        logger = TradeLogger(log_dir=tmp_log_dir)
        logger.log_performance_snapshot({"cycle": 1, "value": 100000})
        logger.log_performance_snapshot({"cycle": 2, "value": 101000})
        with open(logger.perf_file) as f:
            data = json.load(f)
        assert len(data) == 2

    def test_snapshot_has_timestamp(self, tmp_log_dir):
        logger = TradeLogger(log_dir=tmp_log_dir)
        logger.log_performance_snapshot({"test": True})
        with open(logger.perf_file) as f:
            data = json.load(f)
        assert "timestamp" in data[0]


class TestFileCreation:
    """Test file creation and appending behavior."""

    def test_log_directory_created(self, tmp_path):
        log_dir = str(tmp_path / "new_logs")
        logger = TradeLogger(log_dir=log_dir)
        assert os.path.isdir(log_dir)

    def test_log_file_created(self, tmp_log_dir):
        logger = TradeLogger(log_dir=tmp_log_dir)
        log_files = [f for f in os.listdir(tmp_log_dir) if f.endswith(".log")]
        assert len(log_files) >= 1

    def test_csv_not_overwritten(self, tmp_log_dir):
        logger1 = TradeLogger(log_dir=tmp_log_dir)
        logger1.log_trade(
            symbol="AAPL", action="BUY", quantity=10, price=150.0,
            strategy="test", signal_strength=0.5, portfolio_value=100000.0,
        )
        # Creating another logger shouldn't overwrite existing trades
        logger2 = TradeLogger(log_dir=tmp_log_dir)
        with open(logger2.trades_file) as f:
            rows = list(csv.reader(f))
        assert len(rows) >= 2  # header + at least 1 trade


class TestLogMethods:
    """Test general logging methods."""

    def test_info(self, tmp_log_dir):
        logger = TradeLogger(log_dir=tmp_log_dir)
        logger.info("Test info message")  # should not raise

    def test_warning(self, tmp_log_dir):
        logger = TradeLogger(log_dir=tmp_log_dir)
        logger.warning("Test warning")

    def test_error(self, tmp_log_dir):
        logger = TradeLogger(log_dir=tmp_log_dir)
        logger.error("Test error")

    def test_debug(self, tmp_log_dir):
        logger = TradeLogger(log_dir=tmp_log_dir)
        logger.debug("Test debug")

    def test_log_strategy_weights(self, tmp_log_dir):
        logger = TradeLogger(log_dir=tmp_log_dir)
        weights = {"momentum": 0.3, "mean_reversion": 0.3}
        logger.log_strategy_weights(weights)


class TestTradeSummary:
    """Test get_trade_summary."""

    def test_empty_summary(self, tmp_log_dir):
        logger = TradeLogger(log_dir=tmp_log_dir)
        summary = logger.get_trade_summary()
        assert summary["total_trades"] == 0

    def test_summary_with_trades(self, tmp_log_dir):
        logger = TradeLogger(log_dir=tmp_log_dir)
        logger.log_trade("AAPL", "BUY", 10, 150.0, "momentum", 0.5, 100000.0)
        logger.log_trade("AAPL", "SELL", 10, 155.0, "momentum", -0.3, 100050.0)
        summary = logger.get_trade_summary()
        assert summary["total_trades"] == 2
        assert summary["buys"] == 1
        assert summary["sells"] == 1
