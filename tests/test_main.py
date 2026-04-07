"""Tests for main.py — CLI argument parsing and command routing."""

import argparse
from unittest.mock import patch, MagicMock

import pytest


class TestCLIArgumentParsing:
    """Test argument parser configuration."""

    def test_no_args_defaults_to_live(self):
        from main import main

        with patch("main.run_live") as mock_live:
            with patch("sys.argv", ["main.py"]):
                main()
            mock_live.assert_called_once()

    def test_live_command(self):
        from main import main

        with patch("main.run_live") as mock_live:
            with patch("sys.argv", ["main.py", "live"]):
                main()
            mock_live.assert_called_once()

    def test_backtest_command(self):
        from main import main

        with patch("main.run_backtest") as mock_bt:
            with patch("sys.argv", ["main.py", "backtest"]):
                main()
            mock_bt.assert_called_once()

    def test_backtest_with_symbols(self):
        from main import main

        with patch("main.run_backtest") as mock_bt:
            with patch("sys.argv", ["main.py", "backtest", "--symbols", "AAPL,MSFT"]):
                main()
            args = mock_bt.call_args[0][0]
            assert args.symbols == "AAPL,MSFT"

    def test_backtest_with_dates(self):
        from main import main

        with patch("main.run_backtest") as mock_bt:
            with patch(
                "sys.argv",
                ["main.py", "backtest", "--start", "2024-01-01", "--end", "2024-12-31"],
            ):
                main()
            args = mock_bt.call_args[0][0]
            assert args.start == "2024-01-01"
            assert args.end == "2024-12-31"

    def test_single_command(self):
        from main import main

        with patch("main.run_single_cycle") as mock_single:
            with patch("sys.argv", ["main.py", "single"]):
                main()
            mock_single.assert_called_once()

    def test_status_command(self):
        from main import main

        with patch("main.show_status") as mock_status:
            with patch("sys.argv", ["main.py", "status"]):
                main()
            mock_status.assert_called_once()


class TestCommandRouting:
    """Test that commands route to the correct handler."""

    def test_backtest_default_symbols_none(self):
        from main import main

        with patch("main.run_backtest") as mock_bt:
            with patch("sys.argv", ["main.py", "backtest"]):
                main()
            args = mock_bt.call_args[0][0]
            assert args.symbols is None

    def test_backtest_default_dates_none(self):
        from main import main

        with patch("main.run_backtest") as mock_bt:
            with patch("sys.argv", ["main.py", "backtest"]):
                main()
            args = mock_bt.call_args[0][0]
            assert args.start is None
            assert args.end is None
