#!/usr/bin/env python3
"""
AI Trading Agent — Entry Point.

Usage:
  python main.py                  # Run live paper trading
  python main.py --backtest       # Run backtest on historical data
  python main.py --status         # Show current agent status
  python main.py --single         # Run a single trading cycle

Before running:
  1. Copy your API keys into config.py or set environment variables
  2. pip install -r requirements.txt
  3. (Optional) Set up Alpaca paper trading account at https://alpaca.markets
"""

import argparse
import json
import signal
import sys
import time

from config import config
from agent import TradingAgent
from backtester import Backtester


def run_live(args):
    """Run the live paper-trading loop."""
    agent = TradingAgent(
        capital=config.initial_capital,
        paper_trade=True,
    )

    # Graceful shutdown
    running = True

    def handle_signal(sig, frame):
        nonlocal running
        print("\nShutting down gracefully...")
        running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print("=" * 60)
    print("  AI Trading Agent — LIVE PAPER TRADING")
    print("=" * 60)
    print(f"  Symbols:  {config.symbols}")
    print(f"  Capital:  ${config.initial_capital:,.2f}")
    print(f"  Interval: {config.rebalance_interval_minutes} min")
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    while running:
        try:
            actions = agent.run_cycle()
            buys = len(actions.get("buys", []))
            sells = len(actions.get("sells", []))
            print(
                f"\n  Cycle {agent.cycle_count}: "
                f"{buys} buys, {sells} sells | "
                f"Total PnL: ${agent.total_pnl:+,.2f}"
            )

            # Wait for next cycle
            wait = config.rebalance_interval_minutes * 60
            print(f"  Next cycle in {config.rebalance_interval_minutes} min...")
            for _ in range(int(wait)):
                if not running:
                    break
                time.sleep(1)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n  ERROR: {e}")
            time.sleep(60)

    # Final status
    print("\n" + "=" * 60)
    print("  FINAL STATUS")
    print("=" * 60)
    status = agent.get_status()
    print(json.dumps(status, indent=2, default=str))


def run_backtest(args):
    """Run historical backtest."""
    symbols = args.symbols.split(",") if args.symbols else config.symbols

    print("=" * 60)
    print("  AI Trading Agent — BACKTEST MODE")
    print("=" * 60)

    bt = Backtester(
        symbols=symbols,
        start_date=args.start or config.backtest_start,
        end_date=args.end or config.backtest_end,
        initial_capital=config.initial_capital,
    )

    results = bt.run()

    # Save results
    with open("logs/backtest/results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to logs/backtest/results.json")


def run_single_cycle(args):
    """Run a single trading cycle (useful for cron jobs)."""
    agent = TradingAgent(
        capital=config.initial_capital,
        paper_trade=True,
    )
    actions = agent.run_cycle()
    print(json.dumps(actions, indent=2, default=str))


def show_status(args):
    """Show current agent status."""
    agent = TradingAgent(
        capital=config.initial_capital,
        paper_trade=True,
    )
    status = agent.get_status()
    print(json.dumps(status, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(
        description="AI Trading Agent — Multi-Strategy Paper Trader"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Live trading
    live_parser = subparsers.add_parser("live", help="Run live paper trading")

    # Backtest
    bt_parser = subparsers.add_parser("backtest", help="Run backtest")
    bt_parser.add_argument(
        "--symbols", type=str, default=None,
        help="Comma-separated symbols (default: config.symbols)",
    )
    bt_parser.add_argument(
        "--start", type=str, default=None,
        help="Backtest start date (YYYY-MM-DD)",
    )
    bt_parser.add_argument(
        "--end", type=str, default=None,
        help="Backtest end date (YYYY-MM-DD)",
    )

    # Single cycle
    subparsers.add_parser("single", help="Run a single trading cycle")

    # Status
    subparsers.add_parser("status", help="Show agent status")

    args = parser.parse_args()

    if args.command == "live" or args.command is None:
        run_live(args)
    elif args.command == "backtest":
        run_backtest(args)
    elif args.command == "single":
        run_single_cycle(args)
    elif args.command == "status":
        show_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
