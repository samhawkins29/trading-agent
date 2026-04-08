"""
Paper Trading Dashboard — Portfolio monitoring and performance tracking.

Shows portfolio status, tracks performance vs SPY benchmark, calculates
running Sharpe ratio, maximum drawdown, and win rate. Saves daily
snapshots to JSON for historical analysis.

Usage:
    python paper_trading_dashboard.py              # Print dashboard once
    python paper_trading_dashboard.py --live        # Refresh every 60s
    python paper_trading_dashboard.py --export      # Export snapshot to JSON
"""

import csv
import json
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    tradeapi = None

try:
    import yfinance as yf
except ImportError:
    yf = None

from config import ALPACA_API_KEY, ALPACA_BASE_URL, ALPACA_SECRET_KEY, config


class PaperTradingDashboard:
    """
    Real-time dashboard for monitoring paper trading performance.

    Tracks:
      - Current portfolio value, cash, positions
      - Daily and cumulative P&L
      - Performance vs SPY benchmark
      - Running Sharpe ratio (annualized)
      - Maximum drawdown (peak-to-trough)
      - Win rate from trade log
      - Per-strategy breakdown
    """

    def __init__(self):
        self.api = None
        self.connected = False
        self._connect_alpaca()

        # Snapshot storage
        self.snapshot_dir = os.path.join(config.log_dir, "dashboard_snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # Trade log path
        self.trade_log_path = os.path.join(config.log_dir, "paper_trades.csv")

        # Daily snapshots for performance tracking
        self.daily_snapshots_path = os.path.join(self.snapshot_dir, "daily_snapshots.json")
        self.daily_snapshots: List[Dict] = self._load_daily_snapshots()

    def _connect_alpaca(self):
        """Connect to Alpaca API."""
        if tradeapi is None:
            return
        if ALPACA_API_KEY == "YOUR_ALPACA_API_KEY":
            return
        try:
            self.api = tradeapi.REST(
                key_id=ALPACA_API_KEY,
                secret_key=ALPACA_SECRET_KEY,
                base_url=ALPACA_BASE_URL,
                api_version="v2",
            )
            self.api.get_account()
            self.connected = True
        except Exception:
            self.connected = False

    # ── Data Collection ──────────────────────────────────────────────────

    def get_portfolio_status(self) -> Dict[str, Any]:
        """Fetch current portfolio state from Alpaca."""
        if not self.connected or not self.api:
            return self._get_offline_status()

        try:
            account = self.api.get_account()
            positions = self.api.list_positions()

            position_data = []
            for pos in positions:
                position_data.append({
                    "symbol": pos.symbol,
                    "qty": int(pos.qty),
                    "side": pos.side,
                    "avg_entry_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "market_value": float(pos.market_value),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                    "change_today": float(pos.change_today),
                })

            return {
                "timestamp": datetime.now().isoformat(),
                "portfolio_value": float(account.portfolio_value),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "long_market_value": float(account.long_market_value),
                "daily_pl": float(account.equity) - float(account.last_equity),
                "daily_pl_pct": (
                    (float(account.equity) - float(account.last_equity))
                    / float(account.last_equity)
                    if float(account.last_equity) > 0 else 0.0
                ),
                "positions": position_data,
                "position_count": len(position_data),
                "connected": True,
            }
        except Exception as e:
            return {"error": str(e), "connected": False}

    def _get_offline_status(self) -> Dict[str, Any]:
        """Build status from local trade log when Alpaca is not connected."""
        trades = self._load_trade_log()
        total_pnl = sum(float(t.get("pnl", 0)) for t in trades)
        return {
            "timestamp": datetime.now().isoformat(),
            "portfolio_value": config.initial_capital + total_pnl,
            "cash": config.initial_capital + total_pnl,
            "total_pnl": total_pnl,
            "total_trades": len(trades),
            "positions": [],
            "position_count": 0,
            "connected": False,
        }

    # ── SPY Benchmark ────────────────────────────────────────────────────

    def get_spy_benchmark(self, start_date: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate SPY performance over the same period for comparison.

        Returns total return, annualized return, and volatility.
        """
        if yf is None:
            return {"error": "yfinance not installed"}

        if start_date is None:
            if self.daily_snapshots:
                start_date = self.daily_snapshots[0].get("date", "2025-01-01")
            else:
                start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        try:
            spy = yf.download("SPY", start=start_date, progress=False)
            if spy.empty:
                return {"total_return": 0, "annualized_return": 0, "volatility": 0}

            # Handle multi-level columns from yfinance
            close_col = spy["Close"]
            if hasattr(close_col, "columns"):
                close_col = close_col.iloc[:, 0]

            returns = close_col.pct_change().dropna()
            total_return = (close_col.iloc[-1] / close_col.iloc[0]) - 1
            trading_days = len(returns)

            annualized_return = (1 + total_return) ** (252 / max(trading_days, 1)) - 1
            annualized_vol = returns.std() * np.sqrt(252)
            sharpe = (annualized_return / annualized_vol) if annualized_vol > 0 else 0

            return {
                "total_return": float(total_return),
                "annualized_return": float(annualized_return),
                "volatility": float(annualized_vol),
                "sharpe": float(sharpe),
                "start_price": float(close_col.iloc[0]),
                "end_price": float(close_col.iloc[-1]),
                "trading_days": trading_days,
            }
        except Exception as e:
            return {"error": str(e)}

    # ── Performance Metrics ──────────────────────────────────────────────

    def calculate_performance(self) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics from daily snapshots.

        Metrics:
          - Cumulative return
          - Annualized return (CAGR)
          - Annualized Sharpe ratio
          - Maximum drawdown
          - Win rate (from trades)
          - Profit factor
          - Average trade P&L
        """
        metrics = {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_date": None,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pnl": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "trading_days": 0,
        }

        # From daily snapshots (portfolio-level metrics)
        if len(self.daily_snapshots) >= 2:
            values = [s["portfolio_value"] for s in self.daily_snapshots]
            initial = values[0]
            final = values[-1]

            metrics["total_return"] = (final / initial) - 1 if initial > 0 else 0
            n_days = len(values)
            metrics["trading_days"] = n_days

            if n_days > 1:
                metrics["annualized_return"] = (
                    (final / initial) ** (252 / n_days) - 1
                    if initial > 0 else 0
                )

            # Daily returns for Sharpe
            daily_returns = []
            for i in range(1, len(values)):
                if values[i - 1] > 0:
                    daily_returns.append(values[i] / values[i - 1] - 1)

            if daily_returns:
                mean_daily = np.mean(daily_returns)
                std_daily = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 1e-6
                metrics["sharpe_ratio"] = (
                    mean_daily / max(std_daily, 1e-8) * np.sqrt(252)
                )

            # Maximum drawdown
            peak = values[0]
            max_dd = 0.0
            max_dd_date = None
            for i, v in enumerate(values):
                if v > peak:
                    peak = v
                dd = (peak - v) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
                    max_dd_date = self.daily_snapshots[i].get("date")
            metrics["max_drawdown"] = max_dd
            metrics["max_drawdown_date"] = max_dd_date

        # From trade log (trade-level metrics)
        trades = self._load_trade_log()
        sell_trades = [t for t in trades if t.get("action") == "SELL"]
        if sell_trades:
            pnls = [float(t.get("pnl", 0)) for t in sell_trades]
            metrics["total_trades"] = len(sell_trades)
            metrics["winning_trades"] = sum(1 for p in pnls if p > 0)
            metrics["losing_trades"] = sum(1 for p in pnls if p < 0)
            metrics["win_rate"] = (
                metrics["winning_trades"] / len(sell_trades)
                if sell_trades else 0
            )
            metrics["avg_trade_pnl"] = float(np.mean(pnls))
            metrics["best_trade"] = max(pnls)
            metrics["worst_trade"] = min(pnls)

            gross_profit = sum(p for p in pnls if p > 0)
            gross_loss = abs(sum(p for p in pnls if p < 0))
            metrics["profit_factor"] = (
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            )

        return metrics

    def get_strategy_breakdown(self) -> Dict[str, Dict]:
        """Per-strategy performance breakdown from trade log."""
        trades = self._load_trade_log()
        sell_trades = [t for t in trades if t.get("action") == "SELL"]

        strategies: Dict[str, List[float]] = {}
        for t in sell_trades:
            strat = t.get("strategy", "unknown")
            pnl = float(t.get("pnl", 0))
            strategies.setdefault(strat, []).append(pnl)

        breakdown = {}
        for strat, pnls in strategies.items():
            wins = sum(1 for p in pnls if p > 0)
            breakdown[strat] = {
                "trades": len(pnls),
                "total_pnl": sum(pnls),
                "avg_pnl": float(np.mean(pnls)),
                "win_rate": wins / len(pnls) if pnls else 0,
                "best": max(pnls),
                "worst": min(pnls),
            }
        return breakdown

    # ── Daily Snapshots ──────────────────────────────────────────────────

    def save_daily_snapshot(self):
        """Save today's portfolio snapshot to the daily history."""
        status = self.get_portfolio_status()
        perf = self.calculate_performance()
        spy = self.get_spy_benchmark()

        snapshot = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "timestamp": datetime.now().isoformat(),
            "portfolio_value": status.get("portfolio_value", 0),
            "cash": status.get("cash", 0),
            "position_count": status.get("position_count", 0),
            "daily_pl": status.get("daily_pl", 0),
            "cumulative_return": perf.get("total_return", 0),
            "sharpe_ratio": perf.get("sharpe_ratio", 0),
            "max_drawdown": perf.get("max_drawdown", 0),
            "win_rate": perf.get("win_rate", 0),
            "total_trades": perf.get("total_trades", 0),
            "spy_total_return": spy.get("total_return", 0),
            "spy_sharpe": spy.get("sharpe", 0),
            "alpha": perf.get("total_return", 0) - spy.get("total_return", 0),
        }

        # Avoid duplicate snapshots for the same date
        today = snapshot["date"]
        self.daily_snapshots = [
            s for s in self.daily_snapshots if s.get("date") != today
        ]
        self.daily_snapshots.append(snapshot)
        self._save_daily_snapshots()
        return snapshot

    def _load_daily_snapshots(self) -> List[Dict]:
        """Load daily snapshot history from JSON."""
        if os.path.exists(self.daily_snapshots_path):
            try:
                with open(self.daily_snapshots_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save_daily_snapshots(self):
        """Persist daily snapshots to JSON."""
        with open(self.daily_snapshots_path, "w") as f:
            json.dump(self.daily_snapshots, f, indent=2)

    # ── Trade Log Reader ─────────────────────────────────────────────────

    def _load_trade_log(self) -> List[Dict]:
        """Load trades from the paper trading CSV log."""
        if not os.path.exists(self.trade_log_path):
            return []
        try:
            with open(self.trade_log_path, "r") as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception:
            return []

    # ── Display ──────────────────────────────────────────────────────────

    def print_dashboard(self):
        """Print a formatted dashboard to the console."""
        status = self.get_portfolio_status()
        perf = self.calculate_performance()
        spy = self.get_spy_benchmark()
        breakdown = self.get_strategy_breakdown()

        w = 60
        print("\n" + "=" * w)
        print("  PAPER TRADING DASHBOARD".center(w))
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(w))
        print("=" * w)

        # Portfolio overview
        print("\n  PORTFOLIO")
        print(f"  {'Value:':<25} ${status.get('portfolio_value', 0):>12,.2f}")
        print(f"  {'Cash:':<25} ${status.get('cash', 0):>12,.2f}")
        print(f"  {'Buying Power:':<25} ${status.get('buying_power', 0):>12,.2f}")
        print(f"  {'Open Positions:':<25} {status.get('position_count', 0):>12}")
        daily_pl = status.get('daily_pl', 0)
        sign = "+" if daily_pl >= 0 else ""
        print(f"  {'Daily P&L:':<25} {sign}${daily_pl:>11,.2f}")

        # Positions
        positions = status.get("positions", [])
        if positions:
            print(f"\n  {'POSITIONS'}")
            print(f"  {'Symbol':<8} {'Qty':>6} {'Entry':>10} {'Current':>10} {'P&L':>10} {'%':>8}")
            print("  " + "-" * 54)
            for p in positions:
                pnl_str = f"${p['unrealized_pl']:+,.2f}"
                pct_str = f"{p['unrealized_plpc']:+.2%}"
                print(f"  {p['symbol']:<8} {p['qty']:>6} "
                      f"${p['avg_entry_price']:>9,.2f} "
                      f"${p['current_price']:>9,.2f} "
                      f"{pnl_str:>10} {pct_str:>8}")

        # Performance metrics
        print(f"\n  PERFORMANCE METRICS")
        print(f"  {'Total Return:':<25} {perf['total_return']:>11.2%}")
        print(f"  {'Annualized Return:':<25} {perf['annualized_return']:>11.2%}")
        print(f"  {'Sharpe Ratio:':<25} {perf['sharpe_ratio']:>11.2f}")
        print(f"  {'Max Drawdown:':<25} {perf['max_drawdown']:>11.2%}")
        print(f"  {'Win Rate:':<25} {perf['win_rate']:>11.2%}")
        print(f"  {'Profit Factor:':<25} {perf['profit_factor']:>11.2f}")
        print(f"  {'Total Trades:':<25} {perf['total_trades']:>11}")
        print(f"  {'Avg Trade P&L:':<25} ${perf['avg_trade_pnl']:>10,.2f}")

        # SPY benchmark comparison
        if "error" not in spy:
            print(f"\n  VS SPY BENCHMARK")
            print(f"  {'SPY Return:':<25} {spy.get('total_return', 0):>11.2%}")
            print(f"  {'SPY Sharpe:':<25} {spy.get('sharpe', 0):>11.2f}")
            alpha = perf["total_return"] - spy.get("total_return", 0)
            print(f"  {'Alpha (vs SPY):':<25} {alpha:>11.2%}")

        # Strategy breakdown
        if breakdown:
            print(f"\n  STRATEGY BREAKDOWN")
            print(f"  {'Strategy':<25} {'Trades':>7} {'Win%':>7} {'Total P&L':>12}")
            print("  " + "-" * 53)
            for strat, stats in breakdown.items():
                print(f"  {strat:<25} {stats['trades']:>7} "
                      f"{stats['win_rate']:>6.1%} "
                      f"${stats['total_pnl']:>11,.2f}")

        print("\n" + "=" * w)

    def run_live(self, interval: int = 60):
        """Continuously refresh dashboard at the given interval (seconds)."""
        print("Live dashboard mode. Press Ctrl+C to stop.\n")
        try:
            while True:
                os.system("cls" if os.name == "nt" else "clear")
                self.print_dashboard()
                self.save_daily_snapshot()
                time.sleep(interval)
        except KeyboardInterrupt:
            print("\nDashboard stopped.")

    def export_snapshot(self) -> str:
        """Export current snapshot to a timestamped JSON file."""
        snapshot = self.save_daily_snapshot()
        filename = f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.json"
        filepath = os.path.join(self.snapshot_dir, filename)
        with open(filepath, "w") as f:
            json.dump(snapshot, f, indent=2)
        print(f"Snapshot exported to: {filepath}")
        return filepath


# ── CLI Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Paper Trading Dashboard")
    parser.add_argument("--live", action="store_true",
                        help="Continuously refresh dashboard")
    parser.add_argument("--interval", type=int, default=60,
                        help="Refresh interval in seconds (default: 60)")
    parser.add_argument("--export", action="store_true",
                        help="Export snapshot to JSON and exit")
    args = parser.parse_args()

    dashboard = PaperTradingDashboard()

    if args.export:
        dashboard.export_snapshot()
    elif args.live:
        dashboard.run_live(interval=args.interval)
    else:
        dashboard.print_dashboard()
