"""
Results Collector — End-of-day data collection and feedback loop.

Collects daily trading results, feeds them back to the self_improver module,
tracks per-strategy performance over time, and updates strategy weights
based on realized performance.

Designed to run once at market close (or shortly after) to process the
day's trades and prepare the system for the next session.

Usage:
    python results_collector.py                  # Run EOD collection
    python results_collector.py --report         # Print performance report
    python results_collector.py --reset-weights  # Reset weights to defaults
"""

import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    tradeapi = None

from config import ALPACA_API_KEY, ALPACA_BASE_URL, ALPACA_SECRET_KEY, config
from logger import TradeLogger
from self_improver import SelfImprover


class ResultsCollector:
    """
    End-of-day results collection and self-improvement feedback loop.

    Pipeline:
      1. Fetch today's filled orders from Alpaca (or local trade log)
      2. Match buy/sell pairs to compute realized P&L per trade
      3. Attribute each trade to its originating strategy
      4. Feed results to self_improver.record_experience()
      5. Trigger weight update based on accumulated performance
      6. Save per-strategy performance history for dashboarding
    """

    def __init__(self):
        self.logger = TradeLogger()
        self.self_improver = SelfImprover(self.logger)

        # Alpaca connection (optional — falls back to local log)
        self.api = None
        self.connected = False
        self._connect_alpaca()

        # Paths
        self.trade_log_path = os.path.join(config.log_dir, "paper_trades.csv")
        self.results_dir = os.path.join(config.log_dir, "daily_results")
        self.strategy_history_path = os.path.join(
            config.log_dir, "strategy_performance_history.json"
        )
        os.makedirs(self.results_dir, exist_ok=True)

        # Load historical strategy performance
        self.strategy_history: List[Dict] = self._load_strategy_history()

    def _connect_alpaca(self):
        """Connect to Alpaca for order history."""
        if tradeapi is None or ALPACA_API_KEY == "YOUR_ALPACA_API_KEY":
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

    def collect_daily_results(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Collect and process results for a given trading day.

        Steps:
          1. Load today's trades from log (and optionally verify with Alpaca)
          2. Pair buys with sells to compute per-trade P&L
          3. Feed each completed trade to self_improver
          4. Trigger weight update
          5. Save daily summary

        Args:
            date: Date string (YYYY-MM-DD). Defaults to today.

        Returns:
            Daily results summary dict.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        self.logger.info(f"{'='*50}")
        self.logger.info(f"COLLECTING RESULTS FOR {date}")
        self.logger.info(f"{'='*50}")

        # Load trades for the day
        trades = self._get_trades_for_date(date)
        self.logger.info(f"Found {len(trades)} trades for {date}")

        if not trades:
            self.logger.info("No trades to process.")
            return {"date": date, "trades": 0, "pnl": 0}

        # Pair trades and compute P&L
        completed_trades = self._pair_trades(trades)
        self.logger.info(f"Matched {len(completed_trades)} completed round-trips")

        # Feed to self-improver
        total_pnl = 0.0
        strategy_pnls: Dict[str, List[float]] = defaultdict(list)

        for trade in completed_trades:
            pnl = trade["pnl"]
            total_pnl += pnl
            strategy_pnls[trade["strategy"]].append(pnl)

            # Record in self-improver
            self.self_improver.record_experience(
                symbol=trade["symbol"],
                strategy=trade["strategy"],
                action="SELL",
                signal_strength=trade.get("signal_strength", 0.0),
                entry_price=trade["entry_price"],
                exit_price=trade["exit_price"],
                holding_period_hours=trade.get("holding_hours", 0),
                market_regime=trade.get("regime", "unknown"),
            )

        # Update weights
        regime = self._get_dominant_regime(trades)
        new_weights = self.self_improver.update_weights(regime_name=regime)

        # Build per-strategy summary
        strategy_summary = {}
        for strat, pnls in strategy_pnls.items():
            wins = sum(1 for p in pnls if p > 0)
            strategy_summary[strat] = {
                "trades": len(pnls),
                "total_pnl": sum(pnls),
                "avg_pnl": float(np.mean(pnls)),
                "win_rate": wins / len(pnls) if pnls else 0,
                "weight": new_weights.get(strat, 0),
            }

        # Daily summary
        daily_result = {
            "date": date,
            "timestamp": datetime.now().isoformat(),
            "total_trades": len(trades),
            "completed_roundtrips": len(completed_trades),
            "total_pnl": total_pnl,
            "strategy_breakdown": strategy_summary,
            "updated_weights": dict(new_weights),
            "dominant_regime": regime,
        }

        # Persist
        self._save_daily_result(daily_result)
        self._update_strategy_history(daily_result)

        # Log summary
        self.logger.info(f"\nDAILY SUMMARY ({date})")
        self.logger.info(f"  Total P&L: ${total_pnl:+,.2f}")
        self.logger.info(f"  Round-trips: {len(completed_trades)}")
        self.logger.info(f"  Regime: {regime}")
        for strat, stats in strategy_summary.items():
            self.logger.info(
                f"  {strat}: {stats['trades']} trades, "
                f"${stats['total_pnl']:+,.2f}, "
                f"win={stats['win_rate']:.0%}, "
                f"new_weight={stats['weight']:.3f}"
            )

        return daily_result

    def _get_trades_for_date(self, date: str) -> List[Dict]:
        """Load all trades for a specific date from the paper trades CSV."""
        if not os.path.exists(self.trade_log_path):
            return []

        trades = []
        try:
            with open(self.trade_log_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts = row.get("timestamp", "")
                    if ts.startswith(date):
                        trades.append(row)
        except Exception as e:
            self.logger.error(f"Error reading trade log: {e}")

        return trades

    def _pair_trades(self, trades: List[Dict]) -> List[Dict]:
        """
        Match BUY and SELL trades into completed round-trips.

        Returns list of dicts with entry/exit prices and P&L.
        """
        buys: Dict[str, List[Dict]] = defaultdict(list)
        completed = []

        for trade in sorted(trades, key=lambda t: t.get("timestamp", "")):
            symbol = trade.get("symbol", "")
            action = trade.get("action", "")

            if action == "BUY":
                buys[symbol].append(trade)
            elif action == "SELL" and buys[symbol]:
                buy_trade = buys[symbol].pop(0)  # FIFO matching
                entry_price = float(buy_trade.get("filled_price",
                                    buy_trade.get("price", 0)))
                exit_price = float(trade.get("filled_price",
                                   trade.get("price", 0)))
                qty = int(trade.get("quantity", 0))

                pnl = (exit_price - entry_price) * qty

                # Calculate holding time
                try:
                    buy_time = datetime.fromisoformat(buy_trade.get("timestamp", ""))
                    sell_time = datetime.fromisoformat(trade.get("timestamp", ""))
                    holding_hours = (sell_time - buy_time).total_seconds() / 3600
                except (ValueError, TypeError):
                    holding_hours = 0

                completed.append({
                    "symbol": symbol,
                    "strategy": buy_trade.get("strategy", "unknown"),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "quantity": qty,
                    "pnl": pnl,
                    "pnl_pct": (exit_price / entry_price - 1) if entry_price > 0 else 0,
                    "holding_hours": holding_hours,
                    "regime": trade.get("regime", "unknown"),
                    "signal_strength": float(buy_trade.get("signal_strength", 0)),
                    "buy_time": buy_trade.get("timestamp"),
                    "sell_time": trade.get("timestamp"),
                })

        return completed

    def _get_dominant_regime(self, trades: List[Dict]) -> str:
        """Find the most common regime label among today's trades."""
        regimes = [t.get("regime", "unknown") for t in trades if t.get("regime")]
        if not regimes:
            return "unknown"
        from collections import Counter
        return Counter(regimes).most_common(1)[0][0]

    # ── Persistence ──────────────────────────────────────────────────────

    def _save_daily_result(self, result: Dict):
        """Save daily result to a dated JSON file."""
        date = result.get("date", datetime.now().strftime("%Y-%m-%d"))
        filepath = os.path.join(self.results_dir, f"results_{date}.json")
        with open(filepath, "w") as f:
            json.dump(result, f, indent=2)
        self.logger.info(f"Daily results saved to {filepath}")

    def _load_strategy_history(self) -> List[Dict]:
        """Load cumulative strategy performance history."""
        if os.path.exists(self.strategy_history_path):
            try:
                with open(self.strategy_history_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save_strategy_history(self):
        """Persist strategy performance history."""
        with open(self.strategy_history_path, "w") as f:
            json.dump(self.strategy_history, f, indent=2)

    def _update_strategy_history(self, daily_result: Dict):
        """Append today's strategy performance to the rolling history."""
        entry = {
            "date": daily_result["date"],
            "weights": daily_result["updated_weights"],
            "strategies": {},
        }
        for strat, stats in daily_result.get("strategy_breakdown", {}).items():
            entry["strategies"][strat] = {
                "trades": stats["trades"],
                "pnl": stats["total_pnl"],
                "win_rate": stats["win_rate"],
                "weight": stats["weight"],
            }

        self.strategy_history.append(entry)
        self._save_strategy_history()

    # ── Reporting ────────────────────────────────────────────────────────

    def print_performance_report(self, days: int = 30):
        """Print a multi-day performance report."""
        print(f"\n{'='*60}")
        print(f"  STRATEGY PERFORMANCE REPORT (Last {days} days)")
        print(f"{'='*60}")

        # Aggregate from history
        recent = self.strategy_history[-days:]
        if not recent:
            print("  No data available yet. Run collect_daily_results first.")
            return

        # Per-strategy aggregation
        agg: Dict[str, Dict] = defaultdict(lambda: {
            "trades": 0, "pnl": 0.0, "wins": 0, "days_active": 0,
        })
        weight_history: Dict[str, List[float]] = defaultdict(list)

        for entry in recent:
            for strat, stats in entry.get("strategies", {}).items():
                agg[strat]["trades"] += stats.get("trades", 0)
                agg[strat]["pnl"] += stats.get("pnl", 0)
                agg[strat]["wins"] += int(
                    stats.get("win_rate", 0) * stats.get("trades", 0)
                )
                agg[strat]["days_active"] += 1
            for strat, w in entry.get("weights", {}).items():
                weight_history[strat].append(w)

        print(f"\n  {'Strategy':<25} {'Trades':>7} {'P&L':>12} "
              f"{'Win%':>7} {'Avg Wt':>8} {'Cur Wt':>8}")
        print("  " + "-" * 69)

        current_weights = self.self_improver.weights
        for strat in sorted(agg.keys()):
            stats = agg[strat]
            win_rate = stats["wins"] / stats["trades"] if stats["trades"] > 0 else 0
            avg_wt = np.mean(weight_history.get(strat, [0]))
            cur_wt = current_weights.get(strat, 0)
            print(f"  {strat:<25} {stats['trades']:>7} "
                  f"${stats['pnl']:>11,.2f} "
                  f"{win_rate:>6.1%} "
                  f"{avg_wt:>7.3f} "
                  f"{cur_wt:>7.3f}")

        # Overall
        total_pnl = sum(s["pnl"] for s in agg.values())
        total_trades = sum(s["trades"] for s in agg.values())
        total_wins = sum(s["wins"] for s in agg.values())
        overall_wr = total_wins / total_trades if total_trades > 0 else 0

        print("  " + "-" * 69)
        print(f"  {'TOTAL':<25} {total_trades:>7} "
              f"${total_pnl:>11,.2f} "
              f"{overall_wr:>6.1%}")

        # Weight evolution
        print(f"\n  WEIGHT EVOLUTION (last {min(days, len(recent))} days)")
        for strat in sorted(weight_history.keys()):
            wts = weight_history[strat]
            if len(wts) >= 2:
                delta = wts[-1] - wts[0]
                arrow = "+" if delta > 0 else ""
                print(f"  {strat:<25} {wts[0]:.3f} -> {wts[-1]:.3f} ({arrow}{delta:.3f})")

        # Self-improver report
        improver_report = self.self_improver.get_report()
        print(f"\n  SELF-IMPROVER STATUS")
        print(f"  Total experiences: {improver_report['total_experiences']}")
        print(f"  Regime weights available: "
              f"{list(improver_report.get('regime_weights', {}).keys())}")

        print(f"\n{'='*60}\n")

    def reset_weights(self):
        """Reset strategy weights to the defaults from config."""
        self.self_improver.weights = dict(config.strategy_weights)
        self.self_improver._save_state()
        self.logger.info("Strategy weights reset to defaults:")
        for name, w in config.strategy_weights.items():
            self.logger.info(f"  {name}: {w:.3f}")

    # ── Alpaca History (optional verification) ───────────────────────────

    def fetch_alpaca_orders(self, date: Optional[str] = None) -> List[Dict]:
        """Fetch filled orders from Alpaca for cross-referencing."""
        if not self.connected or not self.api:
            return []

        try:
            if date:
                after = f"{date}T00:00:00Z"
                until = f"{date}T23:59:59Z"
                orders = self.api.list_orders(
                    status="filled", after=after, until=until, limit=500
                )
            else:
                orders = self.api.list_orders(status="filled", limit=100)

            return [
                {
                    "id": o.id,
                    "symbol": o.symbol,
                    "side": o.side,
                    "qty": int(o.qty),
                    "filled_avg_price": float(o.filled_avg_price or 0),
                    "filled_at": str(o.filled_at),
                    "status": o.status,
                }
                for o in orders
            ]
        except Exception as e:
            self.logger.error(f"Failed to fetch Alpaca orders: {e}")
            return []


# ── CLI Entry Point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Results Collector")
    parser.add_argument("--date", type=str, default=None,
                        help="Date to collect (YYYY-MM-DD). Default: today")
    parser.add_argument("--report", action="store_true",
                        help="Print performance report")
    parser.add_argument("--days", type=int, default=30,
                        help="Days to include in report (default: 30)")
    parser.add_argument("--reset-weights", action="store_true",
                        help="Reset strategy weights to config defaults")
    args = parser.parse_args()

    collector = ResultsCollector()

    if args.reset_weights:
        collector.reset_weights()
    elif args.report:
        collector.print_performance_report(days=args.days)
    else:
        collector.collect_daily_results(date=args.date)
