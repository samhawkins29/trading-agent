"""
Backtesting Framework.

Test strategies on historical data before risking real capital.
Simulates the full agent pipeline: data → signals → risk → execution.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import config
from data_fetcher import DataFetcher
from logger import TradeLogger
from risk_manager import RiskManager
from self_improver import SelfImprover
from strategies.mean_reversion import MeanReversionStrategy, Signal
from strategies.momentum import MomentumStrategy
from strategies.pattern_recognition import PatternRecognitionStrategy
from strategies.sentiment import SentimentStrategy


class Backtester:
    """
    Historical backtesting engine.

    Walks through historical data day-by-day, simulating the agent's
    full decision pipeline. Tracks P&L, drawdown, Sharpe, and per-strategy
    performance.
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        start_date: str = config.backtest_start,
        end_date: str = config.backtest_end,
        initial_capital: float = config.initial_capital,
    ):
        self.symbols = symbols or config.symbols
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital

        self.logger = TradeLogger(log_dir="logs/backtest")
        self.data_fetcher = DataFetcher()
        self.risk_manager = RiskManager(initial_capital, self.logger)
        self.self_improver = SelfImprover(self.logger, save_path="logs/backtest")

        # Strategies
        self.strategies = {
            "mean_reversion": MeanReversionStrategy(),
            "momentum": MomentumStrategy(),
            "pattern_recognition": PatternRecognitionStrategy(),
        }
        # Note: sentiment is excluded from backtest (no historical news data)

        # Results tracking
        self.equity_curve: List[float] = []
        self.daily_returns: List[float] = []
        self.trade_log: List[Dict] = []
        self.daily_log: List[Dict] = []

    def run(self) -> Dict:
        """
        Run the backtest across all symbols.
        Returns a results dict with performance metrics.
        """
        self.logger.info(f"Starting backtest: {self.start_date} to {self.end_date}")
        self.logger.info(f"Symbols: {self.symbols}")
        self.logger.info(f"Capital: ${self.initial_capital:,.2f}")
        start_time = time.time()

        # Fetch all historical data
        all_data = {}
        for sym in self.symbols:
            df = self.data_fetcher.get_historical(sym, period="3y", interval="1d")
            if not df.empty:
                df = DataFetcher.compute_indicators(df)
                # Filter to backtest range
                df = df.loc[self.start_date:self.end_date]
                if not df.empty:
                    all_data[sym] = df
                    self.logger.info(f"  {sym}: {len(df)} days loaded")

        if not all_data:
            self.logger.error("No data loaded for any symbol")
            return {"error": "No data"}

        # Get common date range
        all_dates = set()
        for df in all_data.values():
            all_dates.update(df.index.strftime("%Y-%m-%d"))
        dates = sorted(all_dates)

        self.logger.info(f"Backtesting {len(dates)} trading days...")

        # ── Day-by-Day Simulation ────────────────────────────────────
        prev_value = self.initial_capital

        for i, date_str in enumerate(dates):
            day_actions = {"date": date_str, "buys": [], "sells": []}

            # Check stop-losses on existing positions
            current_prices = {}
            for sym, df in all_data.items():
                if date_str in df.index.strftime("%Y-%m-%d").values:
                    mask = df.index.strftime("%Y-%m-%d") == date_str
                    if mask.any():
                        current_prices[sym] = float(df.loc[mask, "Close"].iloc[-1])

            sl_tp = self.risk_manager.check_stop_loss_take_profit(current_prices)
            for sym in sl_tp:
                if sym in current_prices:
                    pos = self.risk_manager.positions.get(sym)
                    if pos:
                        pnl = self.risk_manager.close_position(sym, current_prices[sym])
                        self.trade_log.append({
                            "date": date_str, "symbol": sym, "action": "SELL",
                            "price": current_prices[sym], "pnl": pnl or 0,
                            "reason": "stop_loss_take_profit",
                        })
                        day_actions["sells"].append(sym)

            # Analyze each symbol
            for sym in self.symbols:
                if sym not in all_data:
                    continue
                df = all_data[sym]
                mask = df.index.strftime("%Y-%m-%d") <= date_str
                df_up_to = df[mask]

                if len(df_up_to) < 60:
                    continue

                # Generate signals
                signals = {}
                signals["mean_reversion"] = self.strategies[
                    "mean_reversion"
                ].generate_signal(sym, df_up_to)
                signals["momentum"] = self.strategies[
                    "momentum"
                ].generate_signal(sym, df_up_to)
                signals["pattern_recognition"] = self.strategies[
                    "pattern_recognition"
                ].generate_signal(sym, df_up_to)

                # Weighted combination
                weights = self.self_improver.weights
                combined = sum(
                    weights.get(n, 0) * s.strength
                    for n, s in signals.items()
                    if n in weights
                )
                combined = np.clip(combined, -1.0, 1.0)

                price = float(df_up_to["Close"].iloc[-1])
                atr = float(df_up_to["ATR"].iloc[-1]) if "ATR" in df_up_to.columns else price * 0.02
                vol = float(df_up_to["returns"].std()) if "returns" in df_up_to.columns else 0.02

                can_trade, _ = self.risk_manager.can_trade(sym)

                if combined > 0.3 and can_trade and sym not in self.risk_manager.positions:
                    qty = self.risk_manager.calculate_position_size(
                        sym, price, combined, vol
                    )
                    if qty > 0:
                        dominant = max(signals.items(), key=lambda x: abs(x[1].strength))
                        self.risk_manager.open_position(sym, qty, price, dominant[0], atr)
                        self.trade_log.append({
                            "date": date_str, "symbol": sym, "action": "BUY",
                            "quantity": qty, "price": price, "strategy": dominant[0],
                        })
                        day_actions["buys"].append(sym)

                elif combined < -0.3 and sym in self.risk_manager.positions:
                    pos = self.risk_manager.positions[sym]
                    pnl = self.risk_manager.close_position(sym, price)
                    self.self_improver.record_experience(
                        symbol=sym, strategy=pos.strategy, action="SELL",
                        signal_strength=combined, entry_price=pos.entry_price,
                        exit_price=price,
                    )
                    self.trade_log.append({
                        "date": date_str, "symbol": sym, "action": "SELL",
                        "price": price, "pnl": pnl or 0, "strategy": pos.strategy,
                    })
                    day_actions["sells"].append(sym)

            # End-of-day equity
            total_value = self.risk_manager.current_capital
            for sym, pos in self.risk_manager.positions.items():
                if sym in current_prices:
                    total_value += pos.quantity * current_prices[sym]
                else:
                    total_value += pos.quantity * pos.entry_price

            self.equity_curve.append(total_value)
            daily_ret = (total_value - prev_value) / prev_value if prev_value > 0 else 0
            self.daily_returns.append(daily_ret)
            prev_value = total_value

            self.daily_log.append(day_actions)

            # Periodic weight updates
            if (i + 1) % 20 == 0:
                self.self_improver.update_weights()

        elapsed = time.time() - start_time

        # ── Compute Results ──────────────────────────────────────────
        results = self._compute_metrics(elapsed)
        self._print_report(results)
        return results

    def _compute_metrics(self, elapsed: float) -> Dict:
        """Compute backtest performance metrics."""
        equity = np.array(self.equity_curve)
        returns = np.array(self.daily_returns)

        total_return = (
            (equity[-1] - self.initial_capital) / self.initial_capital
            if len(equity) > 0
            else 0
        )

        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(equity) if len(equity) > 0 else np.array([0])
        drawdowns = (peak - equity) / np.where(peak > 0, peak, 1)
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

        # Win rate
        pnls = [t.get("pnl", 0) for t in self.trade_log if t["action"] == "SELL"]
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls) if pnls else 0

        # Average win / loss
        winning_pnls = [p for p in pnls if p > 0]
        losing_pnls = [p for p in pnls if p < 0]
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0

        return {
            "period": f"{self.start_date} to {self.end_date}",
            "initial_capital": self.initial_capital,
            "final_value": float(equity[-1]) if len(equity) > 0 else self.initial_capital,
            "total_return": total_return,
            "annualized_return": total_return / max(len(returns) / 252, 0.01),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "total_trades": len(self.trade_log),
            "win_rate": win_rate,
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else float("inf"),
            "elapsed_seconds": elapsed,
            "final_weights": dict(self.self_improver.weights),
        }

    def _print_report(self, results: Dict):
        """Print a formatted backtest report."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("         BACKTEST RESULTS")
        self.logger.info("=" * 60)
        self.logger.info(f"  Period:            {results['period']}")
        self.logger.info(f"  Initial Capital:   ${results['initial_capital']:>12,.2f}")
        self.logger.info(f"  Final Value:       ${results['final_value']:>12,.2f}")
        self.logger.info(f"  Total Return:      {results['total_return']:>11.2%}")
        self.logger.info(f"  Annualized Return: {results['annualized_return']:>11.2%}")
        self.logger.info(f"  Sharpe Ratio:      {results['sharpe_ratio']:>11.2f}")
        self.logger.info(f"  Max Drawdown:      {results['max_drawdown']:>11.2%}")
        self.logger.info(f"  Total Trades:      {results['total_trades']:>11d}")
        self.logger.info(f"  Win Rate:          {results['win_rate']:>11.2%}")
        self.logger.info(f"  Avg Win:           ${results['avg_win']:>11.2f}")
        self.logger.info(f"  Avg Loss:          ${results['avg_loss']:>11.2f}")
        self.logger.info(f"  Profit Factor:     {results['profit_factor']:>11.2f}")
        self.logger.info(f"  Elapsed:           {results['elapsed_seconds']:>11.1f}s")
        self.logger.info("-" * 60)
        self.logger.info("  Final Strategy Weights:")
        for name, weight in results["final_weights"].items():
            self.logger.info(f"    {name:25s} {weight:.3f}")
        self.logger.info("=" * 60)

    def get_equity_curve(self) -> pd.DataFrame:
        """Return equity curve as a DataFrame."""
        return pd.DataFrame({
            "equity": self.equity_curve,
        })
