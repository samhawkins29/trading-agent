"""
Backtesting Framework — Redesigned for New Strategy Suite.

Key improvements:
  1. Supports 10-year backtest periods (yfinance max history)
  2. Regime-based dynamic strategy weighting during backtest
  3. Factor Momentum strategy included (replaces sentiment)
  4. Kelly criterion position sizing with rolling estimation
  5. Volatility targeting throughout
  6. Comprehensive metrics: annualized return, Sharpe, Sortino, Calmar, max DD
  7. Per-strategy attribution analysis
  8. SPY benchmark comparison
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
from strategies.pattern_recognition import (
    MarketRegime,
    PatternRecognitionStrategy,
)
from strategies.sentiment import SentimentStrategy


class Backtester:
    """
    Historical backtesting engine with regime-aware strategy weighting.

    Walks through historical data day-by-day, simulating the full
    agent pipeline including regime detection and dynamic weighting.
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

        # All strategies including Factor Momentum
        self.strategies = {
            "mean_reversion": MeanReversionStrategy(),
            "momentum": MomentumStrategy(),
            "sentiment": SentimentStrategy(),
            "pattern_recognition": PatternRecognitionStrategy(),
        }

        # Regime detector
        self.regime_detector = self.strategies["pattern_recognition"]
        self.current_regime = MarketRegime.MEAN_REVERTING

        # Results tracking
        self.equity_curve: List[float] = []
        self.daily_returns: List[float] = []
        self.trade_log: List[Dict] = []
        self.daily_log: List[Dict] = []
        self.regime_log: List[str] = []

        # Per-strategy P&L tracking
        self.strategy_pnl: Dict[str, float] = {
            name: 0.0 for name in self.strategies
        }
        self.strategy_trades: Dict[str, int] = {
            name: 0 for name in self.strategies
        }

        # SPY benchmark
        self.spy_equity: List[float] = []

    def run(self) -> Dict:
        """
        Run the full backtest with regime-based weighting.
        Returns comprehensive results dict.
        """
        self.logger.info(f"Starting backtest: {self.start_date} to {self.end_date}")
        self.logger.info(f"Symbols: {self.symbols}")
        self.logger.info(f"Capital: ${self.initial_capital:,.2f}")
        self.logger.info(f"Kelly enabled: {config.use_kelly}")
        self.logger.info(f"Vol target: {config.vol_target:.0%}")
        start_time = time.time()

        # Fetch all historical data
        all_data = {}
        for sym in self.symbols:
            df = self.data_fetcher.get_historical(sym, period="max", interval="1d")
            if not df.empty:
                df = DataFetcher.compute_indicators(df)
                df = df.loc[self.start_date:self.end_date]
                if not df.empty:
                    all_data[sym] = df
                    self.logger.info(f"  {sym}: {len(df)} days loaded")

        if not all_data:
            self.logger.error("No data loaded for any symbol")
            return {"error": "No data"}

        # SPY benchmark data
        spy_data = all_data.get("SPY")
        if spy_data is None:
            try:
                spy_df = self.data_fetcher.get_historical("SPY", period="max", interval="1d")
                spy_df = DataFetcher.compute_indicators(spy_df)
                spy_data = spy_df.loc[self.start_date:self.end_date]
            except Exception:
                spy_data = None

        # Get common date range
        all_dates = set()
        for df in all_data.values():
            all_dates.update(df.index.strftime("%Y-%m-%d"))
        dates = sorted(all_dates)

        self.logger.info(f"Backtesting {len(dates)} trading days...")

        # -- Day-by-Day Simulation --
        prev_value = self.initial_capital
        spy_start_price = None

        for i, date_str in enumerate(dates):
            day_actions = {"date": date_str, "buys": [], "sells": []}

            # Get current prices
            current_prices = {}
            for sym, df in all_data.items():
                mask = df.index.strftime("%Y-%m-%d") == date_str
                if mask.any():
                    current_prices[sym] = float(df.loc[mask, "Close"].iloc[-1])

            # SPY tracking
            if spy_data is not None:
                spy_mask = spy_data.index.strftime("%Y-%m-%d") == date_str
                if spy_mask.any():
                    spy_price = float(spy_data.loc[spy_mask, "Close"].iloc[-1])
                    if spy_start_price is None:
                        spy_start_price = spy_price
                    self.spy_equity.append(
                        self.initial_capital * (spy_price / spy_start_price)
                    )

            # Regime detection (using SPY data up to current date)
            if spy_data is not None:
                spy_up_to = spy_data[spy_data.index.strftime("%Y-%m-%d") <= date_str]
                if len(spy_up_to) >= 100:
                    self.current_regime = self.regime_detector.detect_regime(spy_up_to)
            self.regime_log.append(self.current_regime.value)

            # Get regime-adjusted weights
            active_weights = self._get_active_weights()

            # Check stop-losses
            sl_tp = self.risk_manager.check_stop_loss_take_profit(current_prices)
            for sym in sl_tp:
                if sym in current_prices:
                    pos = self.risk_manager.positions.get(sym)
                    if pos:
                        pnl = self.risk_manager.close_position(sym, current_prices[sym])
                        pnl_val = pnl or 0
                        self.trade_log.append({
                            "date": date_str, "symbol": sym, "action": "SELL",
                            "price": current_prices[sym], "pnl": pnl_val,
                            "reason": "stop_loss_take_profit",
                            "strategy": pos.strategy,
                        })
                        if pos.strategy in self.strategy_pnl:
                            self.strategy_pnl[pos.strategy] += pnl_val
                        day_actions["sells"].append(sym)

            # Analyze each symbol
            for sym in self.symbols:
                if sym not in all_data:
                    continue
                df = all_data[sym]
                mask = df.index.strftime("%Y-%m-%d") <= date_str
                df_up_to = df[mask]

                if len(df_up_to) < 100:
                    continue

                # Generate signals from all strategies
                signals = {}
                signals["mean_reversion"] = self.strategies[
                    "mean_reversion"
                ].generate_signal(sym, df_up_to)
                signals["momentum"] = self.strategies[
                    "momentum"
                ].generate_signal(sym, df_up_to)
                signals["sentiment"] = self.strategies[
                    "sentiment"
                ].generate_signal(sym, df_up_to)
                signals["pattern_recognition"] = self.strategies[
                    "pattern_recognition"
                ].generate_signal(sym, df_up_to)

                # Weighted combination using regime-adjusted weights
                combined = sum(
                    active_weights.get(n, 0) * s.strength
                    for n, s in signals.items()
                )
                combined = np.clip(combined, -1.0, 1.0)

                price = float(df_up_to["Close"].iloc[-1])
                atr = float(df_up_to["ATR"].iloc[-1]) if "ATR" in df_up_to.columns else price * 0.02
                vol = float(df_up_to["returns"].std()) if "returns" in df_up_to.columns else 0.02

                can_trade, _ = self.risk_manager.can_trade(sym)
                regime_str = self.current_regime.value

                if combined > 0.25 and can_trade and sym not in self.risk_manager.positions:
                    qty = self.risk_manager.calculate_position_size(
                        sym, price, combined, vol, regime_str
                    )
                    if qty > 0:
                        dominant = max(signals.items(), key=lambda x: abs(x[1].strength))
                        self.risk_manager.open_position(sym, qty, price, dominant[0], atr)
                        self.trade_log.append({
                            "date": date_str, "symbol": sym, "action": "BUY",
                            "quantity": qty, "price": price, "strategy": dominant[0],
                        })
                        if dominant[0] in self.strategy_trades:
                            self.strategy_trades[dominant[0]] += 1
                        day_actions["buys"].append(sym)

                elif combined < -0.25 and sym in self.risk_manager.positions:
                    pos = self.risk_manager.positions[sym]
                    pnl = self.risk_manager.close_position(sym, price)
                    pnl_val = pnl or 0
                    self.self_improver.record_experience(
                        symbol=sym, strategy=pos.strategy, action="SELL",
                        signal_strength=combined, entry_price=pos.entry_price,
                        exit_price=price, market_regime=regime_str,
                    )
                    self.trade_log.append({
                        "date": date_str, "symbol": sym, "action": "SELL",
                        "price": price, "pnl": pnl_val, "strategy": pos.strategy,
                    })
                    if pos.strategy in self.strategy_pnl:
                        self.strategy_pnl[pos.strategy] += pnl_val
                    if pos.strategy in self.strategy_trades:
                        self.strategy_trades[pos.strategy] += 1
                    day_actions["sells"].append(sym)

            # End-of-day equity
            total_value = self.risk_manager.current_capital
            for sym, pos in self.risk_manager.positions.items():
                p = current_prices.get(sym, pos.entry_price)
                total_value += pos.quantity * p

            self.equity_curve.append(total_value)
            daily_ret = (total_value - prev_value) / prev_value if prev_value > 0 else 0
            self.daily_returns.append(daily_ret)
            prev_value = total_value

            self.daily_log.append(day_actions)

            # Periodic weight updates
            if (i + 1) % 20 == 0:
                self.self_improver.update_weights(regime_name=self.current_regime.value)

        elapsed = time.time() - start_time

        # Compute results
        results = self._compute_metrics(elapsed)
        self._print_report(results)
        return results

    def _get_active_weights(self) -> Dict[str, float]:
        """Get regime-blended strategy weights for the current regime."""
        base = self.self_improver.weights
        if not config.use_regime_weighting:
            return base

        regime_rec = self.regime_detector.get_regime_weights(self.current_regime)
        alpha = config.regime_blend_alpha

        blended = {}
        for name in base:
            b = base.get(name, 0.25)
            r = regime_rec.get(name, 0.25)
            blended[name] = (1 - alpha) * b + alpha * r

        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}
        return blended

    def _compute_metrics(self, elapsed: float) -> Dict:
        """Compute comprehensive backtest performance metrics."""
        equity = np.array(self.equity_curve)
        returns = np.array(self.daily_returns)

        if len(equity) == 0:
            return {"error": "No equity data"}

        total_return = (equity[-1] - self.initial_capital) / self.initial_capital
        n_years = len(returns) / 252

        # CAGR (Compound Annual Growth Rate)
        if n_years > 0 and equity[-1] > 0:
            cagr = (equity[-1] / self.initial_capital) ** (1 / n_years) - 1
        else:
            cagr = 0.0

        # Sharpe ratio (annualized)
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0

        # Sortino ratio (only downside deviation)
        downside = returns[returns < 0]
        if len(downside) > 1:
            sortino = np.mean(returns) / np.std(downside) * np.sqrt(252)
        else:
            sortino = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdowns = (peak - equity) / np.where(peak > 0, peak, 1)
        max_dd = float(np.max(drawdowns))

        # Calmar ratio (CAGR / max_dd)
        calmar = cagr / max_dd if max_dd > 0 else 0.0

        # Win rate
        pnls = [t.get("pnl", 0) for t in self.trade_log if t["action"] == "SELL"]
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls) if pnls else 0

        winning_pnls = [p for p in pnls if p > 0]
        losing_pnls = [p for p in pnls if p < 0]
        avg_win = np.mean(winning_pnls) if winning_pnls else 0
        avg_loss = np.mean(losing_pnls) if losing_pnls else 0

        # SPY benchmark metrics
        spy_metrics = {}
        if self.spy_equity:
            spy_eq = np.array(self.spy_equity)
            spy_return = (spy_eq[-1] - self.initial_capital) / self.initial_capital
            spy_cagr = (spy_eq[-1] / self.initial_capital) ** (1 / max(n_years, 0.01)) - 1
            spy_daily = np.diff(spy_eq) / spy_eq[:-1]
            spy_sharpe = np.mean(spy_daily) / np.std(spy_daily) * np.sqrt(252) if np.std(spy_daily) > 0 else 0
            spy_peak = np.maximum.accumulate(spy_eq)
            spy_dd = np.max((spy_peak - spy_eq) / np.where(spy_peak > 0, spy_peak, 1))
            spy_metrics = {
                "total_return": spy_return,
                "cagr": spy_cagr,
                "sharpe": spy_sharpe,
                "max_drawdown": spy_dd,
                "final_value": float(spy_eq[-1]),
            }

        # Per-strategy attribution
        strategy_attribution = {}
        for name in self.strategies:
            strategy_attribution[name] = {
                "total_pnl": self.strategy_pnl.get(name, 0),
                "trades": self.strategy_trades.get(name, 0),
            }

        # Regime distribution
        from collections import Counter
        regime_counts = Counter(self.regime_log)
        regime_pcts = {k: v / len(self.regime_log) for k, v in regime_counts.items()}

        return {
            "period": f"{self.start_date} to {self.end_date}",
            "trading_days": len(returns),
            "years": n_years,
            "initial_capital": self.initial_capital,
            "final_value": float(equity[-1]),
            "total_return": total_return,
            "cagr": cagr,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "total_trades": len(self.trade_log),
            "win_rate": win_rate,
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else float("inf"),
            "elapsed_seconds": elapsed,
            "final_weights": dict(self.self_improver.weights),
            "spy_benchmark": spy_metrics,
            "strategy_attribution": strategy_attribution,
            "regime_distribution": regime_pcts,
            "kelly_stats": self.risk_manager.get_kelly_stats(),
        }

    def _print_report(self, results: Dict):
        """Print a comprehensive backtest comparison report."""
        self.logger.info("\n" + "=" * 70)
        self.logger.info("              BACKTEST RESULTS — v2 Redesigned Strategies")
        self.logger.info("=" * 70)
        self.logger.info(f"  Period:              {results['period']}")
        self.logger.info(f"  Trading Days:        {results['trading_days']}")
        self.logger.info(f"  Years:               {results['years']:.1f}")
        self.logger.info("")
        self.logger.info(f"  Initial Capital:     ${results['initial_capital']:>14,.2f}")
        self.logger.info(f"  Final Value:         ${results['final_value']:>14,.2f}")
        self.logger.info(f"  Total Return:        {results['total_return']:>13.2%}")
        self.logger.info(f"  CAGR:                {results['cagr']:>13.2%}")
        self.logger.info(f"  Sharpe Ratio:        {results['sharpe_ratio']:>13.2f}")
        self.logger.info(f"  Sortino Ratio:       {results['sortino_ratio']:>13.2f}")
        self.logger.info(f"  Max Drawdown:        {results['max_drawdown']:>13.2%}")
        self.logger.info(f"  Calmar Ratio:        {results['calmar_ratio']:>13.2f}")
        self.logger.info("")
        self.logger.info(f"  Total Trades:        {results['total_trades']:>13d}")
        self.logger.info(f"  Win Rate:            {results['win_rate']:>13.2%}")
        self.logger.info(f"  Avg Win:             ${results['avg_win']:>13.2f}")
        self.logger.info(f"  Avg Loss:            ${results['avg_loss']:>13.2f}")
        self.logger.info(f"  Profit Factor:       {results['profit_factor']:>13.2f}")

        # SPY comparison
        spy = results.get("spy_benchmark", {})
        if spy:
            self.logger.info("")
            self.logger.info("-" * 70)
            self.logger.info("  SPY BENCHMARK COMPARISON")
            self.logger.info("-" * 70)
            self.logger.info(f"                        {'Strategy':>14s}  {'SPY':>14s}")
            self.logger.info(f"  CAGR:                 {results['cagr']:>13.2%}  {spy['cagr']:>13.2%}")
            self.logger.info(f"  Sharpe:               {results['sharpe_ratio']:>13.2f}  {spy['sharpe']:>13.2f}")
            self.logger.info(f"  Max Drawdown:         {results['max_drawdown']:>13.2%}  {spy['max_drawdown']:>13.2%}")
            self.logger.info(f"  Final Value:          ${results['final_value']:>13,.2f}  ${spy['final_value']:>13,.2f}")
            alpha = results['cagr'] - spy['cagr']
            self.logger.info(f"  Alpha vs SPY:         {alpha:>13.2%}")

        # Strategy attribution
        attr = results.get("strategy_attribution", {})
        if attr:
            self.logger.info("")
            self.logger.info("-" * 70)
            self.logger.info("  STRATEGY ATTRIBUTION")
            self.logger.info("-" * 70)
            for name, stats in attr.items():
                self.logger.info(
                    f"    {name:25s} PnL=${stats['total_pnl']:>10,.2f}  "
                    f"Trades={stats['trades']:>5d}"
                )

        # Regime distribution
        regime_dist = results.get("regime_distribution", {})
        if regime_dist:
            self.logger.info("")
            self.logger.info("-" * 70)
            self.logger.info("  REGIME DISTRIBUTION")
            self.logger.info("-" * 70)
            for regime, pct in sorted(regime_dist.items()):
                self.logger.info(f"    {regime:25s} {pct:>8.1%}")

        # Kelly stats
        kelly = results.get("kelly_stats", {})
        if kelly and kelly.get("trades", 0) > 0:
            self.logger.info("")
            self.logger.info("-" * 70)
            self.logger.info("  KELLY CRITERION STATS")
            self.logger.info("-" * 70)
            self.logger.info(f"    Win Rate:           {kelly['win_rate']:>8.2%}")
            self.logger.info(f"    Payoff Ratio:       {kelly['payoff_ratio']:>8.2f}")
            self.logger.info(f"    Raw Kelly %:        {kelly['kelly_pct']:>8.2%}")
            self.logger.info(f"    Half-Kelly %:       {kelly['half_kelly_pct']:>8.2%}")

        # Final weights
        self.logger.info("")
        self.logger.info("-" * 70)
        self.logger.info("  FINAL STRATEGY WEIGHTS")
        self.logger.info("-" * 70)
        for name, weight in results["final_weights"].items():
            self.logger.info(f"    {name:25s} {weight:.3f}")

        self.logger.info("")
        self.logger.info(f"  Elapsed: {results['elapsed_seconds']:.1f}s")
        self.logger.info("=" * 70)

    def get_equity_curve(self) -> pd.DataFrame:
        """Return equity curve as DataFrame."""
        data = {"strategy_equity": self.equity_curve}
        if self.spy_equity:
            # Align SPY equity to same length
            spy_padded = self.spy_equity + [self.spy_equity[-1]] * (
                len(self.equity_curve) - len(self.spy_equity)
            ) if len(self.spy_equity) < len(self.equity_curve) else self.spy_equity[:len(self.equity_curve)]
            data["spy_equity"] = spy_padded
        return pd.DataFrame(data)
