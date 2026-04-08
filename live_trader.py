"""
Live Paper Trader — Connects to Alpaca Paper Trading API.

Reads strategy signals from the existing agent pipeline, executes trades
via the Alpaca paper trading API, logs all activity to CSV, and runs on
configurable intervals during market hours.

Respects the risk manager (Kelly + vol targeting, stop-loss/take-profit,
drawdown limits) and leverage manager settings from config.

Usage:
    Typically launched via start_paper_trading.py, but can run standalone:
        python live_trader.py
"""

import csv
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import alpaca_trade_api as tradeapi
except ImportError:
    tradeapi = None

from config import ALPACA_API_KEY, ALPACA_BASE_URL, ALPACA_SECRET_KEY, config
from data_fetcher import DataFetcher
from leverage_manager import LeverageConfig, LeverageManager
from logger import TradeLogger
from risk_manager import RiskManager
from self_improver import SelfImprover
from strategies.mean_reversion import MeanReversionStrategy, Signal
from strategies.momentum import MomentumStrategy
from strategies.pattern_recognition import MarketRegime, PatternRecognitionStrategy
from strategies.sentiment import SentimentStrategy


class LiveTrader:
    """
    Paper trading execution engine.

    Connects to Alpaca's paper trading API, runs the full signal pipeline
    at regular intervals during market hours, and logs every trade to CSV.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.logger = TradeLogger()
        self.data_fetcher = DataFetcher()

        # Core modules
        self.risk_manager = RiskManager(config.initial_capital, self.logger)
        self.self_improver = SelfImprover(self.logger)

        # Leverage manager
        lev_cfg = config.leverage
        self.leverage_manager = LeverageManager(LeverageConfig(
            mode=lev_cfg.get("mode", "none"),
            fixed_multiplier=lev_cfg.get("fixed_multiplier", 3.0),
            max_leverage=lev_cfg.get("max_leverage", 5.0),
            vol_target_annual=lev_cfg.get("vol_target_annual", 0.15),
            max_drawdown_trigger=lev_cfg.get("max_drawdown_trigger", 0.10),
            ramp_days=lev_cfg.get("ramp_days", 5),
            funding_cost_annual=lev_cfg.get("funding_cost_annual", 0.02),
            min_leverage=lev_cfg.get("min_leverage", 0.5),
        ))
        self._daily_returns: List[float] = []

        # Strategies
        self.strategies = {
            "mean_reversion": MeanReversionStrategy(),
            "momentum": MomentumStrategy(),
            "sentiment": SentimentStrategy(),
            "pattern_recognition": PatternRecognitionStrategy(),
        }

        # Alpaca API client
        self.api = None
        self.alpaca_connected = False
        if not dry_run:
            self._connect_alpaca()

        # Paper trading config
        pt = getattr(config, "paper_trading", {})
        self.symbols = pt.get("symbols", config.symbols)
        self.interval_minutes = pt.get("interval_minutes", 15)
        self.market_open_hour = pt.get("market_open_hour", 9)
        self.market_open_minute = pt.get("market_open_minute", 30)
        self.market_close_hour = pt.get("market_close_hour", 16)
        self.market_close_minute = pt.get("market_close_minute", 0)

        # Trade log CSV
        self.trade_log_path = os.path.join(config.log_dir, "paper_trades.csv")
        self._init_trade_log()

        # State
        self.cycle_count = 0
        self.current_regime = MarketRegime.MEAN_REVERTING
        self.running = False
        self.total_pnl = 0.0

        mode = "DRY RUN" if dry_run else "LIVE PAPER"
        self.logger.info(f"LiveTrader initialized [{mode}]")
        self.logger.info(f"  Symbols: {self.symbols}")
        self.logger.info(f"  Interval: {self.interval_minutes} min")
        self.logger.info(f"  Market hours: {self.market_open_hour}:{self.market_open_minute:02d}"
                         f" - {self.market_close_hour}:{self.market_close_minute:02d} ET")
        self.logger.info(f"  Alpaca connected: {self.alpaca_connected}")

    # ── Alpaca Connection ────────────────────────────────────────────────

    def _connect_alpaca(self):
        """Establish connection to Alpaca paper trading API."""
        if tradeapi is None:
            self.logger.warning("alpaca-trade-api not installed. Install with: "
                                "pip install alpaca-trade-api")
            return

        if ALPACA_API_KEY == "YOUR_ALPACA_API_KEY":
            self.logger.warning("Alpaca API keys not configured. "
                                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars.")
            return

        try:
            self.api = tradeapi.REST(
                key_id=ALPACA_API_KEY,
                secret_key=ALPACA_SECRET_KEY,
                base_url=ALPACA_BASE_URL,
                api_version="v2",
            )
            account = self.api.get_account()
            self.alpaca_connected = True
            self.risk_manager.current_capital = float(account.cash)
            self.logger.info(f"Alpaca connected | Account: {account.account_number} | "
                             f"Cash: ${float(account.cash):,.2f} | "
                             f"Portfolio: ${float(account.portfolio_value):,.2f}")
        except Exception as e:
            self.logger.error(f"Alpaca connection failed: {e}")
            self.alpaca_connected = False

    def get_account_info(self) -> Optional[Dict]:
        """Fetch current account information from Alpaca."""
        if not self.alpaca_connected or not self.api:
            return None
        try:
            account = self.api.get_account()
            return {
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "buying_power": float(account.buying_power),
                "equity": float(account.equity),
                "long_market_value": float(account.long_market_value),
                "short_market_value": float(account.short_market_value),
                "status": account.status,
                "pattern_day_trader": account.pattern_day_trader,
                "daytrade_count": account.daytrade_count,
            }
        except Exception as e:
            self.logger.error(f"Failed to fetch account info: {e}")
            return None

    # ── Market Hours ─────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        """Check if we're within configured market hours (Eastern Time)."""
        now = datetime.utcnow() - timedelta(hours=4)  # Approximate ET
        if now.weekday() >= 5:  # Saturday/Sunday
            return False
        market_open = now.replace(
            hour=self.market_open_hour, minute=self.market_open_minute, second=0
        )
        market_close = now.replace(
            hour=self.market_close_hour, minute=self.market_close_minute, second=0
        )
        return market_open <= now <= market_close

    def time_until_market_open(self) -> float:
        """Return seconds until market opens. Returns 0 if market is open."""
        if self.is_market_open():
            return 0.0
        now = datetime.utcnow() - timedelta(hours=4)
        next_open = now.replace(
            hour=self.market_open_hour, minute=self.market_open_minute, second=0
        )
        if now >= next_open:
            # Market closed for today, calculate to next business day
            days_ahead = 1
            if now.weekday() == 4:  # Friday
                days_ahead = 3
            elif now.weekday() == 5:  # Saturday
                days_ahead = 2
            next_open += timedelta(days=days_ahead)
        return (next_open - now).total_seconds()

    # ── Trade Logging ────────────────────────────────────────────────────

    def _init_trade_log(self):
        """Initialize the paper trading CSV log."""
        os.makedirs(config.log_dir, exist_ok=True)
        if not os.path.exists(self.trade_log_path):
            with open(self.trade_log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "cycle", "symbol", "action", "quantity",
                    "price", "filled_price", "strategy", "signal_strength",
                    "regime", "leverage", "portfolio_value", "pnl", "reason",
                ])

    def _log_trade(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        filled_price: float,
        strategy: str,
        signal_strength: float,
        regime: str,
        leverage: float,
        portfolio_value: float,
        pnl: float,
        reason: str,
    ):
        """Append a trade record to the paper trading CSV."""
        with open(self.trade_log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().isoformat(),
                self.cycle_count,
                symbol,
                action,
                quantity,
                f"{price:.4f}",
                f"{filled_price:.4f}",
                strategy,
                f"{signal_strength:.4f}",
                regime,
                f"{leverage:.2f}",
                f"{portfolio_value:.2f}",
                f"{pnl:.2f}",
                reason,
            ])

    # ── Signal Generation ────────────────────────────────────────────────

    def _detect_regime(self) -> MarketRegime:
        """Detect market regime using SPY as proxy."""
        try:
            df_spy = self.data_fetcher.get_historical("SPY", period="2y")
            if df_spy.empty or len(df_spy) < 100:
                return MarketRegime.MEAN_REVERTING
            df_spy = DataFetcher.compute_indicators(df_spy)
            regime = self.strategies["pattern_recognition"].detect_regime(df_spy)
            self.current_regime = regime
            return regime
        except Exception:
            return self.current_regime

    def _get_active_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """Get regime-adjusted strategy weights."""
        base_weights = self.self_improver.weights
        if not config.use_regime_weighting:
            return base_weights

        regime_weights = self.strategies[
            "pattern_recognition"
        ].get_regime_weights(regime)

        alpha = config.regime_blend_alpha
        blended = {}
        for name in base_weights:
            base = base_weights.get(name, 0.25)
            regime_rec = regime_weights.get(name, 0.25)
            blended[name] = (1 - alpha) * base + alpha * regime_rec

        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}
        return blended

    def _generate_signals(
        self, symbol: str, weights: Dict[str, float]
    ) -> Tuple[Optional[Signal], Dict]:
        """Run all strategies on a symbol, return combined signal."""
        df = self.data_fetcher.get_historical(symbol, period="2y")
        if df.empty or len(df) < 50:
            return None, {}

        df = DataFetcher.compute_indicators(df)
        current_price = float(df["Close"].iloc[-1])
        atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else 0.0
        volatility = float(df["returns"].std()) if "returns" in df.columns else 0.02

        meta = {"price": current_price, "atr": atr, "volatility": volatility}

        signals: Dict[str, Signal] = {}
        signals["mean_reversion"] = self.strategies["mean_reversion"].generate_signal(symbol, df)
        signals["momentum"] = self.strategies["momentum"].generate_signal(symbol, df)
        signals["sentiment"] = self.strategies["sentiment"].generate_signal(symbol, df)
        signals["pattern_recognition"] = self.strategies["pattern_recognition"].generate_signal(symbol, df)

        combined_strength = sum(
            weights.get(name, 0) * sig.strength
            for name, sig in signals.items()
        )
        combined_strength = np.clip(combined_strength, -1.0, 1.0)
        dominant = max(signals.items(), key=lambda x: abs(x[1].strength))

        if combined_strength > 0:
            action = "BUY"
        elif combined_strength < 0:
            action = "SELL"
        else:
            action = "HOLD"

        reasons = [f"{n}={s.strength:.2f}" for n, s in signals.items()]
        combined_signal = Signal(
            symbol=symbol,
            action=action,
            strength=combined_strength,
            strategy=f"combined({dominant[0]})",
            reason=f"Regime={self.current_regime.value}, {', '.join(reasons)}",
        )
        return combined_signal, meta

    # ── Order Execution ──────────────────────────────────────────────────

    def _submit_order(
        self, symbol: str, qty: int, side: str
    ) -> Tuple[bool, float]:
        """
        Submit an order via Alpaca API (or simulate in dry-run mode).

        Returns (success, filled_price).
        """
        if self.dry_run:
            price = self._get_latest_price(symbol)
            self.logger.info(f"[DRY RUN] {side.upper()} {qty} {symbol} @ ~${price:.2f}")
            return True, price

        if not self.alpaca_connected or not self.api:
            self.logger.warning(f"Alpaca not connected — cannot submit {side} {qty} {symbol}")
            return False, 0.0

        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=str(qty),
                side=side,
                type="market",
                time_in_force="day",
            )
            self.logger.info(f"Order submitted: {side} {qty} {symbol} | Order ID: {order.id}")

            # Wait briefly for fill
            filled_price = 0.0
            for _ in range(10):
                time.sleep(1)
                updated = self.api.get_order(order.id)
                if updated.status == "filled":
                    filled_price = float(updated.filled_avg_price or 0)
                    break
                elif updated.status in ("canceled", "expired", "rejected"):
                    self.logger.warning(f"Order {updated.status}: {side} {qty} {symbol}")
                    return False, 0.0

            if filled_price == 0:
                filled_price = self._get_latest_price(symbol)
                self.logger.warning(f"Order not yet filled, using market price: ${filled_price:.2f}")

            return True, filled_price

        except Exception as e:
            self.logger.error(f"Order submission failed: {e}")
            return False, 0.0

    def _get_latest_price(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        prices = self.data_fetcher.get_latest_prices([symbol])
        return prices.get(symbol, 0.0)

    # ── Trading Cycle ────────────────────────────────────────────────────

    def run_cycle(self) -> Dict:
        """Execute one full trading cycle."""
        self.cycle_count += 1
        cycle_start = time.time()
        self.logger.info(f"\n{'='*50} LIVE CYCLE {self.cycle_count} {'='*50}")

        actions = {"buys": [], "sells": [], "holds": [], "errors": []}

        # Drawdown check
        if self.risk_manager.check_drawdown():
            self.logger.error("Trading halted — max drawdown breached")
            return actions

        # Check stop-loss / take-profit on existing positions
        open_symbols = list(self.risk_manager.positions.keys())
        if open_symbols:
            prices = self.data_fetcher.get_latest_prices(open_symbols)
            triggers = self.risk_manager.check_stop_loss_take_profit(prices)
            for symbol in triggers:
                price = prices.get(symbol)
                if price:
                    self._execute_sell(symbol, price, "stop_loss_or_take_profit", actions)

        # Detect regime
        regime = self._detect_regime()
        active_weights = self._get_active_weights(regime)
        self.logger.info(f"Regime: {regime.value} | Weights: "
                         + ", ".join(f"{k}={v:.2f}" for k, v in active_weights.items()))

        # Analyze each symbol
        for symbol in self.symbols:
            try:
                signal, meta = self._generate_signals(symbol, active_weights)
                if signal is None:
                    continue

                can_trade, reason = self.risk_manager.can_trade(symbol)
                if not can_trade:
                    self.logger.info(f"  {symbol}: SKIP ({reason})")
                    actions["holds"].append(symbol)
                    continue

                regime_str = regime.value if regime else "normal"

                if signal.action == "BUY" and signal.strength > 0.25:
                    self._execute_buy(symbol, signal, meta, regime_str, actions)
                elif signal.action == "SELL" and signal.strength < -0.25:
                    if symbol in self.risk_manager.positions:
                        self._execute_sell(
                            symbol, meta["price"], signal.reason, actions
                        )
                else:
                    actions["holds"].append(symbol)

            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
                actions["errors"].append({"symbol": symbol, "error": str(e)})

        # Periodic weight update
        if self.cycle_count % 5 == 0:
            self.self_improver.update_weights(regime_name=regime.value)

        # Performance snapshot
        risk_status = self.risk_manager.get_status()
        self.logger.log_performance_snapshot({
            "cycle": self.cycle_count,
            "regime": regime.value,
            "total_value": risk_status["total_value"],
            "capital": risk_status["capital"],
            "open_positions": risk_status["open_positions"],
            "exposure_pct": f"{risk_status['exposure_pct']:.2%}",
            "drawdown_pct": f"{risk_status['drawdown_pct']:.2%}",
            "total_pnl": self.total_pnl,
            "buys": len(actions["buys"]),
            "sells": len(actions["sells"]),
        })

        elapsed = time.time() - cycle_start
        self.logger.info(f"Cycle {self.cycle_count} completed in {elapsed:.1f}s | "
                         f"PnL: ${self.total_pnl:+,.2f}")
        return actions

    def _execute_buy(
        self, symbol: str, signal: Signal, meta: Dict, regime: str,
        actions: Dict,
    ):
        """Execute a buy with Kelly + vol + leverage sizing."""
        price = meta["price"]
        atr = meta["atr"]
        volatility = meta["volatility"]

        quantity = self.risk_manager.calculate_position_size(
            symbol, price, signal.strength, volatility, regime
        )
        if quantity <= 0:
            return

        # Apply leverage
        equity = self.risk_manager._portfolio_value_estimate()
        leverage = self.leverage_manager.get_leverage(
            current_equity=equity, daily_returns=self._daily_returns
        )
        quantity = int(quantity * leverage)
        if quantity <= 0:
            return

        success, filled_price = self._submit_order(symbol, quantity, "buy")
        if not success:
            return

        self.risk_manager.open_position(symbol, quantity, filled_price, signal.strategy, atr)

        self._log_trade(
            symbol=symbol, action="BUY", quantity=quantity,
            price=price, filled_price=filled_price,
            strategy=signal.strategy, signal_strength=signal.strength,
            regime=regime, leverage=leverage,
            portfolio_value=self.risk_manager._portfolio_value_estimate(),
            pnl=0.0, reason=signal.reason,
        )
        self.logger.log_trade(
            symbol=symbol, action="BUY", quantity=quantity,
            price=filled_price, strategy=signal.strategy,
            signal_strength=signal.strength,
            portfolio_value=self.risk_manager._portfolio_value_estimate(),
            reason=signal.reason,
        )
        actions["buys"].append(symbol)

    def _execute_sell(
        self, symbol: str, price: float, reason: str, actions: Dict,
    ):
        """Execute a sell (close position)."""
        if symbol not in self.risk_manager.positions:
            return

        pos = self.risk_manager.positions[symbol]
        quantity = pos.quantity
        strategy = pos.strategy
        entry_price = pos.entry_price

        success, filled_price = self._submit_order(symbol, quantity, "sell")
        if not success:
            return

        pnl = self.risk_manager.close_position(symbol, filled_price)
        if pnl is not None:
            self.total_pnl += pnl

        holding_hours = (datetime.now() - pos.entry_time).total_seconds() / 3600
        self.self_improver.record_experience(
            symbol=symbol, strategy=strategy, action="SELL",
            signal_strength=0.0, entry_price=entry_price,
            exit_price=filled_price,
            holding_period_hours=holding_hours,
            market_regime=self.current_regime.value,
        )

        self._log_trade(
            symbol=symbol, action="SELL", quantity=quantity,
            price=price, filled_price=filled_price,
            strategy=strategy, signal_strength=0.0,
            regime=self.current_regime.value, leverage=1.0,
            portfolio_value=self.risk_manager._portfolio_value_estimate(),
            pnl=pnl or 0.0, reason=reason,
        )
        self.logger.log_trade(
            symbol=symbol, action="SELL", quantity=quantity,
            price=filled_price, strategy=strategy,
            signal_strength=0.0,
            portfolio_value=self.risk_manager._portfolio_value_estimate(),
            reason=reason,
        )
        actions["sells"].append(symbol)

    # ── Main Loop ────────────────────────────────────────────────────────

    def run(self):
        """
        Main trading loop. Runs cycles at configured intervals during
        market hours. Sleeps outside market hours.

        Ctrl+C triggers graceful shutdown.
        """
        self.running = True
        self.logger.info("=" * 60)
        self.logger.info("LIVE PAPER TRADING STARTED")
        self.logger.info("=" * 60)

        def _shutdown(signum, frame):
            self.logger.info("\nShutdown signal received. Closing gracefully...")
            self.running = False

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        while self.running:
            try:
                if self.is_market_open():
                    self.run_cycle()
                    if self.running:
                        self.logger.info(
                            f"Next cycle in {self.interval_minutes} minutes..."
                        )
                        self._interruptible_sleep(self.interval_minutes * 60)
                else:
                    wait = self.time_until_market_open()
                    hours = wait / 3600
                    self.logger.info(
                        f"Market closed. Next open in {hours:.1f} hours. Sleeping..."
                    )
                    self._interruptible_sleep(min(wait, 300))  # Check every 5 min max

            except Exception as e:
                self.logger.error(f"Unexpected error in main loop: {e}")
                if self.running:
                    self._interruptible_sleep(60)

        self._shutdown()

    def _interruptible_sleep(self, seconds: float):
        """Sleep that can be interrupted by setting self.running = False."""
        end_time = time.time() + seconds
        while self.running and time.time() < end_time:
            time.sleep(min(1, end_time - time.time()))

    def _shutdown(self):
        """Graceful shutdown: log final state."""
        self.logger.info("=" * 60)
        self.logger.info("SHUTTING DOWN PAPER TRADER")
        self.logger.info(f"  Total cycles: {self.cycle_count}")
        self.logger.info(f"  Total PnL: ${self.total_pnl:+,.2f}")
        self.logger.info(f"  Open positions: {len(self.risk_manager.positions)}")

        if self.risk_manager.positions:
            self.logger.info("  Open positions at shutdown:")
            for sym, pos in self.risk_manager.positions.items():
                self.logger.info(
                    f"    {sym}: {pos.quantity} shares @ ${pos.entry_price:.2f}"
                )

        self.self_improver._save_state()
        self.logger.info("State saved. Goodbye.")
        self.logger.info("=" * 60)

    def get_status(self) -> Dict:
        """Return current trader status."""
        risk = self.risk_manager.get_status()
        account = self.get_account_info()
        return {
            "cycle_count": self.cycle_count,
            "total_pnl": self.total_pnl,
            "current_regime": self.current_regime.value,
            "market_open": self.is_market_open(),
            "alpaca_connected": self.alpaca_connected,
            "dry_run": self.dry_run,
            "risk": risk,
            "strategy_weights": dict(self.self_improver.weights),
            "account": account,
        }


# ── Standalone entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live Paper Trader")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate trades without hitting Alpaca API")
    args = parser.parse_args()

    trader = LiveTrader(dry_run=args.dry_run)
    trader.run()
