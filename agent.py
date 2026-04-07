"""
Main AI Trading Agent — the brain that combines all strategies.

This is the central decision engine. It:
  1. Fetches latest market data for the trading universe
  2. Runs all strategies to generate signals
  3. Combines signals using learned weights (from self-improver)
  4. Applies risk management checks
  5. Executes trades via Alpaca paper trading API
  6. Records outcomes for continuous self-improvement
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

from config import (
    ALPACA_API_KEY,
    ALPACA_BASE_URL,
    ALPACA_SECRET_KEY,
    config,
)
from data_fetcher import DataFetcher
from logger import TradeLogger
from risk_manager import RiskManager
from self_improver import SelfImprover
from strategies.mean_reversion import MeanReversionStrategy, Signal
from strategies.momentum import MomentumStrategy
from strategies.pattern_recognition import PatternRecognitionStrategy
from strategies.sentiment import SentimentStrategy


class TradingAgent:
    """
    AI Trading Agent that orchestrates strategies, risk, and execution.

    Pipeline per cycle:
      data → indicators → strategies → aggregate → risk check → execute → log
    """

    def __init__(
        self,
        capital: float = config.initial_capital,
        paper_trade: bool = True,
    ):
        # Core components
        self.logger = TradeLogger()
        self.data_fetcher = DataFetcher()
        self.risk_manager = RiskManager(capital, self.logger)
        self.self_improver = SelfImprover(self.logger)

        # Strategies
        self.strategies = {
            "mean_reversion": MeanReversionStrategy(),
            "momentum": MomentumStrategy(),
            "sentiment": SentimentStrategy(),
            "pattern_recognition": PatternRecognitionStrategy(),
        }

        # Execution mode
        self.paper_trade = paper_trade
        self.alpaca_available = self._check_alpaca()

        # Tracking
        self.cycle_count = 0
        self.total_pnl = 0.0

        self.logger.info("=" * 60)
        self.logger.info("AI Trading Agent initialized")
        self.logger.info(f"  Capital: ${capital:,.2f}")
        self.logger.info(f"  Symbols: {config.symbols}")
        self.logger.info(f"  Paper trade: {paper_trade}")
        self.logger.info(f"  Alpaca connected: {self.alpaca_available}")
        self.logger.info(f"  Strategy weights: {self.self_improver.weights}")
        self.logger.info("=" * 60)

    # ── Main Trading Loop ────────────────────────────────────────────
    def run_cycle(self) -> Dict:
        """
        Execute one full trading cycle.
        Returns a summary dict of actions taken.
        """
        self.cycle_count += 1
        cycle_start = time.time()
        self.logger.info(f"\n{'='*40} CYCLE {self.cycle_count} {'='*40}")

        actions_taken = {"buys": [], "sells": [], "holds": []}

        # Step 0: Check drawdown
        if self.risk_manager.check_drawdown():
            self.logger.error("Trading halted — max drawdown breached")
            return actions_taken

        # Step 1: Check stop-loss / take-profit on existing positions
        prices = self.data_fetcher.get_latest_prices(
            list(self.risk_manager.positions.keys())
        )
        sl_tp_triggers = self.risk_manager.check_stop_loss_take_profit(prices)
        for symbol in sl_tp_triggers:
            price = prices.get(symbol)
            if price:
                self._execute_sell(symbol, price, "stop_loss_or_take_profit")
                actions_taken["sells"].append(symbol)

        # Step 2: Fetch data and generate signals for universe
        for symbol in config.symbols:
            try:
                signal, meta = self._analyze_symbol(symbol)
                if signal is None:
                    continue

                can_trade, reason = self.risk_manager.can_trade(symbol)
                if not can_trade:
                    self.logger.debug(f"Cannot trade {symbol}: {reason}")
                    actions_taken["holds"].append(symbol)
                    continue

                # Step 3: Execute based on signal
                if signal.action == "BUY" and signal.strength > 0.3:
                    self._execute_buy(symbol, signal, meta)
                    actions_taken["buys"].append(symbol)
                elif signal.action == "SELL" and signal.strength < -0.3:
                    if symbol in self.risk_manager.positions:
                        price = meta.get("price", 0)
                        self._execute_sell(symbol, price, signal.reason)
                        actions_taken["sells"].append(symbol)
                else:
                    actions_taken["holds"].append(symbol)

            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")

        # Step 4: Update self-improver weights periodically
        if self.cycle_count % 5 == 0:
            self.self_improver.update_weights()

        # Step 5: Log performance snapshot
        risk_status = self.risk_manager.get_status()
        self.logger.log_performance_snapshot({
            "cycle": self.cycle_count,
            "total_value": risk_status["total_value"],
            "capital": risk_status["capital"],
            "open_positions": risk_status["open_positions"],
            "exposure_pct": f"{risk_status['exposure_pct']:.2%}",
            "drawdown_pct": f"{risk_status['drawdown_pct']:.2%}",
            "buys": len(actions_taken["buys"]),
            "sells": len(actions_taken["sells"]),
        })

        elapsed = time.time() - cycle_start
        self.logger.info(f"Cycle {self.cycle_count} completed in {elapsed:.1f}s")

        return actions_taken

    # ── Symbol Analysis ──────────────────────────────────────────────
    def _analyze_symbol(
        self, symbol: str
    ) -> Tuple[Optional[Signal], Dict]:
        """Run all strategies on a symbol and return combined signal."""
        # Fetch historical data
        df = self.data_fetcher.get_historical(symbol, period="6mo")
        if df.empty:
            return None, {}

        df = DataFetcher.compute_indicators(df)
        current_price = float(df["Close"].iloc[-1])
        atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else 0.0
        volatility = float(df["returns"].std()) if "returns" in df.columns else 0.02

        meta = {"price": current_price, "atr": atr, "volatility": volatility}

        # Collect signals from each strategy
        signals: Dict[str, Signal] = {}

        # Mean reversion
        signals["mean_reversion"] = self.strategies[
            "mean_reversion"
        ].generate_signal(symbol, df)

        # Momentum
        signals["momentum"] = self.strategies["momentum"].generate_signal(
            symbol, df
        )

        # Sentiment (fetch news if API key is set)
        articles = self.data_fetcher.get_news(symbol, days_back=3)
        signals["sentiment"] = self.strategies["sentiment"].generate_signal(
            symbol, articles
        )

        # Pattern recognition
        signals["pattern_recognition"] = self.strategies[
            "pattern_recognition"
        ].generate_signal(symbol, df)

        # ── Weighted Combination ─────────────────────────────────────
        weights = self.self_improver.weights
        combined_strength = sum(
            weights.get(name, 0) * sig.strength
            for name, sig in signals.items()
        )
        combined_strength = np.clip(combined_strength, -1.0, 1.0)

        # Determine dominant strategy
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
            reason=f"Weighted: {', '.join(reasons)}",
        )

        self.logger.debug(
            f"{symbol}: {combined_signal.action} "
            f"str={combined_signal.strength:.3f} | {combined_signal.reason}"
        )

        return combined_signal, meta

    # ── Execution ────────────────────────────────────────────────────
    def _execute_buy(self, symbol: str, signal: Signal, meta: Dict):
        """Execute a buy order."""
        price = meta["price"]
        atr = meta["atr"]
        volatility = meta["volatility"]

        quantity = self.risk_manager.calculate_position_size(
            symbol, price, signal.strength, volatility
        )
        if quantity <= 0:
            return

        # Execute via Alpaca (paper) or simulate
        executed_price = price
        if self.alpaca_available and self.paper_trade:
            executed_price = self._alpaca_order(symbol, quantity, "buy") or price

        # Record position
        self.risk_manager.open_position(
            symbol, quantity, executed_price, signal.strategy, atr
        )

        self.logger.log_trade(
            symbol=symbol,
            action="BUY",
            quantity=quantity,
            price=executed_price,
            strategy=signal.strategy,
            signal_strength=signal.strength,
            portfolio_value=self.risk_manager._portfolio_value_estimate(),
            reason=signal.reason,
        )

    def _execute_sell(self, symbol: str, price: float, reason: str):
        """Execute a sell order (close position)."""
        if symbol not in self.risk_manager.positions:
            return

        pos = self.risk_manager.positions[symbol]
        quantity = pos.quantity
        strategy = pos.strategy
        entry_price = pos.entry_price

        # Execute via Alpaca or simulate
        executed_price = price
        if self.alpaca_available and self.paper_trade:
            executed_price = (
                self._alpaca_order(symbol, quantity, "sell") or price
            )

        pnl = self.risk_manager.close_position(symbol, executed_price)
        if pnl is not None:
            self.total_pnl += pnl

        # Record experience for self-improvement
        holding_hours = (
            (datetime.now() - pos.entry_time).total_seconds() / 3600
        )
        self.self_improver.record_experience(
            symbol=symbol,
            strategy=strategy,
            action="SELL",
            signal_strength=0.0,
            entry_price=entry_price,
            exit_price=executed_price,
            holding_period_hours=holding_hours,
        )

        self.logger.log_trade(
            symbol=symbol,
            action="SELL",
            quantity=quantity,
            price=executed_price,
            strategy=strategy,
            signal_strength=0.0,
            portfolio_value=self.risk_manager._portfolio_value_estimate(),
            reason=reason,
        )

    # ── Alpaca API ───────────────────────────────────────────────────
    def _check_alpaca(self) -> bool:
        """Verify Alpaca API connection."""
        if ALPACA_API_KEY == "YOUR_ALPACA_API_KEY":
            return False
        try:
            resp = requests.get(
                f"{ALPACA_BASE_URL}/v2/account",
                headers={
                    "APCA-API-KEY-ID": ALPACA_API_KEY,
                    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
                },
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def _alpaca_order(
        self, symbol: str, qty: int, side: str
    ) -> Optional[float]:
        """Submit a market order to Alpaca. Returns fill price or None."""
        try:
            resp = requests.post(
                f"{ALPACA_BASE_URL}/v2/orders",
                headers={
                    "APCA-API-KEY-ID": ALPACA_API_KEY,
                    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
                },
                json={
                    "symbol": symbol,
                    "qty": str(qty),
                    "side": side,
                    "type": "market",
                    "time_in_force": "day",
                },
                timeout=10,
            )
            if resp.status_code in (200, 201):
                order = resp.json()
                self.logger.info(
                    f"Alpaca order submitted: {side} {qty} {symbol} "
                    f"(order_id={order.get('id', 'N/A')})"
                )
                return float(order.get("filled_avg_price", 0)) or None
            else:
                self.logger.warning(
                    f"Alpaca order failed: {resp.status_code} {resp.text}"
                )
        except Exception as e:
            self.logger.error(f"Alpaca order error: {e}")
        return None

    # ── Status ───────────────────────────────────────────────────────
    def get_status(self) -> Dict:
        """Return comprehensive agent status."""
        risk = self.risk_manager.get_status()
        improver = self.self_improver.get_report()
        trades = self.logger.get_trade_summary()
        return {
            "cycle_count": self.cycle_count,
            "total_pnl": self.total_pnl,
            "risk": risk,
            "strategy_weights": improver["current_weights"],
            "strategy_stats": improver["strategy_stats"],
            "trade_summary": trades,
        }
