"""
Main AI Trading Agent — Redesigned with Regime-Based Dynamic Weighting.

Key improvements over v1:
  1. Regime detection drives strategy selection (momentum in trends, MR in ranges)
  2. Factor Momentum replaces keyword-based sentiment
  3. Kelly criterion + vol targeting for position sizing
  4. Wider stop-losses prevent premature exits
  5. Multi-timeframe momentum (1m, 3m, 6m, 12m lookback windows)
  6. Dynamic strategy weights shift based on detected market regime

Pipeline per cycle:
  data -> indicators -> regime detection -> strategy signals ->
  regime-weighted aggregation -> Kelly+vol position sizing ->
  risk check -> execute -> log -> self-improve
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
from leverage_manager import LeverageConfig, LeverageManager
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


class TradingAgent:
    """
    AI Trading Agent with regime-based dynamic strategy weighting.

    The agent detects the current market regime (trending, mean-reverting,
    or crisis) and adjusts strategy weights accordingly. In trending markets,
    momentum gets higher weight. In range-bound markets, mean reversion
    dominates. In crisis, the agent goes defensive.
    """

    def __init__(
        self,
        capital: float = config.initial_capital,
        paper_trade: bool = True,
    ):
        self.logger = TradeLogger()
        self.data_fetcher = DataFetcher()
        self.risk_manager = RiskManager(capital, self.logger)
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
        self._daily_returns: list = []

        # Strategies (redesigned)
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
        self.current_regime = MarketRegime.MEAN_REVERTING

        self.logger.info("=" * 60)
        self.logger.info("AI Trading Agent v2 initialized")
        self.logger.info(f"  Capital: ${capital:,.2f}")
        self.logger.info(f"  Symbols: {config.symbols}")
        self.logger.info(f"  Paper trade: {paper_trade}")
        self.logger.info(f"  Alpaca connected: {self.alpaca_available}")
        self.logger.info(f"  Strategy weights: {self.self_improver.weights}")
        self.logger.info(f"  Kelly enabled: {config.use_kelly}")
        self.logger.info(f"  Vol target: {config.vol_target:.0%}")
        self.logger.info(f"  Regime weighting: {config.use_regime_weighting}")
        self.logger.info(f"  Leverage mode: {lev_cfg.get('mode', 'none')}")
        self.logger.info(f"  Max leverage: {lev_cfg.get('max_leverage', 5.0)}x")
        self.logger.info("=" * 60)

    # -- Main Trading Loop --

    def run_cycle(self) -> Dict:
        """Execute one full trading cycle with regime-aware weighting."""
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

        # Step 2: Detect regime using SPY (market proxy)
        regime = self._detect_market_regime()

        # Step 3: Get regime-adjusted strategy weights
        active_weights = self._get_active_weights(regime)

        # Step 4: Analyze each symbol
        for symbol in config.symbols:
            try:
                signal, meta = self._analyze_symbol(symbol, active_weights)
                if signal is None:
                    continue

                can_trade, reason = self.risk_manager.can_trade(symbol)
                if not can_trade:
                    actions_taken["holds"].append(symbol)
                    continue

                # Step 5: Execute based on signal
                regime_str = regime.value if regime else "normal"
                if signal.action == "BUY" and signal.strength > 0.25:
                    self._execute_buy(symbol, signal, meta, regime_str)
                    actions_taken["buys"].append(symbol)
                elif signal.action == "SELL" and signal.strength < -0.25:
                    if symbol in self.risk_manager.positions:
                        price = meta.get("price", 0)
                        self._execute_sell(symbol, price, signal.reason)
                        actions_taken["sells"].append(symbol)
                else:
                    actions_taken["holds"].append(symbol)

            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")

        # Step 6: Update self-improver weights periodically
        if self.cycle_count % 5 == 0:
            self.self_improver.update_weights(regime_name=regime.value)

        # Step 7: Log performance snapshot
        risk_status = self.risk_manager.get_status()
        self.logger.log_performance_snapshot({
            "cycle": self.cycle_count,
            "regime": regime.value,
            "total_value": risk_status["total_value"],
            "capital": risk_status["capital"],
            "open_positions": risk_status["open_positions"],
            "exposure_pct": f"{risk_status['exposure_pct']:.2%}",
            "drawdown_pct": f"{risk_status['drawdown_pct']:.2%}",
            "buys": len(actions_taken["buys"]),
            "sells": len(actions_taken["sells"]),
            "kelly": risk_status.get("kelly", {}),
        })

        elapsed = time.time() - cycle_start
        self.logger.info(f"Cycle {self.cycle_count} completed in {elapsed:.1f}s")
        return actions_taken

    # -- Regime Detection --

    def _detect_market_regime(self) -> MarketRegime:
        """
        Detect overall market regime using SPY as proxy.

        The regime detector (pattern_recognition strategy) analyzes
        SPY's return distribution to classify the current environment.
        """
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
        """
        Get strategy weights adjusted for current regime.

        Blends the self-improver's learned weights with the regime
        detector's recommended weights.
        """
        base_weights = self.self_improver.weights

        if not config.use_regime_weighting:
            return base_weights

        regime_weights = self.strategies[
            "pattern_recognition"
        ].get_regime_weights(regime)

        # Blend: (1-alpha) * learned_weights + alpha * regime_weights
        alpha = config.regime_blend_alpha
        blended = {}
        for name in base_weights:
            base = base_weights.get(name, 0.25)
            regime_rec = regime_weights.get(name, 0.25)
            blended[name] = (1 - alpha) * base + alpha * regime_rec

        # Normalize
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}

        return blended

    # -- Symbol Analysis --

    def _analyze_symbol(
        self, symbol: str, weights: Dict[str, float]
    ) -> Tuple[Optional[Signal], Dict]:
        """Run all strategies on a symbol with regime-aware weighting."""
        df = self.data_fetcher.get_historical(symbol, period="2y")
        if df.empty:
            return None, {}

        df = DataFetcher.compute_indicators(df)
        current_price = float(df["Close"].iloc[-1])
        atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else 0.0
        volatility = float(df["returns"].std()) if "returns" in df.columns else 0.02

        meta = {"price": current_price, "atr": atr, "volatility": volatility}

        # Collect signals from each strategy
        signals: Dict[str, Signal] = {}

        signals["mean_reversion"] = self.strategies[
            "mean_reversion"
        ].generate_signal(symbol, df)

        signals["momentum"] = self.strategies[
            "momentum"
        ].generate_signal(symbol, df)

        # Factor Momentum (pass DataFrame, not articles)
        signals["sentiment"] = self.strategies[
            "sentiment"
        ].generate_signal(symbol, df)

        signals["pattern_recognition"] = self.strategies[
            "pattern_recognition"
        ].generate_signal(symbol, df)

        # -- Weighted combination --
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
            reason=f"Regime={self.current_regime.value}, Weighted: {', '.join(reasons)}",
        )

        return combined_signal, meta

    # -- Execution --

    def _execute_buy(
        self, symbol: str, signal: Signal, meta: Dict, regime: str = "normal"
    ):
        """Execute a buy order with Kelly + vol-targeted sizing, scaled by leverage."""
        price = meta["price"]
        atr = meta["atr"]
        volatility = meta["volatility"]

        quantity = self.risk_manager.calculate_position_size(
            symbol, price, signal.strength, volatility, regime
        )
        if quantity <= 0:
            return

        # Apply leverage scaling
        current_equity = self.risk_manager._portfolio_value_estimate()
        leverage = self.leverage_manager.get_leverage(
            current_equity=current_equity,
            daily_returns=self._daily_returns,
        )
        quantity = int(quantity * leverage)
        if quantity <= 0:
            return

        executed_price = price
        if self.alpaca_available and self.paper_trade:
            executed_price = self._alpaca_order(symbol, quantity, "buy") or price

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

        executed_price = price
        if self.alpaca_available and self.paper_trade:
            executed_price = (
                self._alpaca_order(symbol, quantity, "sell") or price
            )

        pnl = self.risk_manager.close_position(symbol, executed_price)
        if pnl is not None:
            self.total_pnl += pnl

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
            market_regime=self.current_regime.value,
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

    # -- Alpaca API --

    def _check_alpaca(self) -> bool:
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
                    f"Alpaca order submitted: {side} {qty} {symbol}"
                )
                return float(order.get("filled_avg_price", 0)) or None
            else:
                self.logger.warning(
                    f"Alpaca order failed: {resp.status_code}"
                )
        except Exception as e:
            self.logger.error(f"Alpaca order error: {e}")
        return None

    # -- Status --

    def get_status(self) -> Dict:
        risk = self.risk_manager.get_status()
        improver = self.self_improver.get_report()
        trades = self.logger.get_trade_summary()
        return {
            "cycle_count": self.cycle_count,
            "total_pnl": self.total_pnl,
            "current_regime": self.current_regime.value,
            "risk": risk,
            "strategy_weights": improver["current_weights"],
            "strategy_stats": improver["strategy_stats"],
            "trade_summary": trades,
        }
