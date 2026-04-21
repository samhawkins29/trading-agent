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
from risk_manager import Position, RiskManager
from self_improver import SelfImprover
from strategies.mean_reversion import MeanReversionStrategy, Signal
from strategies.momentum import MomentumStrategy
from strategies.pattern_recognition import MarketRegime, PatternRecognitionStrategy
from strategies.sentiment import SentimentStrategy
from agent_brain import AgentBrain
from news_fetcher import NewsFetcher


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

        # Claude Sonnet decision brain and news fetcher
        self.brain = AgentBrain()
        self.news_fetcher = NewsFetcher()

        # Signal thresholds — can be overridden by learned_params.json
        self._buy_threshold = 0.10
        self._sell_threshold = -0.10

        # Load any previously learned parameters from weekly_review.py
        self._apply_learned_params()

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

        # Reset any stale drawdown halt from a previous (buggy) run so the
        # agent resumes trading cleanly after the fix.
        if getattr(self.risk_manager, "trading_halted", False):
            self.risk_manager.reset_halt()
            self.logger.warning(
                "Cleared stale trading_halted flag from previous session"
            )

        # Reconcile any positions that already exist at Alpaca from a prior
        # session so the duplicate-buy guard can see them and the PERF log
        # reflects real exposure.
        if not dry_run and self.alpaca_connected:
            self._sync_alpaca_positions()

        mode = "DRY RUN" if dry_run else "LIVE PAPER"
        self.logger.info(f"LiveTrader initialized [{mode}]")
        self.logger.info(f"  Symbols: {self.symbols}")
        self.logger.info(f"  Interval: {self.interval_minutes} min")
        self.logger.info(f"  Market hours: {self.market_open_hour}:{self.market_open_minute:02d}"
                         f" - {self.market_close_hour}:{self.market_close_minute:02d} ET")
        self.logger.info(f"  Alpaca connected: {self.alpaca_connected}")
        self.logger.info(f"  AgentBrain: Claude Sonnet decision layer {'ENABLED' if self.brain.enabled else 'DISABLED'}")

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
            self.logger.info(f"  {symbol}: SKIP — insufficient data ({len(df) if not df.empty else 0} rows)")
            return None, {}

        df = DataFetcher.compute_indicators(df)
        current_price = float(df["Close"].iloc[-1])
        atr = float(df["ATR"].iloc[-1]) if "ATR" in df.columns else 0.0
        volatility = float(df["returns"].std()) if "returns" in df.columns else 0.02

        raw_signals: Dict[str, Signal] = {}
        raw_signals["mean_reversion"] = self.strategies["mean_reversion"].generate_signal(symbol, df)
        raw_signals["momentum"] = self.strategies["momentum"].generate_signal(symbol, df)
        raw_signals["sentiment"] = self.strategies["sentiment"].generate_signal(symbol, df)
        raw_signals["pattern_recognition"] = self.strategies["pattern_recognition"].generate_signal(symbol, df)

        # --- Verbose signal logging (per-strategy) ---
        for name, sig in raw_signals.items():
            w = weights.get(name, 0)
            self.logger.info(
                f"  {symbol} | {name:20s}: action={sig.action:4s} "
                f"raw_strength={sig.strength:+.4f}  weight={w:.2f}  "
                f"weighted={sig.strength * w:+.4f}"
            )

        combined_strength = sum(
            weights.get(name, 0) * sig.strength
            for name, sig in raw_signals.items()
        )
        combined_strength = np.clip(combined_strength, -1.0, 1.0)
        dominant = max(raw_signals.items(), key=lambda x: abs(x[1].strength))

        if combined_strength > 0:
            action = "BUY"
        elif combined_strength < 0:
            action = "SELL"
        else:
            action = "HOLD"

        self.logger.info(
            f"  {symbol} | COMBINED: strength={combined_strength:+.4f}  "
            f"action={action}  dominant={dominant[0]}({dominant[1].strength:+.4f})  "
            f"threshold=0.10"
        )

        reasons = [f"{n}={s.strength:.2f}" for n, s in raw_signals.items()]
        combined_signal = Signal(
            symbol=symbol,
            action=action,
            strength=combined_strength,
            strategy=f"combined({dominant[0]})",
            reason=f"Regime={self.current_regime.value}, {', '.join(reasons)}",
        )

        # Include per-strategy detail for the agent brain prompt
        per_strategy = {
            name: {
                "strength": sig.strength,
                "action": sig.action,
                "reason": (sig.reason or "")[:100],
            }
            for name, sig in raw_signals.items()
        }
        meta = {
            "price": current_price,
            "atr": atr,
            "volatility": volatility,
            "per_strategy": per_strategy,
        }
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
            notional = qty * price if price else 0
            self.logger.info(
                f"[DRY RUN] Would {side.upper()} {qty} shares of {symbol} "
                f"@ ~${price:.2f} (notional ~${notional:,.2f})"
            )
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

    # ── Alpaca Position Reconciliation ───────────────────────────────────

    def _fetch_alpaca_positions(self) -> List:
        """Return Alpaca's list_positions() or [] on any failure."""
        if not self.alpaca_connected or self.api is None:
            return []
        try:
            return list(self.api.list_positions())
        except Exception as e:
            self.logger.warning(f"Failed to fetch Alpaca positions: {e}")
            return []

    def _sync_alpaca_positions(self):
        """
        Reconcile risk_manager.positions with the positions actually held
        at Alpaca. Ensures the duplicate-buy guard works and that PERF
        snapshots reflect real exposure after a restart.
        """
        alpaca_positions = self._fetch_alpaca_positions()
        if not alpaca_positions:
            self.logger.info("Alpaca position sync: no existing positions")
            return

        reconciled: List[str] = []
        for ap in alpaca_positions:
            try:
                symbol = ap.symbol
                qty = int(float(ap.qty))
                if qty == 0:
                    continue
                entry_price = float(ap.avg_entry_price)

                # Approximate ATR from recent data so stops are sensible.
                atr = 0.0
                try:
                    df = self.data_fetcher.get_historical(symbol, period="3mo")
                    if not df.empty:
                        df = DataFetcher.compute_indicators(df)
                        if "ATR" in df.columns:
                            atr = float(df["ATR"].iloc[-1])
                except Exception:
                    pass

                stop_loss, take_profit = self.risk_manager.compute_stop_take(
                    entry_price, atr
                )

                self.risk_manager.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=qty,
                    entry_price=entry_price,
                    entry_time=datetime.now(),
                    strategy="reconciled_from_alpaca",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    trailing_stop=stop_loss,
                    highest_price=entry_price,
                )
                reconciled.append(
                    f"{symbol}(qty={qty}, entry=${entry_price:.2f}, "
                    f"SL=${stop_loss:.2f}, TP=${take_profit:.2f})"
                )
            except Exception as e:
                self.logger.warning(
                    f"Failed to reconcile Alpaca position {getattr(ap, 'symbol', '?')}: {e}"
                )

        if reconciled:
            self.logger.info(
                f"Alpaca position sync: reconciled {len(reconciled)} "
                f"position(s): {', '.join(reconciled)}"
            )
        else:
            self.logger.info("Alpaca position sync: no positions reconciled")

    # ── Portfolio Valuation (BUG FIX #1) ─────────────────────────────────

    def _get_actual_portfolio_value(self) -> float:
        """
        Compute the true portfolio value = cash + market value of open
        positions. Prefers Alpaca's authoritative account.equity when
        connected; otherwise falls back to a local mark-to-market using
        latest prices. This replaces the buggy estimate in RiskManager
        which used entry-price notional and broke when the positions dict
        was overwritten by duplicate buys.
        """
        # Preferred: authoritative equity from Alpaca
        if self.alpaca_connected and self.api is not None:
            try:
                account = self.api.get_account()
                return float(account.equity)
            except Exception as e:
                self.logger.warning(f"Falling back to local portfolio valuation: {e}")

        # Fallback: cash + market value of open positions at latest prices
        cash = self.risk_manager.current_capital
        positions = self.risk_manager.positions
        if not positions:
            return cash

        symbols = list(positions.keys())
        try:
            prices = self.data_fetcher.get_latest_prices(symbols)
        except Exception:
            prices = {}

        market_value = 0.0
        for sym, pos in positions.items():
            px = prices.get(sym) or pos.entry_price
            market_value += pos.quantity * px
        return cash + market_value

    def _check_real_drawdown(self) -> bool:
        """
        Drawdown check using the ACTUAL portfolio value. Updates the
        risk manager's peak_capital from real equity so the buggy
        entry-price estimate can't trigger a false halt.
        """
        from config import config as _cfg
        total_value = self._get_actual_portfolio_value()
        rm = self.risk_manager
        rm.peak_capital = max(getattr(rm, "peak_capital", 0.0) or 0.0, total_value)
        if rm.peak_capital > 0:
            drawdown = (rm.peak_capital - total_value) / rm.peak_capital
        else:
            drawdown = 0.0

        max_dd = getattr(_cfg, "max_drawdown_pct", 0.20)
        self.logger.info(
            f"Real portfolio value: ${total_value:,.2f} | "
            f"Peak: ${rm.peak_capital:,.2f} | "
            f"Drawdown: {drawdown:.2%} | Limit: {max_dd:.2%}"
        )
        if drawdown >= max_dd:
            rm.trading_halted = True
            self.logger.error(
                f"MAX DRAWDOWN BREACHED (real): {drawdown:.2%} — trading halted"
            )
            return True
        return False

    # ── Trading Cycle ────────────────────────────────────────────────────

    def run_cycle(self) -> Dict:
        """Execute one full trading cycle."""
        self.cycle_count += 1
        cycle_start = time.time()
        self.logger.info(f"\n{'='*50} LIVE CYCLE {self.cycle_count} {'='*50}")

        actions = {"buys": [], "sells": [], "holds": [], "errors": []}

        # Drawdown check — use the real portfolio value (BUG FIX #1),
        # not the buggy entry-price-based estimate in RiskManager.
        if self._check_real_drawdown():
            self.logger.error("Trading halted — max drawdown breached")
            return actions

        # Check stop-loss / take-profit on existing positions
        open_symbols = list(self.risk_manager.positions.keys())
        if open_symbols:
            prices = self.data_fetcher.get_latest_prices(open_symbols)
            triggers = self.risk_manager.check_stop_loss_take_profit(prices)
            for symbol in triggers:
                price = prices.get(symbol)
                if not price:
                    continue
                pos = self.risk_manager.positions.get(symbol)
                if pos and pos.is_short:
                    self._execute_close_short(symbol, price, "stop_loss_or_take_profit", actions)
                else:
                    self._execute_sell(symbol, price, "stop_loss_or_take_profit", actions)

        # Detect regime
        regime = self._detect_regime()
        active_weights = self._get_active_weights(regime)
        self.logger.info(f"Regime: {regime.value} | Weights: "
                         + ", ".join(f"{k}={v:.2f}" for k, v in active_weights.items()))

        regime_str = regime.value if regime else "normal"

        # ── Phase 1: Collect all strategy signals ─────────────────────────
        signal_summary = {"buy_signals": 0, "sell_signals": 0, "hold_signals": 0,
                          "skipped_data": 0, "skipped_risk": 0, "threshold_filtered": 0}
        all_signals: Dict[str, Signal] = {}
        all_meta: Dict[str, Dict] = {}
        for symbol in self.symbols:
            try:
                signal, meta = self._generate_signals(symbol, active_weights)
                if signal is None:
                    signal_summary["skipped_data"] += 1
                    continue
                all_signals[symbol] = signal
                all_meta[symbol] = meta
                if signal.action == "BUY":
                    signal_summary["buy_signals"] += 1
                elif signal.action == "SELL":
                    signal_summary["sell_signals"] += 1
                else:
                    signal_summary["hold_signals"] += 1
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
                actions["errors"].append({"symbol": symbol, "error": str(e)})

        # ── Phase 2: Claude Sonnet decision layer ─────────────────────────
        brain_actions = None
        if self.brain.enabled and all_signals:
            try:
                signals_for_brain = {
                    sym: {
                        "combined_strength": sig.strength,
                        "combined_action": sig.action,
                        "per_strategy": all_meta[sym].get("per_strategy", {}),
                    }
                    for sym, sig in all_signals.items()
                }
                strong_symbols = [
                    s for s, sig in all_signals.items() if abs(sig.strength) > 0.10
                ]
                news = (
                    self.news_fetcher.fetch_news_batch(strong_symbols)
                    if strong_symbols else {}
                )
                portfolio = self._get_portfolio_state()
                positions_for_brain = {
                    sym: {
                        "quantity": pos.quantity,
                        "entry_price": pos.entry_price,
                        "unrealized_pnl_pct": (
                            (all_meta.get(sym, {}).get("price", pos.entry_price) - pos.entry_price)
                            / pos.entry_price if pos.entry_price > 0 else 0.0
                        ),
                    }
                    for sym, pos in self.risk_manager.positions.items()
                }
                brain_actions = self.brain.decide(
                    signals=signals_for_brain,
                    positions=positions_for_brain,
                    portfolio=portfolio,
                    regime=regime_str,
                    news=news or None,
                )
                if brain_actions is not None:
                    self.logger.info(
                        f"AgentBrain: {len(brain_actions)} action(s) — "
                        + (
                            ", ".join(
                                f"{a['symbol']}:{a['action']}({a['confidence']:.0%})"
                                for a in brain_actions
                            ) or "all hold"
                        )
                    )
            except Exception as e:
                self.logger.error(f"AgentBrain error (falling back to threshold system): {e}")
                brain_actions = None

        # ── Phase 3: Execute decisions ─────────────────────────────────────
        if brain_actions is not None:
            self._execute_brain_actions(brain_actions, all_signals, all_meta, regime_str, actions)
            brain_symbols = {a["symbol"] for a in brain_actions}
            error_symbols = {e.get("symbol") for e in actions["errors"]}
            for sym in self.symbols:
                if sym not in brain_symbols and sym not in error_symbols:
                    actions["holds"].append(sym)
        else:
            # Fallback: original threshold-based system
            for symbol, signal in all_signals.items():
                meta = all_meta[symbol]
                can_trade, reason = self.risk_manager.can_trade(symbol)
                if not can_trade:
                    self.logger.info(f"  {symbol}: SKIP ({reason})")
                    actions["holds"].append(symbol)
                    signal_summary["skipped_risk"] += 1
                    continue
                if signal.action == "BUY" and signal.strength > self._buy_threshold:
                    self._execute_buy(symbol, signal, meta, regime_str, actions)
                elif signal.action == "SELL" and signal.strength < self._sell_threshold:
                    if symbol in self.risk_manager.positions:
                        self._execute_sell(symbol, meta["price"], signal.reason, actions)
                    else:
                        self.logger.info(
                            f"  {symbol}: SELL signal ({signal.strength:+.4f}) but no position to sell"
                        )
                        actions["holds"].append(symbol)
                else:
                    if signal.action in ("BUY", "SELL"):
                        signal_summary["threshold_filtered"] += 1
                        self.logger.info(
                            f"  {symbol}: {signal.action} signal too weak "
                            f"({signal.strength:+.4f}), threshold={self._buy_threshold}"
                        )
                    actions["holds"].append(symbol)

        # --- Cycle signal summary ---
        self.logger.info(
            f"Signal summary: {signal_summary['buy_signals']} BUY, "
            f"{signal_summary['sell_signals']} SELL, "
            f"{signal_summary['hold_signals']} HOLD | "
            f"Filtered below threshold: {signal_summary['threshold_filtered']} | "
            f"Skipped (data): {signal_summary['skipped_data']}, "
            f"Skipped (risk): {signal_summary['skipped_risk']}"
        )

        # Periodic weight update
        if self.cycle_count % 5 == 0:
            self.self_improver.update_weights(regime_name=regime.value)

        # Performance snapshot — pull every field from authoritative
        # sources (Alpaca + real portfolio value), not the buggy
        # risk_manager estimates.
        real_total_value = self._get_actual_portfolio_value()

        # Prefer Alpaca for cash, positions, exposure.
        account_info = self.get_account_info()
        alpaca_positions = self._fetch_alpaca_positions()

        if account_info is not None:
            cash_balance = account_info["cash"]
            long_mv = account_info.get("long_market_value", 0.0) or 0.0
            short_mv = abs(account_info.get("short_market_value", 0.0) or 0.0)
            exposure_value = long_mv + short_mv
        else:
            cash_balance = self.risk_manager.current_capital
            exposure_value = sum(
                pos.quantity * pos.entry_price
                for pos in self.risk_manager.positions.values()
            )

        open_positions_count = (
            len(alpaca_positions) if alpaca_positions
            else len(self.risk_manager.positions)
        )
        exposure_pct = (
            exposure_value / real_total_value if real_total_value > 0 else 0.0
        )

        # Real drawdown from updated peak_capital (set in _check_real_drawdown).
        peak = getattr(self.risk_manager, "peak_capital", 0.0) or 0.0
        real_drawdown = (
            (peak - real_total_value) / peak if peak > 0 else 0.0
        )

        self.logger.log_performance_snapshot({
            "cycle": self.cycle_count,
            "regime": regime.value,
            "total_value": real_total_value,
            "capital": cash_balance,
            "open_positions": open_positions_count,
            "exposure_pct": f"{exposure_pct:.2%}",
            "drawdown_pct": f"{real_drawdown:.2%}",
            "total_pnl": self.total_pnl,
            "buys": len(actions["buys"]),
            "sells": len(actions["sells"]),
        })

        elapsed = time.time() - cycle_start
        self.logger.info(f"Cycle {self.cycle_count} completed in {elapsed:.1f}s | "
                         f"PnL: ${self.total_pnl:+,.2f}")
        return actions

    def _apply_learned_params(self):
        """Load parameters produced by weekly_review.py and apply them."""
        import json as _json
        path = "learned_params.json"
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                params = _json.load(f)
            if "strategy_weights" in params:
                for name, weight in params["strategy_weights"].items():
                    if name in self.self_improver.weights:
                        self.self_improver.weights[name] = weight
            if "buy_threshold" in params:
                self._buy_threshold = float(params["buy_threshold"])
            if "sell_threshold" in params:
                self._sell_threshold = float(params["sell_threshold"])
            updated = params.get("updated_at", "unknown")
            self.logger.info(
                f"Loaded learned params from {path} (updated {updated}) | "
                f"buy_threshold={self._buy_threshold}, sell_threshold={self._sell_threshold}"
            )
        except Exception as e:
            self.logger.warning(f"Could not load learned params: {e}")

    def _get_portfolio_state(self) -> Dict:
        """Return current portfolio state for the agent brain."""
        account = self.get_account_info()
        if account:
            return {
                "portfolio_value": account["portfolio_value"],
                "cash": account["cash"],
                "buying_power": account["buying_power"],
                "equity": account["equity"],
            }
        estimated = self._get_actual_portfolio_value()
        return {
            "portfolio_value": estimated,
            "cash": self.risk_manager.current_capital,
            "buying_power": self.risk_manager.current_capital,
            "equity": estimated,
        }

    def _execute_brain_actions(
        self,
        brain_actions: List[Dict],
        all_signals: Dict[str, "Signal"],
        all_meta: Dict[str, Dict],
        regime_str: str,
        actions: Dict,
    ):
        """
        Execute trades from Claude Sonnet's decision list.

        BUY logic:
          - Holding short  → close the short (don't also open a long)
          - Holding long   → skip (duplicate-buy guard)
          - No position    → open long normally

        SELL logic:
          - Holding long   → close the long
          - Holding short  → skip (already short)
          - No position    → open short if shorting_enabled and confidence > 0.60
        """
        for brain_action in brain_actions:
            symbol = brain_action.get("symbol", "")
            action = brain_action.get("action", "HOLD")
            confidence = max(0.0, min(1.0, float(brain_action.get("confidence", 0.5))))
            rationale = brain_action.get("rationale", "")

            if not symbol or action == "HOLD":
                continue

            meta = all_meta.get(symbol)
            if meta is None:
                try:
                    price = self._get_latest_price(symbol)
                    if price <= 0:
                        continue
                    meta = {"price": price, "atr": 0.0, "volatility": 0.02, "per_strategy": {}}
                except Exception:
                    continue

            can_trade, skip_reason = self.risk_manager.can_trade(symbol)
            original = all_signals.get(symbol)
            original_reason = original.reason if original else ""
            full_reason = f"[Brain] {rationale}" + (
                f" | {original_reason}" if original_reason else ""
            )

            existing_pos = self.risk_manager.positions.get(symbol)

            if action == "BUY":
                if existing_pos and existing_pos.is_short:
                    # Close the short on a BUY signal
                    self._execute_close_short(symbol, meta["price"], full_reason, actions)
                elif existing_pos:
                    self.logger.info(f"  {symbol}: already holding long, skipping BUY")
                else:
                    if not can_trade:
                        self.logger.info(f"  {symbol}: SKIP — brain BUY but {skip_reason}")
                        continue
                    synthetic = Signal(
                        symbol=symbol,
                        action="BUY",
                        strength=confidence,
                        strategy="agent_brain",
                        reason=full_reason,
                    )
                    self._execute_buy(symbol, synthetic, meta, regime_str, actions)

            elif action == "SELL":
                if existing_pos and not existing_pos.is_short:
                    # Close existing long
                    self._execute_sell(symbol, meta["price"], full_reason, actions)
                elif existing_pos and existing_pos.is_short:
                    self.logger.info(f"  {symbol}: already short, ignoring additional SELL")
                elif config.shorting_enabled and confidence > 0.80:
                    if not can_trade:
                        self.logger.info(f"  {symbol}: SKIP — brain SHORT but {skip_reason}")
                        continue
                    self._execute_short_entry(symbol, confidence, meta, regime_str, full_reason, actions)
                else:
                    self.logger.info(
                        f"  {symbol}: Brain SELL — no position, "
                        + ("low confidence ({confidence:.0%})" if confidence <= 0.80 else "shorting disabled")
                    )

    def _execute_short_entry(
        self,
        symbol: str,
        confidence: float,
        meta: Dict,
        regime: str,
        reason: str,
        actions: Dict,
    ):
        """Open a short position via Alpaca (sell without owning shares)."""
        price = meta["price"]
        volatility = meta.get("volatility", 0.02)
        atr = meta.get("atr", 0.0)

        quantity = self.risk_manager.calculate_position_size(
            symbol, price, confidence, volatility, regime
        )
        if quantity <= 0:
            self.logger.info(f"  {symbol}: short size = 0, skipping")
            return

        # Apply leverage
        equity = self.risk_manager._portfolio_value_estimate()
        leverage = self.leverage_manager.get_leverage(
            current_equity=equity, daily_returns=self._daily_returns
        )
        quantity = max(1, int(quantity * leverage) // 2)  # shorts capped at half long size
        if quantity <= 0:
            return

        # Submit a sell order with no existing position = short entry
        success, filled_price = self._submit_order(symbol, quantity, "sell")
        if not success:
            return

        self.risk_manager.open_short_position(symbol, quantity, filled_price, "agent_brain_short", atr)

        self._log_trade(
            symbol=symbol, action="SHORT", quantity=quantity,
            price=price, filled_price=filled_price,
            strategy="agent_brain_short", signal_strength=-confidence,
            regime=regime, leverage=leverage,
            portfolio_value=self.risk_manager._portfolio_value_estimate(),
            pnl=0.0, reason=reason,
        )
        self.logger.log_trade(
            symbol=symbol, action="SHORT", quantity=quantity,
            price=filled_price, strategy="agent_brain_short",
            signal_strength=-confidence,
            portfolio_value=self.risk_manager._portfolio_value_estimate(),
            reason=reason,
        )
        self.logger.info(
            f"  SHORT entered: {quantity} {symbol} @ ${filled_price:.2f} | {reason[:80]}"
        )
        actions["sells"].append(symbol)

    def _execute_close_short(
        self, symbol: str, price: float, reason: str, actions: Dict,
    ):
        """Close a short position by submitting a BUY order to cover."""
        pos = self.risk_manager.positions.get(symbol)
        if pos is None or not pos.is_short:
            return

        quantity = pos.quantity
        strategy = pos.strategy
        entry_price = pos.entry_price

        success, filled_price = self._submit_order(symbol, quantity, "buy")
        if not success:
            return

        pnl = self.risk_manager.close_position(symbol, filled_price)
        if pnl is not None:
            self.total_pnl += pnl

        holding_hours = (datetime.now() - pos.entry_time).total_seconds() / 3600
        self.self_improver.record_experience(
            symbol=symbol, strategy=strategy, action="COVER",
            signal_strength=0.0, entry_price=entry_price,
            exit_price=filled_price,
            holding_period_hours=holding_hours,
            market_regime=self.current_regime.value,
        )

        self._log_trade(
            symbol=symbol, action="COVER", quantity=quantity,
            price=price, filled_price=filled_price,
            strategy=strategy, signal_strength=0.0,
            regime=self.current_regime.value, leverage=1.0,
            portfolio_value=self.risk_manager._portfolio_value_estimate(),
            pnl=pnl or 0.0, reason=reason,
        )
        self.logger.log_trade(
            symbol=symbol, action="COVER", quantity=quantity,
            price=filled_price, strategy=strategy,
            signal_strength=0.0,
            portfolio_value=self.risk_manager._portfolio_value_estimate(),
            reason=reason,
        )
        actions["buys"].append(symbol)

    def _already_holding(self, symbol: str) -> bool:
        """
        Check whether we already have an open position in `symbol`, either
        locally tracked by the risk manager or reported by Alpaca. Used to
        prevent duplicate buy fills across cycles (BUG FIX #2).
        """
        if symbol in self.risk_manager.positions:
            return True
        if self.alpaca_connected and self.api is not None:
            try:
                pos = self.api.get_position(symbol)
                if pos and float(getattr(pos, "qty", 0) or 0) != 0:
                    return True
            except Exception:
                # Alpaca raises when no position exists — that's fine.
                pass
        return False

    def _execute_buy(
        self, symbol: str, signal: Signal, meta: Dict, regime: str,
        actions: Dict,
    ):
        """Execute a buy with Kelly + vol + leverage sizing."""
        # BUG FIX #2: skip if we already hold this symbol, so the trader
        # doesn't repeatedly buy the same name every cycle.
        if self._already_holding(symbol):
            self.logger.info(
                f"  {symbol}: already holding position, skipping BUY"
            )
            actions["holds"].append(symbol)
            return

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
