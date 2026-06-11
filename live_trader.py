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
import json
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
from daily_review import DailyReview
from data_fetcher import DataFetcher
from entry_filter import entry_filter
from leverage_manager import LeverageConfig, LeverageManager
from logger import TradeLogger
from risk_manager import Position, RiskManager
from self_improver import SelfImprover
from strategies.mean_reversion import MeanReversionStrategy, Signal
from strategies.momentum import MomentumStrategy
from strategies.pattern_recognition import MarketRegime, PatternRecognitionStrategy
from strategies.sentiment import SentimentStrategy
from trade_journal import TradeJournal
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

        # Limit-order tracking: consecutive cancel count per symbol (feature #2)
        self._consecutive_cancels: Dict[str, int] = {}

        # Last error message from _submit_order per symbol — surfaced to callers
        # so failed-order log lines can include the actual Alpaca response
        # (e.g. "insufficient qty available") rather than a generic failure.
        self._last_order_error: Dict[str, str] = {}

        # Conviction decay: last brain recommendation per symbol (feature #3)
        # {symbol: {"action": str, "confidence": float, "cycle": int}}
        self._last_recommendation: Dict[str, Dict] = {}

        # Consecutive conviction-decay skip counter per symbol. Capped at 3
        # so similar signals don't block trading indefinitely.
        self._conviction_skip_count: Dict[str, int] = {}

        # Per-symbol cooldown after closing a position (feature #3)
        # {symbol: datetime when position was closed}
        self._position_closed_at: Dict[str, datetime] = {}

        # Consecutive brain BUY-signal count per symbol — used by
        # _review_open_positions to detect thesis reversal on shorts.
        self._brain_buy_streak: Dict[str, int] = {}

        # Broker-native resting stops: {symbol: broker_stop_order_id}. Each open
        # position gets a GTC stop order resting at the broker so it is
        # protected even when no cycle is running. Cancelled when the position
        # is closed by any other path.
        self._resting_stops: Dict[str, str] = {}

        # Real-time kill switch state.
        self._kill_switch_cfg = getattr(config, "kill_switch", {}) or {}
        self._kill_switch_tripped = False
        self._day_start_equity: Optional[float] = None
        self._day_start_date: Optional[str] = None
        # Persistent halt flag — survives process restarts so a kill-switch
        # trip cannot be silently undone by simply restarting the agent.
        self._halt_flag_path = os.path.join(config.log_dir, "KILL_SWITCH.flag")

        # Trade journal path for entry fingerprinting (feature #9a)
        self._trade_journal_path = os.path.join(config.log_dir, "trade_journal.jsonl")
        os.makedirs(config.log_dir, exist_ok=True)

        # Daily plain-text log: prune old files and reset per-cycle attempts.
        self._cycle_attempts: List[Dict] = []
        self._cleanup_old_daily_logs(keep_days=7)
        self._write_daily_log(
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
            f"=== Trader started ({'DRY RUN' if dry_run else 'LIVE PAPER'}) "
            f"| symbols={','.join(self.symbols)} ==="
        )

        # Structured trade journal (data/trade_journal.jsonl) consumed by daily_review.py
        self.journal = TradeJournal()

        # A tripped kill switch is a HARD halt: if the persistent flag exists,
        # stay halted across restarts (manual removal of the flag file is
        # required to resume). This must be checked before the soft-halt reset
        # below so a restart cannot silently undo a liquidation event.
        if os.path.exists(self._halt_flag_path):
            self._kill_switch_tripped = True
            self.risk_manager.trading_halted = True
            self.logger.error(
                f"KILL SWITCH ACTIVE — persistent halt flag present at "
                f"{self._halt_flag_path}. Trading stays halted until the flag "
                f"is manually removed and the cause is investigated."
            )
        elif getattr(self.risk_manager, "trading_halted", False):
            # Reset any stale soft drawdown halt from a previous (buggy) run so
            # the agent resumes trading cleanly after the fix.
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

    # ── Daily Human-Readable Log ─────────────────────────────────────────

    def _daily_log_path(self) -> str:
        """Return path to today's plain-text daily log (logs/daily-YYYY-MM-DD.log)."""
        today = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(config.log_dir, f"daily-{today}.log")

    def _write_daily_log(self, line: str):
        """
        Append a single line to today's daily log. Failures are swallowed —
        logging must never break trading.
        """
        try:
            path = self._daily_log_path()
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            try:
                self.logger.warning(f"Daily log write failed: {exc}")
            except Exception:
                pass

    def _log_trade_attempt(
        self, symbol: str, action: str, result: str, reason: str = "",
    ):
        """
        Record one trade-attempt line to the daily log AND the in-memory
        cycle list (used to build the cycle summary). `result` should be
        EXECUTED / BLOCKED / FAILED.
        """
        self._cycle_attempts.append({
            "symbol": symbol, "action": action, "result": result, "reason": reason,
        })
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"{ts} | TRADE | {symbol:<6s} | {action:<10s} | {result:<8s}"
        if reason:
            line += f" | {reason[:140]}"
        self._write_daily_log(line)

    def _cleanup_old_daily_logs(self, keep_days: int = 7):
        """Delete daily-YYYY-MM-DD.log files older than `keep_days` on startup."""
        log_dir = config.log_dir
        if not os.path.isdir(log_dir):
            return
        cutoff = datetime.now() - timedelta(days=keep_days)
        removed = 0
        for name in os.listdir(log_dir):
            if not (name.startswith("daily-") and name.endswith(".log")):
                continue
            date_str = name[len("daily-"):-len(".log")]
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue
            if dt < cutoff:
                try:
                    os.remove(os.path.join(log_dir, name))
                    removed += 1
                except OSError:
                    pass
        if removed:
            self.logger.info(
                f"Daily log cleanup: removed {removed} file(s) older than {keep_days} days"
            )

    def _write_cycle_summary(
        self, regime: str, position_count: int, actions: Dict,
    ):
        """
        Append the end-of-cycle summary line to today's daily log.

        Format:
        HH:MM:SS | Cycle N | regime | Positions: N | Trades: N buys, N sells | PnL: $X | Actions: ...
        """
        ts = datetime.now().strftime("%H:%M:%S")
        n_buys = len(actions.get("buys", []))
        n_sells = len(actions.get("sells", []))

        action_parts: List[str] = []
        if actions.get("buys"):
            action_parts.append("BUYS=[" + ",".join(actions["buys"]) + "]")
        if actions.get("sells"):
            action_parts.append("SELLS=[" + ",".join(actions["sells"]) + "]")

        # Summarize blocked / failed attempts from this cycle for visibility.
        blocked = [a for a in self._cycle_attempts if a["result"] in ("BLOCKED", "FAILED")]
        if blocked:
            blocked_parts = [
                f"{a['symbol']}({a['action'][:1]}):{a['reason'][:40]}"
                for a in blocked[:6]
            ]
            extra = f"+{len(blocked) - 6}" if len(blocked) > 6 else ""
            action_parts.append("BLOCKED=[" + ", ".join(blocked_parts) + extra + "]")

        if actions.get("errors"):
            err_syms = [e.get("symbol", "?") for e in actions["errors"]]
            action_parts.append("ERRORS=[" + ",".join(err_syms) + "]")

        actions_str = "; ".join(action_parts) if action_parts else "no trades"

        line = (
            f"{ts} | Cycle {self.cycle_count} | {regime} | "
            f"Positions: {position_count} | "
            f"Trades: {n_buys} buys, {n_sells} sells | "
            f"PnL: ${self.total_pnl:+,.2f} | "
            f"Actions: {actions_str}"
        )
        self._write_daily_log(line)

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
    ) -> Tuple[Optional[Signal], Dict, Optional[pd.DataFrame]]:
        """Run all strategies on a symbol, return (signal, meta, df)."""
        df = self.data_fetcher.get_historical(symbol, period="2y")
        if df.empty or len(df) < 50:
            self.logger.info(f"  {symbol}: SKIP — insufficient data ({len(df) if not df.empty else 0} rows)")
            return None, {}, None

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
        return combined_signal, meta, df

    # ── Order Execution ──────────────────────────────────────────────────

    def _submit_order(
        self, symbol: str, qty: int, side: str
    ) -> Tuple[bool, float]:
        """
        Submit a limit order via Alpaca (or simulate in dry-run mode).

        Limit price offsets:
          - buys:  ask - 0.1%  (slightly below ask to get passive fill)
          - sells: bid + 0.1%  (slightly above bid to get passive fill)

        If not filled within 2 minutes the order is cancelled. After
        3 consecutive cancels for the same symbol we fall back to a
        market order for that one attempt.

        Returns (success, filled_price).
        """
        # Reset any previous error for this symbol so callers see only the
        # outcome of THIS submission attempt.
        self._last_order_error.pop(symbol, None)

        if self.dry_run:
            price = self._get_latest_price(symbol)
            notional = qty * price if price else 0
            self.logger.info(
                f"[DRY RUN] Would {side.upper()} {qty} shares of {symbol} "
                f"@ ~${price:.2f} (notional ~${notional:,.2f})"
            )
            self._consecutive_cancels[symbol] = 0
            return True, price

        if not self.alpaca_connected or not self.api:
            self.logger.warning(f"Alpaca not connected — cannot submit {side} {qty} {symbol}")
            self._last_order_error[symbol] = "alpaca not connected"
            return False, 0.0

        cancels = self._consecutive_cancels.get(symbol, 0)
        use_market = cancels >= 3

        try:
            ref_price = self._get_latest_price(symbol)
            if ref_price <= 0:
                self.logger.warning(f"Cannot get reference price for {symbol}, skipping")
                self._last_order_error[symbol] = "no reference price available"
                return False, 0.0

            if use_market:
                order_type = "market"
                limit_price = None
                self.logger.info(
                    f"3 consecutive cancels for {symbol} — falling back to market order"
                )
            else:
                order_type = "limit"
                if side == "buy":
                    limit_price = round(ref_price * 0.999, 2)  # ask - 0.1%
                else:
                    limit_price = round(ref_price * 1.001, 2)  # bid + 0.1%

            order_kwargs = dict(
                symbol=symbol,
                qty=str(qty),
                side=side,
                type=order_type,
                time_in_force="day",
            )
            if limit_price is not None:
                order_kwargs["limit_price"] = str(limit_price)

            order = self.api.submit_order(**order_kwargs)
            self.logger.info(
                f"Order submitted: {side} {qty} {symbol} "
                f"type={order_type} limit={limit_price} | ID: {order.id}"
            )

            # Poll for up to 2 minutes
            filled_price = 0.0
            deadline = time.time() + 120
            poll_interval = 5
            while time.time() < deadline:
                time.sleep(poll_interval)
                updated = self.api.get_order(order.id)
                if updated.status == "filled":
                    filled_price = float(updated.filled_avg_price or 0)
                    self._consecutive_cancels[symbol] = 0
                    break
                elif updated.status in ("canceled", "expired", "rejected"):
                    reject_reason = (
                        getattr(updated, "reject_reason", None)
                        or getattr(updated, "failed_at", None)
                        or ""
                    )
                    msg = f"order {updated.status}"
                    if reject_reason:
                        msg += f": {reject_reason}"
                    self.logger.warning(
                        f"Order {updated.status}: {side} {qty} {symbol} "
                        f"(limit={limit_price}) {reject_reason}"
                    )
                    self._consecutive_cancels[symbol] = cancels + 1
                    self._last_order_error[symbol] = msg
                    return False, 0.0

            if filled_price == 0:
                # Still pending after 2 min — cancel it
                try:
                    self.api.cancel_order(order.id)
                    self.logger.warning(
                        f"Limit order not filled in 2 min, cancelled: "
                        f"{side} {qty} {symbol} @ {limit_price}"
                    )
                except Exception:
                    pass
                self._consecutive_cancels[symbol] = cancels + 1
                self._last_order_error[symbol] = "limit order not filled in 2 min"
                return False, 0.0

            return True, filled_price

        except Exception as e:
            # Pull the richest error message we can — Alpaca's APIError stores
            # the server response body, which is what we want surfaced ("insufficient
            # qty available", "asset not shortable", etc.) rather than a generic repr.
            err_msg = str(e) or repr(e)
            response = getattr(e, "response", None)
            if response is not None:
                body = getattr(response, "text", None) or getattr(response, "content", None)
                if body:
                    err_msg = f"{err_msg} | response={body}"
            self.logger.error(
                f"Order submission failed: {side} {qty} {symbol} -> {err_msg}"
            )
            self._last_order_error[symbol] = err_msg
            return False, 0.0

    def _get_latest_price(self, symbol: str) -> float:
        """Get the latest price for a symbol."""
        prices = self.data_fetcher.get_latest_prices([symbol])
        return prices.get(symbol, 0.0)

    # ── Cooldown + Conviction Decay (feature #3) ─────────────────────────

    def _is_in_cooldown(self, symbol: str) -> bool:
        """Return True if symbol is within the 30-min post-close cooldown."""
        closed_at = self._position_closed_at.get(symbol)
        if closed_at is None:
            return False
        elapsed = (datetime.now() - closed_at).total_seconds() / 60
        if elapsed < 30:
            self.logger.info(
                f"  {symbol}: cooldown active ({elapsed:.1f}/30 min since close) — skip"
            )
            return True
        return False

    def _is_conviction_decay(self, symbol: str, action: str, confidence: float) -> bool:
        """
        Skip re-entry if brain gives same direction + similar confidence as last cycle.
        'Similar' = within 5% of last confidence.

        Capped at 3 consecutive skips per symbol — on the 4th identical signal
        the trade is allowed through and the counter resets, so persistent
        signals don't block trading indefinitely.
        """
        last = self._last_recommendation.get(symbol)
        if last is None:
            self._conviction_skip_count[symbol] = 0
            return False
        if last["action"] != action:
            self._conviction_skip_count[symbol] = 0
            return False
        delta = abs(confidence - last["confidence"])
        if delta < 0.05:
            skip_count = self._conviction_skip_count.get(symbol, 0)
            if skip_count < 3:
                self._conviction_skip_count[symbol] = skip_count + 1
                self.logger.info(
                    f"  {symbol}: conviction decay skip {skip_count + 1}/3 -- same "
                    f"{action} @ {confidence:.2f} as cycle {last['cycle']} "
                    f"({last['confidence']:.2f})"
                )
                return True
            else:
                self._conviction_skip_count[symbol] = 0
                self.logger.info(
                    f"  {symbol}: conviction decay max skips reached -- allowing "
                    f"{action} @ {confidence:.2f} through"
                )
                return False
        self._conviction_skip_count[symbol] = 0
        return False

    def _record_recommendation(self, symbol: str, action: str, confidence: float):
        """Store the latest brain recommendation for conviction decay checks."""
        self._last_recommendation[symbol] = {
            "action": action,
            "confidence": confidence,
            "cycle": self.cycle_count,
        }

    # ── Entry Fingerprint Journal (feature #9a) ──────────────────────────

    def _log_entry_fingerprint(
        self,
        symbol: str,
        side: str,
        strategy: str,
        confidence: float,
        meta: Dict,
        df: Optional[pd.DataFrame] = None,
    ):
        """
        Append a full signal-state snapshot to the trade journal JSONL
        at the moment of entry. Used for counterfactual attribution later.
        """
        per_strategy = meta.get("per_strategy", {})
        rsi = None
        z_score = None
        if df is not None:
            try:
                if "RSI" in df.columns:
                    rsi = float(df["RSI"].iloc[-1])
            except Exception:
                pass

        # Pull z-score from mean_reversion per_strategy detail
        mr = per_strategy.get("mean_reversion", {})
        reason_text = mr.get("reason", "")
        if "z=" in reason_text:
            try:
                z_score = float(reason_text.split("z=")[1].split(",")[0].split(" ")[0])
            except Exception:
                pass

        entry = {
            "type": "entry_fingerprint",
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "side": side,
            "strategy": strategy,
            "confidence": confidence,
            "regime": self.current_regime.value,
            "price": meta.get("price"),
            "atr": meta.get("atr"),
            "volatility": meta.get("volatility"),
            "rsi": rsi,
            "z_score": z_score,
            "per_strategy": per_strategy,
        }
        try:
            with open(self._trade_journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            self.logger.warning(f"entry_fingerprint log failed: {exc}")

    def _log_exit_counterfactual(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        strategy: str,
        pnl: float,
        meta: Dict,
    ):
        """
        Log which strategies were 'right' vs 'wrong' at entry time.
        Uses current per_strategy signals as a proxy for entry signals.
        """
        per_strategy = meta.get("per_strategy", {}) if meta else {}
        correct = {}
        was_long = pnl >= 0

        for strat, detail in per_strategy.items():
            strat_action = detail.get("action", "HOLD")
            if strat_action == "BUY":
                correct[strat] = was_long
            elif strat_action == "SELL":
                correct[strat] = not was_long
            else:
                correct[strat] = None  # HOLD — no opinion

        entry = {
            "type": "counterfactual",
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "strategy": strategy,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "regime": self.current_regime.value,
            "strategy_correct": correct,
        }
        try:
            with open(self._trade_journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as exc:
            self.logger.warning(f"counterfactual log failed: {exc}")

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
                is_short = qty < 0
                abs_qty = abs(qty)
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

                if is_short:
                    # Short positions need inverted stop/TP since compute_stop_take
                    # is long-oriented.
                    stop_loss = entry_price * (1 + 0.08)
                    take_profit = entry_price * (1 - 0.20)
                    if atr > 0:
                        stop_loss = max(stop_loss, entry_price + 3.0 * atr)
                else:
                    stop_loss, take_profit = self.risk_manager.compute_stop_take(
                        entry_price, atr, strategy="default"
                    )

                # Derive actual entry_time from Alpaca's order history so
                # days_held reflects real age (otherwise time-based exits in
                # _review_open_positions never fire after a restart).
                entry_time = datetime.now()
                try:
                    side_match = "sell" if is_short else "buy"
                    orders = self.api.list_orders(
                        status="filled",
                        symbols=[symbol],
                        limit=500,
                        direction="asc",
                    )
                    for o in orders:
                        if getattr(o, "side", None) == side_match and getattr(o, "filled_at", None):
                            ft = o.filled_at
                            if hasattr(ft, "tzinfo") and ft.tzinfo is not None:
                                ft = ft.replace(tzinfo=None)
                            entry_time = ft
                            break
                except Exception as e:
                    self.logger.warning(
                        f"Could not derive entry_time for {symbol} from orders: {e}"
                    )

                # Reconciled positions have no known strategy, so the generic
                # 30d "default" max_holding_days is too lenient — stale shorts
                # bled for 3+ weeks before _review_open_positions would fire.
                # Cap reconciled holds at 14 calendar days so the time-based
                # exit kicks in within ~2 weeks of restart.
                reconciled_max_days = 14.0

                self.risk_manager.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=abs_qty,
                    entry_price=entry_price,
                    entry_time=entry_time,
                    strategy="reconciled_from_alpaca",
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    trailing_stop=stop_loss if not is_short else None,
                    highest_price=entry_price if not is_short else None,
                    max_holding_days=reconciled_max_days,
                    is_short=is_short,
                )
                reconciled.append(
                    f"{symbol}(qty={abs_qty}, {'short' if is_short else 'long'}, "
                    f"entry=${entry_price:.2f}, SL=${stop_loss:.2f}, TP=${take_profit:.2f})"
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

    # ── Real-Time Kill Switch / Circuit Breaker ──────────────────────────

    def _check_kill_switch(self) -> bool:
        """
        Hard circuit breaker on REAL broker equity.

        Distinct from _check_real_drawdown (which only sets a soft
        trading_halted flag): when this trips it cancels all open orders,
        liquidates every position at market, and writes a persistent halt flag
        so the halt survives a restart. It is designed to be cheap enough to
        call at the very top of every cycle, and can also be called
        independently (e.g. from a watchdog) to approximate real-time
        protection without the cycle loop.

        Returns True if the kill switch is (or just became) tripped.
        """
        if self._kill_switch_tripped:
            return True
        cfg = self._kill_switch_cfg
        if not cfg.get("enabled", True):
            return False

        equity = self._get_actual_portfolio_value()
        if equity <= 0:
            # No reliable equity read — do nothing rather than liquidate on a
            # transient data failure.
            return False

        # Reset the intraday baseline on the first check of a new day.
        today = datetime.now().strftime("%Y-%m-%d")
        if self._day_start_date != today:
            self._day_start_date = today
            self._day_start_equity = equity

        rm = self.risk_manager
        peak = max(getattr(rm, "peak_capital", 0.0) or 0.0, equity)
        rm.peak_capital = peak
        dd = (peak - equity) / peak if peak > 0 else 0.0
        day_start = self._day_start_equity or equity
        daily_loss = (day_start - equity) / day_start if day_start > 0 else 0.0

        dd_limit = float(cfg.get("max_drawdown_limit", 0.20))
        daily_limit = float(cfg.get("daily_loss_limit", 0.08))
        floor = float(cfg.get("min_equity_floor", 0.0))

        reason = None
        if dd >= dd_limit:
            reason = f"peak-to-trough drawdown {dd:.2%} >= {dd_limit:.2%}"
        elif daily_loss >= daily_limit:
            reason = f"intraday loss {daily_loss:.2%} >= {daily_limit:.2%}"
        elif floor > 0 and equity < floor:
            reason = f"equity ${equity:,.2f} below floor ${floor:,.2f}"

        if reason:
            self._trip_kill_switch(reason)
            return True
        return False

    def _trip_kill_switch(self, reason: str):
        """Liquidate everything and halt hard. Safe in dry-run (logs only)."""
        self._kill_switch_tripped = True
        self.risk_manager.trading_halted = True
        self.logger.error(f"*** KILL SWITCH TRIPPED *** {reason} — liquidating all positions")
        self._write_daily_log(
            f"{datetime.now().strftime('%H:%M:%S')} | KILL SWITCH | {reason} | liquidating"
        )

        # 1) Cancel all resting/open orders so nothing fills after liquidation.
        if not self.dry_run and self.alpaca_connected and self.api:
            try:
                self.api.cancel_all_orders()
            except Exception as e:
                self.logger.error(f"Kill switch: cancel_all_orders failed: {e}")
        self._resting_stops.clear()

        # 2) Liquidate every tracked position at market.
        for symbol in list(self.risk_manager.positions.keys()):
            pos = self.risk_manager.positions.get(symbol)
            if pos is None:
                continue
            close_side = "buy" if getattr(pos, "is_short", False) else "sell"
            qty = pos.quantity
            if self.dry_run or not self.alpaca_connected or not self.api:
                self.logger.error(
                    f"[KILL SWITCH] would {close_side.upper()} {qty} {symbol} (market) to flatten"
                )
                continue
            try:
                self.api.submit_order(
                    symbol=symbol, qty=str(qty), side=close_side,
                    type="market", time_in_force="day",
                )
                self.logger.error(f"[KILL SWITCH] submitted market {close_side} {qty} {symbol}")
            except Exception as e:
                self.logger.error(f"Kill switch: failed to flatten {symbol}: {e}")

        # 3) Persist the halt so a restart cannot silently resume trading.
        try:
            os.makedirs(os.path.dirname(self._halt_flag_path), exist_ok=True)
            with open(self._halt_flag_path, "w") as f:
                f.write(
                    f"{datetime.now().isoformat()} | {reason}\n"
                    "Remove this file only after investigating the loss.\n"
                )
        except Exception as e:
            self.logger.error(f"Kill switch: could not write halt flag: {e}")

    # ── Broker-Native Resting Stops ──────────────────────────────────────

    def _submit_resting_stop(self, symbol: str):
        """
        Submit a GTC stop order resting at the broker for an open position.

        The stop price is taken from the position the risk manager just opened
        (Chandelier/percentage stop). For a long the stop is a SELL below
        entry; for a short it is a BUY above entry. This is what protects the
        position outside cycle execution (overnight, weekends, downtime).

        Safe in dry-run / when not connected: logs the intended order only and
        never contacts a broker.
        """
        if not getattr(config, "use_native_stops", True):
            return
        pos = self.risk_manager.positions.get(symbol)
        if pos is None:
            return
        stop_price = round(float(pos.stop_loss), 2)
        qty = pos.quantity
        close_side = "buy" if getattr(pos, "is_short", False) else "sell"

        if self.dry_run or not self.alpaca_connected or not self.api:
            self.logger.info(
                f"  {symbol}: [resting stop] would submit GTC stop {close_side} "
                f"{qty} @ ${stop_price:.2f} (protects position outside cycles)"
            )
            return
        try:
            order = self.api.submit_order(
                symbol=symbol, qty=str(qty), side=close_side,
                type="stop", stop_price=str(stop_price), time_in_force="gtc",
            )
            self._resting_stops[symbol] = order.id
            self.logger.info(
                f"  {symbol}: resting GTC stop {close_side} {qty} @ ${stop_price:.2f} "
                f"submitted (ID {order.id})"
            )
        except Exception as e:
            self.logger.error(f"  {symbol}: failed to submit resting stop: {e}")

    def _cancel_resting_stop(self, symbol: str):
        """Cancel a symbol's resting stop order, if one exists."""
        order_id = self._resting_stops.pop(symbol, None)
        if not order_id:
            return
        if self.dry_run or not self.alpaca_connected or not self.api:
            self.logger.info(f"  {symbol}: [resting stop] would cancel order {order_id}")
            return
        try:
            self.api.cancel_order(order_id)
            self.logger.info(f"  {symbol}: cancelled resting stop {order_id}")
        except Exception as e:
            self.logger.warning(f"  {symbol}: failed to cancel resting stop {order_id}: {e}")

    # ── Trading Cycle ────────────────────────────────────────────────────

    def _review_open_positions(self, actions: Dict) -> int:
        """
        Explicit position review — runs every cycle BEFORE the brain decides.

        Checks each open position for:
          - Time-based exit (max holding period for the strategy)
          - Loss exceeding the initial stop distance
          - Thesis reversal on shorts (3+ consecutive brain BUY signals)
          - Regime-change flag for longs (logging only)

        Stop-loss / take-profit triggers are handled separately upstream by
        risk_manager.check_stop_loss_take_profit before this runs.

        Closes are routed through _execute_sell (longs) or _execute_close_short
        (shorts) so they always use the FULL position quantity, sidestepping
        the half-size sentiment_multiplier rounding bug that prevented the
        brain from clearing held positions.

        Returns the number of positions closed.
        """
        self.logger.info(
            f"POSITION REVIEW: checking {len(self.risk_manager.positions)} positions"
        )
        if not self.risk_manager.positions:
            return 0

        open_symbols = list(self.risk_manager.positions.keys())
        try:
            prices = self.data_fetcher.get_latest_prices(open_symbols)
        except Exception as e:
            self.logger.warning(f"POSITION REVIEW: failed to fetch prices: {e}")
            return 0

        closed = 0
        for symbol in open_symbols:
            pos = self.risk_manager.positions.get(symbol)
            if pos is None:
                continue
            current_price = prices.get(symbol) or 0.0
            if current_price <= 0:
                continue

            side = "short" if pos.is_short else "long"
            days_held = (datetime.now() - pos.entry_time).total_seconds() / 86400

            # 1. Time-based exit
            max_days = pos.max_holding_days if pos.max_holding_days and pos.max_holding_days > 0 else None
            if max_days is None:
                try:
                    max_days = self.risk_manager._get_max_holding_days(
                        pos.strategy, pos.half_life_days
                    )
                except Exception:
                    max_days = 30.0
            self.logger.info(
                f"  {symbol} {side}: days_held={days_held:.1f}/{max_days:.0f}d "
                f"strategy={pos.strategy} entry=${pos.entry_price:.2f} "
                f"current=${current_price:.2f}"
            )
            if days_held >= max_days:
                reason = (
                    f"max holding period ({days_held:.1f}d >= {max_days:.0f}d) "
                    f"for strategy={pos.strategy}"
                )
                self.logger.info(
                    f"POSITION REVIEW: Closing {symbol} {side} - {reason}"
                )
                if pos.is_short:
                    self._execute_close_short(
                        symbol, current_price, f"position_review: {reason}", actions,
                    )
                else:
                    self._execute_sell(
                        symbol, current_price, f"position_review: {reason}", actions,
                    )
                closed += 1
                continue

            # 2. Loss exceeds the initial stop distance (cut losers proactively)
            initial_stop_dist = abs(pos.entry_price - pos.stop_loss)
            if initial_stop_dist > 0:
                if pos.is_short:
                    adverse = current_price - pos.entry_price
                    if adverse > initial_stop_dist:
                        reason = (
                            f"short losing - current ${current_price:.2f} above "
                            f"entry ${pos.entry_price:.2f} by ${adverse:.2f} "
                            f"(>initial stop dist ${initial_stop_dist:.2f})"
                        )
                        self.logger.info(
                            f"POSITION REVIEW: Closing {symbol} short - {reason}"
                        )
                        self._execute_close_short(
                            symbol, current_price, f"position_review: {reason}", actions,
                        )
                        closed += 1
                        continue
                else:
                    adverse = pos.entry_price - current_price
                    if adverse > initial_stop_dist:
                        reason = (
                            f"long losing - current ${current_price:.2f} below "
                            f"entry ${pos.entry_price:.2f} by ${adverse:.2f} "
                            f"(>initial stop dist ${initial_stop_dist:.2f})"
                        )
                        self.logger.info(
                            f"POSITION REVIEW: Closing {symbol} long - {reason}"
                        )
                        self._execute_sell(
                            symbol, current_price, f"position_review: {reason}", actions,
                        )
                        closed += 1
                        continue

            # 3. Brain reversal for shorts (3+ consecutive BUY signals)
            if pos.is_short:
                buy_streak = self._brain_buy_streak.get(symbol, 0)
                if buy_streak >= 3:
                    reason = (
                        f"brain reversal ({buy_streak} consecutive BUY signals)"
                    )
                    self.logger.info(
                        f"POSITION REVIEW: Closing {symbol} short - {reason}"
                    )
                    self._execute_close_short(
                        symbol, current_price, f"position_review: {reason}", actions,
                    )
                    self._brain_buy_streak[symbol] = 0
                    closed += 1
                    continue

            # 4. Regime-change flag for longs (logging only — surfaces stale theses)
            if not pos.is_short:
                regime_value = (
                    self.current_regime.value if self.current_regime else "normal"
                )
                if regime_value in ("mean_reverting", "crisis"):
                    self.logger.info(
                        f"POSITION REVIEW FLAG: {symbol} long @ ${pos.entry_price:.2f} "
                        f"(strategy={pos.strategy}, days_held={days_held:.1f}) held "
                        f"during regime={regime_value} — review next cycle"
                    )

        if closed:
            self.logger.info(f"POSITION REVIEW: closed {closed} position(s)")
        return closed

    def run_cycle(self) -> Dict:
        """Execute one full trading cycle."""
        self.cycle_count += 1
        cycle_start = time.time()
        self.logger.info(f"\n{'='*50} LIVE CYCLE {self.cycle_count} {'='*50}")

        actions = {"buys": [], "sells": [], "holds": [], "errors": []}

        # Reset per-cycle attempt list so the daily-log summary reflects
        # only this cycle's blocked / failed trades.
        self._cycle_attempts = []

        # Real-time kill switch — hard circuit breaker on real equity. Checked
        # FIRST so a breach liquidates and halts before any new orders. Once
        # tripped it stays tripped (persistent flag), so this returns early
        # every subsequent cycle until the flag is manually cleared.
        if self._check_kill_switch():
            self.logger.error("Trading halted — kill switch tripped")
            ts = datetime.now().strftime("%H:%M:%S")
            self._write_daily_log(
                f"{ts} | Cycle {self.cycle_count} | KILL SWITCH HALT | "
                f"PnL: ${self.total_pnl:+,.2f}"
            )
            return actions

        # Drawdown check — use the real portfolio value (BUG FIX #1),
        # not the buggy entry-price-based estimate in RiskManager.
        if self._check_real_drawdown():
            self.logger.error("Trading halted — max drawdown breached")
            ts = datetime.now().strftime("%H:%M:%S")
            self._write_daily_log(
                f"{ts} | Cycle {self.cycle_count} | HALTED | "
                f"PnL: ${self.total_pnl:+,.2f} | reason: max drawdown breached"
            )
            return actions

        # Check stop-loss / take-profit on existing positions
        open_symbols = list(self.risk_manager.positions.keys())
        if open_symbols:
            prices = self.data_fetcher.get_latest_prices(open_symbols)

            # Momentum scaling out (feature #5) — before full-close check
            scale_actions = self.risk_manager.check_momentum_scaling(prices)
            for sa in scale_actions:
                sym = sa["symbol"]
                qty = sa["qty_to_sell"]
                stage = sa["scale_stage"]
                pos_before = self.risk_manager.positions.get(sym)
                entry_px = pos_before.entry_price if pos_before else 0.0
                strat = pos_before.strategy if pos_before else "momentum"
                self.logger.info(f"  {sym}: {sa['reason']}")
                success, filled = self._submit_order(sym, qty, "sell")
                if success:
                    pnl_part = (filled - entry_px) * qty
                    self.risk_manager.apply_scale_out(sym, qty, stage)
                    # Resync the resting stop to the reduced share count so it
                    # doesn't try to sell more than we still hold.
                    self._cancel_resting_stop(sym)
                    if sym in self.risk_manager.positions:
                        self._submit_resting_stop(sym)
                    self.total_pnl += pnl_part
                    self._log_trade(
                        symbol=sym, action="SCALE_OUT", quantity=qty,
                        price=filled, filled_price=filled,
                        strategy=strat, signal_strength=0.0,
                        regime=self.current_regime.value, leverage=1.0,
                        portfolio_value=self.risk_manager._portfolio_value_estimate(),
                        pnl=pnl_part, reason=sa["reason"],
                    )

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

        # ── Position review (runs BEFORE brain decides) ──────────────────
        # Closes positions that have hit time/loss/thesis-reversal exits
        # so the brain isn't blocked by "already holding" / "already short"
        # guards on positions that should have been cleared.
        self._review_open_positions(actions)

        # ── Phase 1: Collect all strategy signals ─────────────────────────
        signal_summary = {"buy_signals": 0, "sell_signals": 0, "hold_signals": 0,
                          "skipped_data": 0, "skipped_risk": 0, "threshold_filtered": 0}
        all_signals: Dict[str, Signal] = {}
        all_meta: Dict[str, Dict] = {}
        all_dfs: Dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            try:
                signal, meta, df = self._generate_signals(symbol, active_weights)
                if signal is None:
                    signal_summary["skipped_data"] += 1
                    continue
                all_signals[symbol] = signal
                all_meta[symbol] = meta
                if df is not None:
                    all_dfs[symbol] = df
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
                    # Update consecutive BUY-signal streak per symbol —
                    # consumed by _review_open_positions next cycle to detect
                    # thesis reversal on shorts.
                    cycle_buys = {
                        a.get("symbol") for a in brain_actions
                        if a.get("action") == "BUY"
                    }
                    for sym in self.symbols:
                        if sym in cycle_buys:
                            self._brain_buy_streak[sym] = (
                                self._brain_buy_streak.get(sym, 0) + 1
                            )
                        else:
                            self._brain_buy_streak[sym] = 0
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
                    self._execute_buy(
                        symbol, signal, meta, regime_str, actions,
                        df=all_dfs.get(symbol),
                    )
                elif signal.action == "SELL" and signal.strength < self._sell_threshold:
                    if symbol in self.risk_manager.positions:
                        self._execute_sell(symbol, meta["price"], signal.reason, actions, meta)
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

        # Write the human-readable end-of-cycle summary line to today's
        # daily log. Failures here are swallowed inside _write_daily_log.
        self._write_cycle_summary(
            regime=regime.value, position_count=open_positions_count, actions=actions,
        )

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

            # Apply learned stop-loss / take-profit. risk_manager reads these
            # off the shared `config` singleton (config.stop_loss_pct /
            # config.take_profit_pct) at every compute_stop_take / open_*
            # call, so overriding them here actually takes effect on the next
            # position opened. Previously these were silently dropped: the
            # weekly review wrote e.g. stop_loss_pct=0.04 into the file but the
            # live risk manager kept using config.py's 0.08, so the system
            # reported a tightened stop that never existed. Bounds-checked to
            # match weekly_review's own validation ranges so a corrupt file
            # cannot install an absurd (e.g. 0% or 90%) stop.
            applied_sl = applied_tp = None
            if "stop_loss_pct" in params:
                sl = float(params["stop_loss_pct"])
                if 0.02 <= sl <= 0.20:
                    config.stop_loss_pct = sl
                    applied_sl = sl
                else:
                    self.logger.warning(
                        f"Ignoring out-of-range learned stop_loss_pct={sl} "
                        f"(allowed 0.02–0.20); keeping {config.stop_loss_pct}"
                    )
            if "take_profit_pct" in params:
                tp = float(params["take_profit_pct"])
                if 0.05 <= tp <= 0.40:
                    config.take_profit_pct = tp
                    applied_tp = tp
                else:
                    self.logger.warning(
                        f"Ignoring out-of-range learned take_profit_pct={tp} "
                        f"(allowed 0.05–0.40); keeping {config.take_profit_pct}"
                    )

            updated = params.get("updated_at", "unknown")
            self.logger.info(
                f"Loaded learned params from {path} (updated {updated}) | "
                f"buy_threshold={self._buy_threshold}, sell_threshold={self._sell_threshold} | "
                f"stop_loss_pct={config.stop_loss_pct} "
                f"(learned: {applied_sl}), "
                f"take_profit_pct={config.take_profit_pct} "
                f"(learned: {applied_tp})"
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
          - Holding short  -> close the short (don't also open a long)
          - Holding long   -> skip (duplicate-buy guard)
          - No position    -> open long normally

        SELL logic:
          - Holding long   -> close the long
          - Holding short  -> skip (already short)
          - No position    -> open short if shorting_enabled and confidence > 0.60
        """
        for brain_action in brain_actions:
            symbol = brain_action.get("symbol", "")
            action = brain_action.get("action", "HOLD")
            confidence = max(0.0, min(1.0, float(brain_action.get("confidence", 0.5))))
            rationale = brain_action.get("rationale", "")

            if not symbol or action == "HOLD":
                continue

            # Conviction decay check (feature #3)
            if self._is_conviction_decay(symbol, action, confidence):
                actions["holds"].append(symbol)
                continue
            self._record_recommendation(symbol, action, confidence)

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
                    self._execute_sell(symbol, meta["price"], full_reason, actions, meta)
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
                        + (f"low confidence ({confidence:.0%})" if confidence <= 0.80 else "shorting disabled")
                    )

    def _execute_short_entry(
        self,
        symbol: str,
        confidence: float,
        meta: Dict,
        regime: str,
        reason: str,
        actions: Dict,
        df: Optional[pd.DataFrame] = None,
    ):
        """Open a short position via Alpaca (sell without owning shares)."""
        # Short regime gate (feature #7): only allow shorts in CRISIS or TRENDING_DOWN
        allowed_short_regimes = {
            MarketRegime.CRISIS.value,
            "crisis",
            MarketRegime.TRENDING_DOWN.value,
            "trending_down",
        }
        if regime not in allowed_short_regimes:
            self.logger.info(
                f"  {symbol}: SHORT blocked — regime={regime} not in "
                f"[CRISIS, TRENDING_DOWN] (short_win_rate=12%)"
            )
            self._log_trade_attempt(
                symbol, "SHORT", "BLOCKED", f"regime={regime} not allowed for shorts",
            )
            return

        # Cooldown + entry filter for shorts
        if self._is_in_cooldown(symbol):
            self._log_trade_attempt(symbol, "SHORT", "BLOCKED", "post-close cooldown")
            return

        if df is not None:
            passes, filter_reason = entry_filter.check_entry(df, "short")
            self.logger.info(f"  {symbol} (short): {filter_reason}")
            if not passes:
                self.logger.info(f"  {symbol}: entry filter REJECTED — skip SHORT")
                self._log_trade_attempt(
                    symbol, "SHORT", "BLOCKED", f"entry_filter: {filter_reason}",
                )
                return

        price = meta["price"]
        volatility = meta.get("volatility", 0.02)
        atr = meta.get("atr", 0.0)

        raw_quantity = self.risk_manager.calculate_position_size(
            symbol, price, confidence, volatility, regime
        )
        if raw_quantity <= 0:
            self.logger.info(f"  {symbol}: short size = 0, skipping")
            self._log_trade_attempt(
                symbol, "SHORT", "BLOCKED",
                f"size=0 from calculate_position_size (raw={raw_quantity})",
            )
            return

        # Apply leverage
        equity = self.risk_manager._portfolio_value_estimate()
        leverage = self.leverage_manager.get_leverage(
            current_equity=equity, daily_returns=self._daily_returns
        )
        quantity = max(1, int(raw_quantity * leverage) // 2)  # shorts capped at half long size
        if quantity <= 0:
            self._log_trade_attempt(
                symbol, "SHORT", "BLOCKED",
                f"size=0 after leverage (raw={raw_quantity}, lev={leverage:.2f})",
            )
            return

        # Submit a sell order with no existing position = short entry
        success, filled_price = self._submit_order(symbol, quantity, "sell")
        if not success:
            self.logger.warning(f"  {symbol}: SHORT FAILED — _submit_order returned failure")
            self._log_trade_attempt(
                symbol, "SHORT", "FAILED",
                f"_submit_order failure (qty={quantity})",
            )
            return

        self.risk_manager.open_short_position(symbol, quantity, filled_price, "agent_brain_short", atr)

        # Rest a broker-native buy-stop (above entry) to cap short losses
        # outside cycles.
        self._submit_resting_stop(symbol)

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
        self._log_trade_attempt(
            symbol, "SHORT", "EXECUTED",
            f"qty={quantity} @ ${filled_price:.2f}",
        )
        actions["sells"].append(symbol)

    def _execute_close_short(
        self, symbol: str, price: float, reason: str, actions: Dict,
    ):
        """Close a short position by submitting a BUY order to cover."""
        pos = self.risk_manager.positions.get(symbol)
        if pos is None or not pos.is_short:
            self.logger.info(f"  {symbol}: COVER skipped — no tracked short position")
            self._log_trade_attempt(symbol, "COVER", "BLOCKED", "no short position")
            return

        quantity = pos.quantity
        strategy = pos.strategy
        entry_price = pos.entry_price

        # Cancel the resting broker buy-stop before covering ourselves.
        self._cancel_resting_stop(symbol)

        success, filled_price = self._submit_order(symbol, quantity, "buy")
        if not success:
            err = self._last_order_error.get(symbol, "")
            # Auto-recover from the most common cover failure: local position
            # qty drifts out of sync with Alpaca by a share (partial fill, manual
            # tweak, etc.) and Alpaca rejects with "insufficient qty available".
            # Re-fetch the broker-side qty and retry once with that.
            retried = False
            if "insufficient qty available" in err.lower() and self.alpaca_connected and self.api is not None:
                try:
                    ap = self.api.get_position(symbol)
                    alpaca_qty = abs(int(float(getattr(ap, "qty", 0) or 0)))
                except Exception as e:
                    alpaca_qty = 0
                    self.logger.warning(
                        f"  {symbol}: COVER retry skipped — could not fetch Alpaca qty: {e}"
                    )
                if 0 < alpaca_qty < quantity:
                    self.logger.warning(
                        f"  {symbol}: COVER retry — local qty={quantity} but Alpaca "
                        f"reports {alpaca_qty}; reconciling and retrying"
                    )
                    pos.quantity = alpaca_qty
                    quantity = alpaca_qty
                    success, filled_price = self._submit_order(symbol, quantity, "buy")
                    retried = True
            if not success:
                err = self._last_order_error.get(symbol, err) or "unknown"
                self.logger.warning(
                    f"  {symbol}: COVER FAILED — {err} (qty={quantity}"
                    f"{', retried' if retried else ''})"
                )
                self._log_trade_attempt(
                    symbol, "COVER", "FAILED",
                    f"qty={quantity} — {err}",
                )
                return

        pnl = self.risk_manager.close_position(symbol, filled_price)
        if pnl is not None:
            self.total_pnl += pnl

        # 30-min cooldown after close (feature #3)
        self._position_closed_at[symbol] = datetime.now()
        self._last_recommendation.pop(symbol, None)

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
        self._log_trade_attempt(
            symbol, "COVER", "EXECUTED",
            f"qty={quantity} @ ${filled_price:.2f} | pnl=${(pnl or 0.0):+.2f}",
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
        df: Optional[pd.DataFrame] = None,
    ):
        """
        Execute a buy with Kelly + vol + leverage sizing.

        Every decision point logs (both to the main logger and the daily-log
        trade-attempt feed) so a blocked or failed BUY is never silent.
        """
        self.logger.info(
            f"  {symbol}: _execute_buy ENTER | strength={signal.strength:+.4f} "
            f"strategy={signal.strategy} regime={regime}"
        )

        # 1) Duplicate-buy guard
        if self._already_holding(symbol):
            self.logger.info(f"  {symbol}: BLOCKED — already holding position, skipping BUY")
            self._log_trade_attempt(symbol, "BUY", "BLOCKED", "already holding position")
            actions["holds"].append(symbol)
            return

        # 2) Per-symbol cooldown (feature #3)
        if self._is_in_cooldown(symbol):
            self.logger.info(f"  {symbol}: BLOCKED — post-close cooldown active")
            self._log_trade_attempt(symbol, "BUY", "BLOCKED", "post-close cooldown")
            actions["holds"].append(symbol)
            return

        # 3) Entry timing filter — require 2-of-3 (feature #1)
        if df is not None:
            passes, filter_reason = entry_filter.check_entry(df, "long")
            self.logger.info(f"  {symbol}: entry filter -> {filter_reason}")
            if not passes:
                self.logger.info(f"  {symbol}: BLOCKED — entry filter REJECTED")
                self._log_trade_attempt(
                    symbol, "BUY", "BLOCKED", f"entry_filter: {filter_reason}",
                )
                actions["holds"].append(symbol)
                return
        else:
            self.logger.info(f"  {symbol}: entry filter SKIPPED (no df provided)")

        price = meta["price"]
        atr = meta["atr"]
        volatility = meta["volatility"]

        # 4) Sentiment agreement sizing (feature #10):
        # full size when sentiment + another strategy agree; half size otherwise
        per_strategy = meta.get("per_strategy", {})
        sentiment_action = per_strategy.get("sentiment", {}).get("action", "HOLD")
        other_agree = any(
            per_strategy[s].get("action") == "BUY"
            for s in per_strategy
            if s != "sentiment"
        )
        if sentiment_action == "BUY" and other_agree:
            sentiment_multiplier = 1.0
            self.logger.info(f"  {symbol}: sentiment+other agree -> full size")
        else:
            sentiment_multiplier = 0.5
            self.logger.info(f"  {symbol}: single strategy -> half size")

        # 5) Position sizing
        raw_quantity = self.risk_manager.calculate_position_size(
            symbol, price, signal.strength, volatility, regime
        )
        self.logger.info(
            f"  {symbol}: calculate_position_size -> {raw_quantity} shares "
            f"(price=${price:.2f}, strength={signal.strength:+.4f}, "
            f"vol={volatility:.4f}, regime={regime})"
        )
        if raw_quantity <= 0:
            self.logger.info(
                f"  {symbol}: BLOCKED — calculate_position_size returned "
                f"{raw_quantity} (likely capital/notional cap, halt, or zero Kelly)"
            )
            self._log_trade_attempt(
                symbol, "BUY", "BLOCKED",
                f"size=0 from calculate_position_size (raw={raw_quantity})",
            )
            return

        # 6) Apply leverage + sentiment sizing
        equity = self.risk_manager._portfolio_value_estimate()
        leverage = self.leverage_manager.get_leverage(
            current_equity=equity, daily_returns=self._daily_returns
        )
        multiplier = leverage * sentiment_multiplier
        scaled = raw_quantity * multiplier
        # Round to nearest share (not floor) so half-size positions aren't
        # systematically biased down. The previous "if it truncated to 0, make
        # it 1" hack is REMOVED: forcing a 1-share trade placed unintended
        # positions the sizing logic had decided against. If sizing genuinely
        # rounds to 0, we skip the trade cleanly below.
        quantity = int(round(scaled))
        self.logger.info(
            f"  {symbol}: post-leverage sizing -> {quantity} shares "
            f"(raw={raw_quantity} * leverage={leverage:.2f} "
            f"* sentiment_mult={sentiment_multiplier:.2f})"
        )
        if quantity <= 0:
            self.logger.info(
                f"  {symbol}: BLOCKED — post-leverage quantity={quantity} "
                f"(raw={raw_quantity}, leverage={leverage:.2f}, "
                f"sentiment_mult={sentiment_multiplier:.2f})"
            )
            self._log_trade_attempt(
                symbol, "BUY", "BLOCKED",
                f"size=0 after leverage*sentiment "
                f"(raw={raw_quantity}, lev={leverage:.2f}, sent={sentiment_multiplier:.2f})",
            )
            return

        # 7) Log entry fingerprint before execution (feature #9a)
        self._log_entry_fingerprint(symbol, "long", signal.strategy, signal.strength, meta, df)

        # 8) Submit order
        self.logger.info(
            f"  {symbol}: submitting BUY order for {quantity} shares "
            f"(notional ~${quantity * price:,.2f})"
        )
        success, filled_price = self._submit_order(symbol, quantity, "buy")
        if not success:
            self.logger.warning(
                f"  {symbol}: BLOCKED — _submit_order returned failure "
                f"(check earlier order/limit/cancel logs above)"
            )
            self._log_trade_attempt(
                symbol, "BUY", "FAILED",
                f"_submit_order failure (qty={quantity}) — see order logs",
            )
            return

        # 9) Success path
        self.risk_manager.open_position(symbol, quantity, filled_price, signal.strategy, atr)

        # Rest a broker-native stop so the position is protected outside cycles.
        self._submit_resting_stop(symbol)

        # Structured trade journal (consumed by daily_review.py)
        self.journal.log_entry(
            symbol=symbol, entry_price=filled_price, strategy=signal.strategy,
            regime=regime, meta=meta,
        )

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
        self.logger.info(
            f"  {symbol}: BUY EXECUTED — {quantity} shares @ ${filled_price:.2f}"
        )
        self._log_trade_attempt(
            symbol, "BUY", "EXECUTED",
            f"qty={quantity} @ ${filled_price:.2f} via {signal.strategy}",
        )
        actions["buys"].append(symbol)

    def _execute_sell(
        self, symbol: str, price: float, reason: str, actions: Dict,
        meta: Optional[Dict] = None,
    ):
        """Execute a sell (close position)."""
        if symbol not in self.risk_manager.positions:
            self.logger.info(f"  {symbol}: SELL skipped — no tracked position")
            self._log_trade_attempt(symbol, "SELL", "BLOCKED", "no tracked position")
            return

        pos = self.risk_manager.positions[symbol]
        quantity = pos.quantity
        strategy = pos.strategy
        entry_price = pos.entry_price

        # Cancel the resting broker stop first so it can't fire after we cover
        # the position ourselves (avoids an orphan order / double exit).
        self._cancel_resting_stop(symbol)

        success, filled_price = self._submit_order(symbol, quantity, "sell")
        if not success:
            self.logger.warning(f"  {symbol}: SELL FAILED — _submit_order returned failure")
            self._log_trade_attempt(
                symbol, "SELL", "FAILED",
                f"_submit_order failure (qty={quantity})",
            )
            return

        pnl = self.risk_manager.close_position(symbol, filled_price)
        if pnl is not None:
            self.total_pnl += pnl

        # 30-min cooldown after close (feature #3)
        self._position_closed_at[symbol] = datetime.now()
        # Reset conviction decay so next signal is fresh
        self._last_recommendation.pop(symbol, None)

        holding_hours = (datetime.now() - pos.entry_time).total_seconds() / 3600
        self.self_improver.record_experience(
            symbol=symbol, strategy=strategy, action="SELL",
            signal_strength=0.0, entry_price=entry_price,
            exit_price=filled_price,
            holding_period_hours=holding_hours,
            market_regime=self.current_regime.value,
        )

        # Counterfactual attribution log (feature #9b)
        if meta:
            self._log_exit_counterfactual(
                symbol=symbol, entry_price=entry_price, exit_price=filled_price,
                strategy=strategy, pnl=pnl or 0.0, meta=meta,
            )

        # Structured trade journal exit record (consumed by daily_review.py)
        self.journal.log_exit(
            symbol=symbol, entry_price=entry_price, exit_price=filled_price,
            strategy=strategy, holding_days=holding_hours / 24,
            exit_reason=(reason or "signal")[:80],
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
        self._log_trade_attempt(
            symbol, "SELL", "EXECUTED",
            f"qty={quantity} @ ${filled_price:.2f} | pnl=${(pnl or 0.0):+.2f}",
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
        """Graceful shutdown: log final state and run daily review."""
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
        self._run_daily_review()
        self.logger.info("State saved. Goodbye.")
        self.logger.info("=" * 60)

    def _run_daily_review(self):
        """Run end-of-day DailyReview and save flags to data/daily_reviews/."""
        try:
            positions_data = {
                sym: {
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": pos.entry_price,
                    "strategy": pos.strategy,
                    "entry_time": pos.entry_time.isoformat(),
                }
                for sym, pos in self.risk_manager.positions.items()
            }
            atrs = {
                sym: pos.entry_atr
                for sym, pos in self.risk_manager.positions.items()
                if getattr(pos, "entry_atr", 0) > 0
            }
            journal_entries = self.journal.get_recent_trades(days=7)
            reviewer = DailyReview()
            result = reviewer.run(
                positions=positions_data,
                regime=self.current_regime.value,
                atrs=atrs,
                journal_entries=journal_entries,
            )
            if result:
                n = len(result.get("ai_flags", [])) + len(result.get("local_flags", []))
                self.logger.info(f"Daily review complete: {n} flag(s) saved")
        except Exception as e:
            self.logger.warning(f"Daily review failed: {e}")

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
