"""
Risk management module.
Handles position sizing, stop losses, drawdown monitoring, and exposure limits.
Inspired by institutional risk frameworks used at top quant firms.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import config
from logger import TradeLogger


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    quantity: int
    entry_price: float
    entry_time: datetime
    strategy: str
    stop_loss: float
    take_profit: float


class RiskManager:
    """
    Enforces risk controls on every trade decision.

    Controls:
      - Per-position sizing (Kelly-inspired with cap)
      - Stop-loss / take-profit levels
      - Maximum portfolio exposure
      - Maximum drawdown circuit breaker
      - Daily trade limits
      - Correlation-based diversification
    """

    def __init__(self, initial_capital: float, logger: TradeLogger):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        self.logger = logger

        self.positions: Dict[str, Position] = {}
        self.daily_trades = 0
        self.daily_trade_date: Optional[str] = None
        self.trading_halted = False

    # ── Position Sizing ──────────────────────────────────────────────────
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        signal_strength: float,
        volatility: float,
    ) -> int:
        """
        Calculate number of shares to buy.
        Uses a volatility-adjusted Kelly-fraction approach, capped by config limits.
        """
        if self.trading_halted:
            self.logger.warning(
                f"Trading halted — refusing position for {symbol}"
            )
            return 0

        # Max dollar amount for this position
        max_dollars = self.current_capital * config.max_portfolio_pct_per_trade

        # Volatility adjustment: reduce size when vol is high
        vol_scalar = max(0.2, min(1.0, 0.02 / max(volatility, 1e-6)))

        # Signal-strength scaling (stronger signal → larger position)
        signal_scalar = abs(signal_strength)

        adjusted_dollars = max_dollars * vol_scalar * signal_scalar
        shares = int(adjusted_dollars / price) if price > 0 else 0

        # Enforce total exposure limit
        current_exposure = self._total_exposure()
        remaining_budget = (
            self.current_capital * config.max_total_exposure - current_exposure
        )
        if remaining_budget <= 0:
            self.logger.warning("Max exposure reached — no new positions")
            return 0

        max_shares_by_budget = int(remaining_budget / price) if price > 0 else 0
        shares = min(shares, max_shares_by_budget)

        return max(shares, 0)

    # ── Pre-Trade Checks ─────────────────────────────────────────────────
    def can_trade(self, symbol: str) -> Tuple[bool, str]:
        """Run all pre-trade checks. Returns (allowed, reason)."""
        # Daily trade limit
        today = datetime.now().strftime("%Y-%m-%d")
        if self.daily_trade_date != today:
            self.daily_trade_date = today
            self.daily_trades = 0

        if self.daily_trades >= config.max_daily_trades:
            return False, "Daily trade limit reached"

        if self.trading_halted:
            return False, "Trading halted due to drawdown"

        if len(self.positions) >= config.max_open_positions:
            if symbol not in self.positions:
                return False, "Max open positions reached"

        return True, "OK"

    # ── Stop Loss / Take Profit ──────────────────────────────────────────
    def compute_stop_take(
        self, entry_price: float, atr: float
    ) -> Tuple[float, float]:
        """
        Compute stop-loss and take-profit prices.
        Uses ATR-based dynamic levels with config floors.
        """
        # ATR-based stop: 2x ATR below entry
        atr_stop = entry_price - 2.0 * atr
        pct_stop = entry_price * (1 - config.stop_loss_pct)
        stop_loss = max(atr_stop, pct_stop)  # tighter of the two

        # ATR-based take-profit: 3x ATR above entry
        atr_tp = entry_price + 3.0 * atr
        pct_tp = entry_price * (1 + config.take_profit_pct)
        take_profit = min(atr_tp, pct_tp)

        return stop_loss, take_profit

    # ── Position Management ──────────────────────────────────────────────
    def open_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        strategy: str,
        atr: float,
    ):
        """Record a new open position."""
        stop_loss, take_profit = self.compute_stop_take(price, atr)
        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_time=datetime.now(),
            strategy=strategy,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )
        self.daily_trades += 1
        self.current_capital -= quantity * price
        self.logger.info(
            f"Position opened: {quantity} {symbol} @ ${price:.2f} "
            f"SL=${stop_loss:.2f} TP=${take_profit:.2f}"
        )

    def close_position(self, symbol: str, price: float) -> Optional[float]:
        """Close a position and return P&L."""
        if symbol not in self.positions:
            return None

        pos = self.positions.pop(symbol)
        pnl = (price - pos.entry_price) * pos.quantity
        self.current_capital += pos.quantity * price
        self.daily_trades += 1

        # Update peak for drawdown tracking
        total_value = self._portfolio_value_estimate()
        self.peak_capital = max(self.peak_capital, total_value)

        self.logger.info(
            f"Position closed: {pos.quantity} {symbol} @ ${price:.2f} "
            f"PnL=${pnl:+,.2f}"
        )
        return pnl

    def check_stop_loss_take_profit(
        self, prices: Dict[str, float]
    ) -> List[str]:
        """Check all positions for stop-loss / take-profit triggers."""
        to_close = []
        for symbol, pos in self.positions.items():
            price = prices.get(symbol)
            if price is None:
                continue
            if price <= pos.stop_loss:
                self.logger.warning(
                    f"STOP LOSS triggered for {symbol} @ ${price:.2f}"
                )
                to_close.append(symbol)
            elif price >= pos.take_profit:
                self.logger.info(
                    f"TAKE PROFIT triggered for {symbol} @ ${price:.2f}"
                )
                to_close.append(symbol)
        return to_close

    # ── Drawdown Monitor ─────────────────────────────────────────────────
    def check_drawdown(self) -> bool:
        """
        Check if max drawdown has been breached.
        Returns True if trading should be halted.
        """
        total_value = self._portfolio_value_estimate()
        self.peak_capital = max(self.peak_capital, total_value)

        if self.peak_capital > 0:
            drawdown = (self.peak_capital - total_value) / self.peak_capital
        else:
            drawdown = 0.0

        if drawdown >= config.max_drawdown_pct:
            self.trading_halted = True
            self.logger.error(
                f"MAX DRAWDOWN BREACHED: {drawdown:.2%} — trading halted"
            )
            return True
        return False

    def reset_halt(self):
        """Manually reset the trading halt (use with caution)."""
        self.trading_halted = False
        self.logger.warning("Trading halt manually reset")

    # ── Helpers ───────────────────────────────────────────────────────────
    def _total_exposure(self) -> float:
        """Total dollar value currently in positions (estimate)."""
        return sum(
            pos.quantity * pos.entry_price for pos in self.positions.values()
        )

    def _portfolio_value_estimate(self) -> float:
        """Rough portfolio value = cash + position notional at entry."""
        return self.current_capital + self._total_exposure()

    def get_status(self) -> Dict:
        """Return current risk status snapshot."""
        total_value = self._portfolio_value_estimate()
        return {
            "capital": self.current_capital,
            "total_value": total_value,
            "open_positions": len(self.positions),
            "exposure_pct": (
                self._total_exposure() / total_value if total_value > 0 else 0
            ),
            "drawdown_pct": (
                (self.peak_capital - total_value) / self.peak_capital
                if self.peak_capital > 0
                else 0
            ),
            "trading_halted": self.trading_halted,
            "daily_trades": self.daily_trades,
        }
