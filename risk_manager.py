"""
Risk Management Module — Redesigned with Kelly Criterion and Volatility Targeting.

Key improvements over v1:
  1. Half-Kelly criterion for position sizing (replaces fixed fractional)
  2. Volatility targeting: scale positions so portfolio vol stays near target
  3. Wider stop-losses (6% default vs 3%) to avoid premature exits
  4. ATR-based trailing stops (3x ATR) that adapt to market conditions
  5. Regime-aware risk scaling (reduce exposure in crisis regimes)
  6. Maximum position concentration limits

Research basis:
  - Kelly (1956): Optimal growth criterion for sequential investment
  - Moreira & Muir (2017): Volatility-managed portfolios earn higher Sharpe
  - AQR: Fractional Kelly (50%) captures ~75% of optimal growth with ~50% less drawdown
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
    trailing_stop: Optional[float] = None
    highest_price: Optional[float] = None
    is_short: bool = False

    # ATR-based Chandelier stop state (feature #4)
    entry_atr: float = 0.0                # ATR at time of entry
    chandelier_trail_mult: float = 2.0    # ATR multiple to trail below peak
    chandelier_activation_atr: float = 1.0  # ATR gain needed to activate trailing

    # Momentum scaling state (feature #5)
    original_quantity: int = 0            # quantity at entry (before any scale-outs)
    scaled_40_pct: bool = False           # sold 40% at 1:1
    scaled_30_pct: bool = False           # sold 30% at 2:1

    # Time-based exit (feature #6)
    max_holding_days: float = 30.0        # auto-close after this many days
    half_life_days: Optional[float] = None  # used by mean_reversion strategy


class RiskManager:
    """
    Risk management with Kelly criterion and volatility targeting.

    Position Sizing Pipeline:
      1. Compute Kelly fraction from historical win rate and payoff ratio
      2. Apply half-Kelly (conservative)
      3. Scale by inverse realized volatility (vol targeting)
      4. Cap by per-position and total exposure limits
      5. Reduce further in crisis regime

    Stop-Loss System:
      - Initial stop: wider of (ATR-based, percentage-based)
      - Trailing stop: tracks 3x ATR below highest price since entry
      - Time stop: exit if position held > 2x expected half-life
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

        # Kelly tracking
        self._trade_results: List[float] = []  # List of % returns per trade
        self._kelly_fraction: float = config.kelly_fraction

    # -- Position Sizing (Kelly + Vol Targeting) --

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        signal_strength: float,
        volatility: float,
        regime: str = "normal",
    ) -> int:
        """
        Calculate number of shares using half-Kelly criterion + vol targeting.

        Pipeline:
          1. Base allocation from Kelly fraction
          2. Scale by signal strength
          3. Scale by inverse vol (vol targeting)
          4. Reduce in crisis regime
          5. Cap by position limits and total exposure
        """
        if self.trading_halted or price <= 0:
            return 0

        # Step 1: Kelly-based base allocation
        if config.use_kelly and len(self._trade_results) >= config.kelly_min_trades:
            kelly_pct = self._compute_kelly_fraction()
            base_pct = kelly_pct * config.kelly_fraction  # Half-Kelly
        else:
            base_pct = config.max_portfolio_pct_per_trade

        # Step 2: Signal strength scaling
        signal_scalar = np.clip(abs(signal_strength), 0.2, 1.0)
        adjusted_pct = base_pct * signal_scalar

        # Step 3: Volatility targeting
        vol_scalar = self._vol_target_scalar(volatility)
        adjusted_pct *= vol_scalar

        # Step 4: Regime adjustment
        if regime == "crisis":
            adjusted_pct *= 0.4   # 60% reduction in crisis
        elif regime == "high_vol":
            adjusted_pct *= 0.6   # 40% reduction in high vol

        # Step 5: Dollar amount and share count
        max_dollars = self.current_capital * min(adjusted_pct, config.max_portfolio_pct_per_trade)
        shares = int(max_dollars / price)

        # Enforce total exposure limit
        current_exposure = self._total_exposure()
        remaining_budget = (
            self.current_capital * config.max_total_exposure - current_exposure
        )
        if remaining_budget <= 0:
            return 0

        max_shares_by_budget = int(remaining_budget / price)
        shares = min(shares, max_shares_by_budget)

        # Step 6: Sector concentration cap — don't let one correlated factor
        # (e.g. tech-beta) exceed max_sector_exposure of capital, even if the
        # gross-exposure budget would otherwise allow it.
        sector = self._sector_of(symbol)
        sector_budget = (
            self.current_capital * getattr(config, "max_sector_exposure", 1.0)
            - self._sector_exposure(sector)
        )
        if sector_budget <= 0:
            return 0
        max_shares_by_sector = int(sector_budget / price)
        shares = min(shares, max_shares_by_sector)

        return max(shares, 0)

    def _sector_of(self, symbol: str) -> str:
        """Map a symbol to its sector bucket (see config.sector_map)."""
        return getattr(config, "sector_map", {}).get(symbol, "other")

    def _sector_exposure(self, sector: str) -> float:
        """Gross LONG dollar exposure currently held in a sector (at entry).

        Concentration risk is about correlated longs piling into one factor, so
        shorts (hedges) are not counted toward the cap.
        """
        return sum(
            pos.quantity * pos.entry_price
            for sym, pos in self.positions.items()
            if not pos.is_short and self._sector_of(sym) == sector
        )

    def _compute_kelly_fraction(self) -> float:
        """
        Compute Kelly fraction from historical trade results.

        Kelly% = W - (1-W)/R
        where W = win rate, R = avg_win / avg_loss

        Returns the raw Kelly fraction (before applying the config.kelly_fraction
        multiplier for half/quarter Kelly).
        """
        results = self._trade_results[-config.kelly_lookback:]
        if len(results) < config.kelly_min_trades:
            return config.max_portfolio_pct_per_trade

        wins = [r for r in results if r > 0]
        losses = [r for r in results if r < 0]

        if not wins or not losses:
            return config.max_portfolio_pct_per_trade

        win_rate = len(wins) / len(results)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        if avg_loss < 1e-8:
            return config.max_portfolio_pct_per_trade

        payoff_ratio = avg_win / avg_loss
        kelly = win_rate - (1 - win_rate) / payoff_ratio

        # Clamp: Kelly can be negative (don't trade) or very large
        kelly = np.clip(kelly, 0.0, 0.25)  # Max 25% raw Kelly

        return kelly

    def _vol_target_scalar(self, current_vol: float) -> float:
        """
        Compute position scaling factor for volatility targeting.

        scalar = target_vol / realized_vol
        Capped by config.vol_scale_min and config.vol_scale_max.
        """
        if current_vol < 0.001:
            return 1.0

        annualized_vol = current_vol * np.sqrt(252)
        scalar = config.vol_target / max(annualized_vol, 0.01)
        return np.clip(scalar, config.vol_scale_min, config.vol_scale_max)

    def record_trade_result(self, pct_return: float):
        """Record a completed trade's percentage return for Kelly estimation."""
        self._trade_results.append(pct_return)
        # Keep only the last N results
        if len(self._trade_results) > config.kelly_lookback * 2:
            self._trade_results = self._trade_results[-config.kelly_lookback:]

    # -- Pre-Trade Checks --

    def can_trade(self, symbol: str) -> Tuple[bool, str]:
        """Run all pre-trade checks."""
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

    # -- Chandelier Stop Parameters by Strategy (feature #4) --

    _CHANDELIER_PARAMS = {
        # (initial_stop_mult, activation_atr_mult, trail_mult)
        "momentum":          (2.0, 1.0, 2.0),
        "mean_reversion":    (1.5, 0.5, 1.0),
        "sentiment":         (2.0, 1.0, 1.5),
        "pattern_recognition": (2.5, 1.5, 2.0),
        # Fallback for agent_brain / combined / unknown
        "default":           (2.0, 1.0, 2.0),
    }

    # Max holding days by strategy (feature #6)
    _MAX_HOLDING_DAYS = {
        "momentum":          15.0,
        "mean_reversion":    None,   # uses 2x half-life, min 3 days
        "sentiment":         20.0,
        "pattern_recognition": 30.0,
        "default":           30.0,
    }

    def _get_chandelier_params(self, strategy: str) -> Tuple[float, float, float]:
        """Return (initial_mult, activation_mult, trail_mult) for a strategy."""
        # Strip combined(...) wrapper and agent_brain prefix
        base = strategy.split("(")[0].lower()
        if "momentum" in base:
            return self._CHANDELIER_PARAMS["momentum"]
        if "mean_reversion" in base or "mean-reversion" in base:
            return self._CHANDELIER_PARAMS["mean_reversion"]
        if "sentiment" in base:
            return self._CHANDELIER_PARAMS["sentiment"]
        if "pattern" in base:
            return self._CHANDELIER_PARAMS["pattern_recognition"]
        return self._CHANDELIER_PARAMS["default"]

    def _get_max_holding_days(self, strategy: str, half_life_days: Optional[float]) -> float:
        """Return maximum holding period in days for a strategy."""
        base = strategy.split("(")[0].lower()
        if "mean_reversion" in base or "mean-reversion" in base:
            if half_life_days and half_life_days > 0:
                return max(2.0 * half_life_days, 3.0)
            return 10.0  # default for mean reversion when half-life unknown
        if "momentum" in base:
            return self._MAX_HOLDING_DAYS["momentum"]
        if "sentiment" in base:
            return self._MAX_HOLDING_DAYS["sentiment"]
        if "pattern" in base:
            return self._MAX_HOLDING_DAYS["pattern_recognition"]
        return self._MAX_HOLDING_DAYS["default"]

    # -- Stop Loss / Take Profit --

    def compute_stop_take(
        self, entry_price: float, atr: float, strategy: str = "default"
    ) -> Tuple[float, float]:
        """
        Compute initial stop-loss and take-profit using ATR-based Chandelier levels.

        Initial stop distance = initial_stop_mult * ATR (strategy-specific).
        Falls back to percentage-based levels if ATR is tiny.
        """
        init_mult, _, _ = self._get_chandelier_params(strategy)

        if atr > 0:
            atr_stop = entry_price - init_mult * atr
            # Take-profit targets 2x the initial risk
            atr_tp = entry_price + 2.0 * init_mult * atr
        else:
            atr_stop = entry_price * (1 - config.stop_loss_pct)
            atr_tp = entry_price * (1 + config.take_profit_pct)

        # Percentage-based bounds
        pct_stop = entry_price * (1 - config.stop_loss_pct)
        pct_tp = entry_price * (1 + config.take_profit_pct)

        stop_loss = min(atr_stop, pct_stop)    # WIDER stop
        take_profit = min(atr_tp, pct_tp)       # SMALLER take-profit

        return stop_loss, take_profit

    # -- Position Management --

    def open_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        strategy: str,
        atr: float,
        half_life_days: Optional[float] = None,
    ):
        """Record a new open position with Chandelier trailing stop initialization."""
        stop_loss, take_profit = self.compute_stop_take(price, atr, strategy)
        _, activation_mult, trail_mult = self._get_chandelier_params(strategy)
        max_days = self._get_max_holding_days(strategy, half_life_days)

        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_time=datetime.now(),
            strategy=strategy,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=stop_loss,
            highest_price=price,
            entry_atr=atr,
            chandelier_trail_mult=trail_mult,
            chandelier_activation_atr=activation_mult,
            original_quantity=quantity,
            max_holding_days=max_days,
            half_life_days=half_life_days,
        )
        self.daily_trades += 1
        self.current_capital -= quantity * price
        self.logger.info(
            f"Position opened: {quantity} {symbol} @ ${price:.2f} "
            f"SL=${stop_loss:.2f} TP=${take_profit:.2f} "
            f"trail={trail_mult}xATR max_days={max_days:.0f}"
        )

    def open_short_position(
        self,
        symbol: str,
        quantity: int,
        price: float,
        strategy: str,
        atr: float,
    ):
        """Record a new short position. Stop is above entry; TP is below entry."""
        stop_loss = price * (1 + config.stop_loss_pct)    # e.g. entry * 1.08
        take_profit = price * (1 - config.take_profit_pct)  # e.g. entry * 0.80

        # Widen stop if ATR-based distance is larger
        if atr > 0:
            atr_stop = price + 3.0 * atr
            stop_loss = max(stop_loss, atr_stop)

        self.positions[symbol] = Position(
            symbol=symbol,
            quantity=quantity,
            entry_price=price,
            entry_time=datetime.now(),
            strategy=strategy,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=None,
            highest_price=None,
            is_short=True,
        )
        self.daily_trades += 1
        self.current_capital += quantity * price  # receive proceeds from short sale
        self.logger.info(
            f"Short opened: {quantity} {symbol} @ ${price:.2f} "
            f"SL=${stop_loss:.2f} TP=${take_profit:.2f}"
        )

    def close_position(self, symbol: str, price: float) -> Optional[float]:
        """Close a position (long or short), record result, return P&L."""
        if symbol not in self.positions:
            return None

        pos = self.positions.pop(symbol)
        if pos.is_short:
            pnl = (pos.entry_price - price) * pos.quantity
            pct_return = (pos.entry_price - price) / pos.entry_price
            self.current_capital -= pos.quantity * price  # pay to cover short
        else:
            pnl = (price - pos.entry_price) * pos.quantity
            pct_return = (price - pos.entry_price) / pos.entry_price
            self.current_capital += pos.quantity * price
        self.daily_trades += 1

        # Record for Kelly estimation
        self.record_trade_result(pct_return)

        # Update peak for drawdown tracking
        total_value = self._portfolio_value_estimate()
        self.peak_capital = max(self.peak_capital, total_value)

        self.logger.info(
            f"Position closed: {pos.quantity} {symbol} @ ${price:.2f} "
            f"PnL=${pnl:+,.2f} ({pct_return:+.2%})"
        )
        return pnl

    def check_stop_loss_take_profit(
        self, prices: Dict[str, float]
    ) -> List[str]:
        """
        Check all positions for stop-loss, take-profit, Chandelier trailing
        stops, and time-based exits.

        Longs: ATR-based Chandelier trailing stop (feature #4).
          - Activates once gain >= chandelier_activation_atr * entry_atr.
          - Trails chandelier_trail_mult * entry_atr below peak price.

        Time-based exits (feature #6): close positions older than max_holding_days.
        """
        to_close = []
        for symbol, pos in self.positions.items():
            price = prices.get(symbol)
            if price is None:
                continue

            # Time-based exit (longs and shorts)
            days_held = (datetime.now() - pos.entry_time).total_seconds() / 86400
            if days_held > pos.max_holding_days:
                self.logger.info(
                    f"TIME_EXIT: {symbol} held {days_held:.1f}d > "
                    f"{pos.max_holding_days:.0f}d [{pos.strategy}]"
                )
                to_close.append(symbol)
                continue

            # Short position
            if pos.is_short:
                if price >= pos.stop_loss:
                    self.logger.warning(
                        f"SHORT STOP triggered for {symbol} @ ${price:.2f} "
                        f"(stop=${pos.stop_loss:.2f})"
                    )
                    to_close.append(symbol)
                elif price <= pos.take_profit:
                    self.logger.info(
                        f"SHORT TAKE PROFIT triggered for {symbol} @ ${price:.2f}"
                    )
                    to_close.append(symbol)
                continue

            # Long: update peak
            if pos.highest_price is not None and price > pos.highest_price:
                pos.highest_price = price

            peak = pos.highest_price or pos.entry_price
            gain = peak - pos.entry_price
            activation_threshold = pos.chandelier_activation_atr * pos.entry_atr

            # Activate Chandelier trailing stop once gain crosses activation threshold
            if pos.entry_atr > 0 and gain >= activation_threshold:
                chandelier_stop = peak - pos.chandelier_trail_mult * pos.entry_atr
                pos.trailing_stop = max(pos.trailing_stop or pos.stop_loss, chandelier_stop)

            effective_stop = max(pos.stop_loss, pos.trailing_stop or 0)

            if price <= effective_stop:
                self.logger.warning(
                    f"STOP triggered for {symbol} @ ${price:.2f} "
                    f"(chandelier=${effective_stop:.2f}, peak=${peak:.2f})"
                )
                to_close.append(symbol)
            elif price >= pos.take_profit:
                self.logger.info(
                    f"TAKE PROFIT triggered for {symbol} @ ${price:.2f}"
                )
                to_close.append(symbol)

        return to_close

    # -- Momentum Scaling Out (feature #5) ---

    def check_momentum_scaling(self, prices: Dict[str, float]) -> List[Dict]:
        """
        For momentum positions return partial-sell actions at 1:1 and 2:1 RR.
          40% at 1:1 risk-reward
          30% at 2:1 risk-reward
          Remaining 30% trailed by Chandelier (handled by check_stop_loss_take_profit)

        Returns list of {symbol, qty_to_sell, reason, scale_stage}.
        """
        scale_actions = []
        for symbol, pos in self.positions.items():
            price = prices.get(symbol)
            if price is None or pos.is_short:
                continue
            if "momentum" not in pos.strategy.lower():
                continue

            initial_risk = pos.entry_price - pos.stop_loss
            if initial_risk <= 0:
                continue

            orig_qty = pos.original_quantity or pos.quantity

            # 40% at 1:1
            if not pos.scaled_40_pct and price >= pos.entry_price + initial_risk:
                qty = max(1, int(orig_qty * 0.40))
                qty = min(qty, pos.quantity)
                if qty > 0:
                    scale_actions.append({
                        "symbol": symbol,
                        "qty_to_sell": qty,
                        "reason": f"momentum_scale_40pct@1:1RR price=${price:.2f}",
                        "scale_stage": "40pct",
                    })

            # 30% at 2:1 (only after 40% done)
            elif pos.scaled_40_pct and not pos.scaled_30_pct and \
                    price >= pos.entry_price + 2.0 * initial_risk:
                qty = max(1, int(orig_qty * 0.30))
                qty = min(qty, pos.quantity)
                if qty > 0:
                    scale_actions.append({
                        "symbol": symbol,
                        "qty_to_sell": qty,
                        "reason": f"momentum_scale_30pct@2:1RR price=${price:.2f}",
                        "scale_stage": "30pct",
                    })

        return scale_actions

    def apply_scale_out(self, symbol: str, qty_sold: int, stage: str):
        """Update position record after a partial momentum scale-out."""
        pos = self.positions.get(symbol)
        if pos is None:
            return
        pos.quantity = max(0, pos.quantity - qty_sold)
        if stage == "40pct":
            pos.scaled_40_pct = True
        elif stage == "30pct":
            pos.scaled_30_pct = True
        self.logger.info(
            f"Scale-out: {symbol} remaining_qty={pos.quantity} stage={stage}"
        )

    # -- Drawdown Monitor --

    def check_drawdown(self) -> bool:
        """Check if max drawdown has been breached."""
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
        """Manually reset the trading halt."""
        self.trading_halted = False
        self.logger.warning("Trading halt manually reset")

    # -- Helpers --

    def _total_exposure(self) -> float:
        """Total dollar value in positions."""
        return sum(
            pos.quantity * pos.entry_price for pos in self.positions.values()
        )

    def _portfolio_value_estimate(self) -> float:
        """Rough portfolio value = cash + long notional - short notional (at entry).

        current_capital already includes short-sale proceeds, so short notional
        must be subtracted to avoid double-counting. Without this, every open
        short inflates the estimate by 2x its notional value, which corrupts
        peak_capital and causes false drawdown halts.
        """
        long_val = sum(
            pos.quantity * pos.entry_price
            for pos in self.positions.values()
            if not pos.is_short
        )
        short_val = sum(
            pos.quantity * pos.entry_price
            for pos in self.positions.values()
            if pos.is_short
        )
        return self.current_capital + long_val - short_val

    def get_kelly_stats(self) -> Dict:
        """Return current Kelly criterion statistics."""
        results = self._trade_results[-config.kelly_lookback:]
        if len(results) < 5:
            return {"win_rate": 0, "payoff_ratio": 0, "kelly_pct": 0, "trades": 0}

        wins = [r for r in results if r > 0]
        losses = [r for r in results if r < 0]
        win_rate = len(wins) / len(results) if results else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 1
        payoff = avg_win / max(avg_loss, 1e-8)
        kelly = win_rate - (1 - win_rate) / max(payoff, 1e-8)

        return {
            "win_rate": win_rate,
            "payoff_ratio": payoff,
            "kelly_pct": kelly,
            "half_kelly_pct": kelly * config.kelly_fraction,
            "trades": len(results),
        }

    def get_status(self) -> Dict:
        """Return current risk status snapshot."""
        total_value = self._portfolio_value_estimate()
        kelly_stats = self.get_kelly_stats()
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
            "kelly": kelly_stats,
        }
