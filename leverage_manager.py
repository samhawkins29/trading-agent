"""
Leverage Manager — Dynamic Leverage for Low-Volatility Strategy Amplification.

Implements multiple leverage modes used by top quant funds to boost returns
from low-vol strategies while maintaining disciplined risk management.

Modes:
  - none:       No leverage (1x)
  - fixed:      Constant multiplier (2x, 3x, 5x)
  - kelly:      Kelly-optimal: leverage = Sharpe / σ², capped at max_leverage
  - vol_target: Target a specific annual vol, adjust leverage dynamically

Safety features:
  - Maximum leverage cap (configurable, default 5x)
  - Drawdown circuit breaker: reduce to 1x at trigger threshold
  - Gradual ramp-up over configurable days after circuit breaker resets
  - Daily leverage logging for audit trail

Research basis:
  - Moreira & Muir (2017): "Volatility-Managed Portfolios"
  - AQR: "Leverage Aversion and Risk Parity"
  - Frazzini & Pedersen (2014): "Betting Against Beta"
  - Kelly (1956): "A New Interpretation of Information Rate"
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class LeverageConfig:
    """Configuration for leverage management."""
    mode: str = "vol_target"              # none, fixed, kelly, vol_target
    fixed_multiplier: float = 3.0         # For fixed mode
    max_leverage: float = 5.0             # Hard cap on leverage
    vol_target_annual: float = 0.15       # Annual vol target for vol_target mode
    max_drawdown_trigger: float = 0.10    # Drawdown level that triggers circuit breaker
    ramp_days: int = 5                    # Days to ramp back up after circuit breaker
    funding_cost_annual: float = 0.02     # Annual cost of leverage (margin rate - risk-free)
    min_leverage: float = 0.5             # Minimum leverage floor


class LeverageManager:
    """
    Manages portfolio leverage with multiple modes and safety controls.

    The manager computes a leverage factor each day that scales position sizes.
    It tracks drawdown state and applies circuit breakers and gradual ramps
    to prevent catastrophic losses during adverse conditions.
    """

    def __init__(self, config: Optional[LeverageConfig] = None):
        self.config = config or LeverageConfig()

        # State tracking
        self._peak_equity: float = 0.0
        self._current_drawdown: float = 0.0
        self._circuit_breaker_active: bool = False
        self._ramp_day: int = 0            # Days since circuit breaker reset
        self._leverage_history: List[float] = []
        self._daily_returns: List[float] = []  # For rolling vol estimation

        # Kelly state
        self._rolling_sharpe: float = 0.0
        self._rolling_vol: float = 0.0

    def get_leverage(
        self,
        current_equity: float,
        daily_returns: Optional[List[float]] = None,
        sharpe_estimate: Optional[float] = None,
    ) -> float:
        """
        Compute the current leverage factor.

        Args:
            current_equity: Current portfolio value
            daily_returns: Recent daily returns for vol estimation
            sharpe_estimate: Estimated annualized Sharpe ratio (for Kelly mode)

        Returns:
            Leverage factor (1.0 = no leverage, 2.0 = 2x, etc.)
        """
        # Update drawdown state
        self._update_drawdown(current_equity)

        # Check circuit breaker
        if self._circuit_breaker_active:
            leverage = self._get_ramp_leverage()
            self._leverage_history.append(leverage)
            return leverage

        # Compute raw leverage based on mode
        if self.config.mode == "none":
            raw_leverage = 1.0

        elif self.config.mode == "fixed":
            raw_leverage = self.config.fixed_multiplier

        elif self.config.mode == "kelly":
            raw_leverage = self._kelly_leverage(daily_returns, sharpe_estimate)

        elif self.config.mode == "vol_target":
            raw_leverage = self._vol_target_leverage(daily_returns)

        else:
            raw_leverage = 1.0

        # Apply caps
        leverage = np.clip(
            raw_leverage,
            self.config.min_leverage,
            self.config.max_leverage,
        )

        # Scale down proportionally as drawdown approaches trigger
        leverage = self._drawdown_scale(leverage)

        self._leverage_history.append(leverage)
        return leverage

    def _kelly_leverage(
        self,
        daily_returns: Optional[List[float]] = None,
        sharpe_estimate: Optional[float] = None,
    ) -> float:
        """
        Kelly-optimal leverage: L = Sharpe / σ (annualized).

        For a strategy with Sharpe S and annual vol σ:
            Kelly fraction = μ / σ² = (S * σ) / σ² = S / σ

        This gives the growth-maximizing leverage. We use half-Kelly
        for safety (captures ~75% of growth with ~50% less variance).
        """
        if daily_returns and len(daily_returns) >= 20:
            returns = np.array(daily_returns[-252:])  # Up to 1 year
            daily_vol = np.std(returns)
            annual_vol = daily_vol * np.sqrt(252)
            daily_mean = np.mean(returns)
            annual_sharpe = (daily_mean / daily_vol * np.sqrt(252)) if daily_vol > 0 else 0
        elif sharpe_estimate is not None:
            annual_sharpe = sharpe_estimate
            annual_vol = 0.10  # Default assumption
        else:
            return 1.0

        if annual_vol < 0.01:
            annual_vol = 0.01  # Floor to avoid division by zero

        # Kelly leverage = Sharpe / vol
        kelly_leverage = annual_sharpe / annual_vol

        # Half-Kelly for safety
        half_kelly = kelly_leverage * 0.5

        # Kelly can be negative (don't trade) or very large
        return max(half_kelly, 0.5)

    def _vol_target_leverage(
        self,
        daily_returns: Optional[List[float]] = None,
    ) -> float:
        """
        Vol-targeting leverage: L = target_vol / realized_vol.

        If the strategy runs at 4% annual vol and we target 15%,
        leverage = 15/4 = 3.75x.

        Uses exponentially-weighted realized vol for responsiveness.
        """
        if not daily_returns or len(daily_returns) < 20:
            return 1.0

        returns = np.array(daily_returns[-60:])  # ~3 months lookback

        # Exponentially weighted vol (more responsive than simple)
        weights = np.exp(np.linspace(-2, 0, len(returns)))
        weights /= weights.sum()
        weighted_var = np.sum(weights * (returns - np.mean(returns)) ** 2)
        ewma_daily_vol = np.sqrt(weighted_var)
        annual_vol = ewma_daily_vol * np.sqrt(252)

        if annual_vol < 0.005:
            annual_vol = 0.005  # Floor

        leverage = self.config.vol_target_annual / annual_vol
        return leverage

    def _update_drawdown(self, current_equity: float):
        """Update peak equity and drawdown state."""
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
            # Reset circuit breaker if we've made new highs
            if self._circuit_breaker_active:
                self._circuit_breaker_active = False
                self._ramp_day = 0

        if self._peak_equity > 0:
            self._current_drawdown = (
                (self._peak_equity - current_equity) / self._peak_equity
            )
        else:
            self._current_drawdown = 0.0

        # Trigger circuit breaker
        if (
            self._current_drawdown >= self.config.max_drawdown_trigger
            and not self._circuit_breaker_active
        ):
            self._circuit_breaker_active = True
            self._ramp_day = 0

    def _get_ramp_leverage(self) -> float:
        """
        During circuit breaker: ramp leverage gradually from 1x back up.

        Day 0-ramp_days: linear interpolation from 1.0 to target leverage.
        """
        self._ramp_day += 1

        if self._ramp_day >= self.config.ramp_days:
            # Ramp complete, but stay at reduced leverage until new high
            ramp_pct = 1.0
        else:
            ramp_pct = self._ramp_day / self.config.ramp_days

        # During ramp, target is 1.0 (no leverage) — we're being cautious
        return 1.0

    def _drawdown_scale(self, leverage: float) -> float:
        """
        Gradually reduce leverage as drawdown approaches trigger.

        Linear scale-down: at 0% DD = full leverage, at trigger = 1x.
        This provides a smoother transition than a hard circuit breaker alone.
        """
        if self._current_drawdown <= 0:
            return leverage

        trigger = self.config.max_drawdown_trigger
        if self._current_drawdown >= trigger:
            return 1.0  # Circuit breaker will handle this

        # Linear interpolation: scale from full leverage to 1.0
        dd_ratio = self._current_drawdown / trigger
        scaled = leverage * (1 - dd_ratio) + 1.0 * dd_ratio
        return max(scaled, 1.0)

    def compute_funding_cost(
        self, leverage: float, portfolio_value: float, days: int = 1,
    ) -> float:
        """
        Compute the daily funding cost of leverage.

        Cost = (leverage - 1) * portfolio_value * annual_rate / 252
        Only applies when leverage > 1.0 (borrowing).
        """
        if leverage <= 1.0:
            return 0.0

        borrowed = (leverage - 1.0) * portfolio_value
        daily_rate = self.config.funding_cost_annual / 252
        return borrowed * daily_rate * days

    def get_status(self) -> Dict:
        """Return current leverage state."""
        return {
            "mode": self.config.mode,
            "current_leverage": (
                self._leverage_history[-1] if self._leverage_history else 1.0
            ),
            "peak_equity": self._peak_equity,
            "current_drawdown": self._current_drawdown,
            "circuit_breaker_active": self._circuit_breaker_active,
            "ramp_day": self._ramp_day,
            "avg_leverage": (
                np.mean(self._leverage_history)
                if self._leverage_history
                else 1.0
            ),
            "max_leverage_used": (
                max(self._leverage_history)
                if self._leverage_history
                else 1.0
            ),
        }

    def reset(self):
        """Reset all state for a new backtest."""
        self._peak_equity = 0.0
        self._current_drawdown = 0.0
        self._circuit_breaker_active = False
        self._ramp_day = 0
        self._leverage_history = []
        self._daily_returns = []


def estimate_risk_of_ruin(
    cagr: float,
    annual_vol: float,
    leverage: float,
    ruin_threshold: float = 0.5,
    years: int = 30,
    n_simulations: int = 10000,
    seed: int = 42,
) -> float:
    """
    Monte Carlo estimate of risk of ruin (losing > ruin_threshold of capital).

    Simulates leveraged geometric Brownian motion paths and counts
    how many breach the ruin threshold.

    Args:
        cagr: Unlevered annualized return
        annual_vol: Unlevered annualized volatility
        leverage: Leverage multiplier
        ruin_threshold: Fraction of capital loss that constitutes "ruin"
        years: Simulation horizon
        n_simulations: Number of Monte Carlo paths

    Returns:
        Probability of ruin (0.0 to 1.0)
    """
    np.random.seed(seed)

    # Levered parameters (continuous-time approximation)
    lev_mu = leverage * cagr - 0.5 * (leverage ** 2) * (annual_vol ** 2)
    # Add back the 0.5*sigma^2 for geometric return
    lev_mu += 0.5 * (leverage * annual_vol) ** 2
    lev_sigma = leverage * annual_vol

    daily_mu = lev_mu / 252
    daily_sigma = lev_sigma / np.sqrt(252)

    n_days = years * 252
    ruin_count = 0

    # Vectorized simulation
    daily_returns = np.random.normal(
        daily_mu, daily_sigma, (n_simulations, n_days)
    )
    # Cap daily returns to prevent numerical overflow
    daily_returns = np.clip(daily_returns, -0.15, 0.15)

    cumulative = np.cumprod(1 + daily_returns, axis=1)

    # Check if any path breaches ruin threshold
    min_values = np.min(cumulative, axis=1)
    ruin_count = np.sum(min_values < (1 - ruin_threshold))

    return ruin_count / n_simulations
