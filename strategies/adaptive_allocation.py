"""
Adaptive Asset Allocation Strategy.

Based on:
  - Butler, Philbrick, Gordillo (2012): "Adaptive Asset Allocation"
  - Risk Parity / All-Weather concepts (Bridgewater, Dalio)
  - Keller's DAAA (Defensive Adaptive Asset Allocation)
  - Hierarchical Risk Parity (Lopez de Prado, 2016)

WHY IT WORKS:
  Traditional fixed-weight portfolios (60/40) suffer because correlations
  and volatilities change over time. Adaptive allocation continuously
  adjusts position weights based on:
  1. Recent return momentum (tilt toward winners)
  2. Inverse volatility weighting (equal risk contribution)
  3. Correlation-aware sizing (reduce correlated positions)

  Backtests show 10-15% CAGR with <15% max drawdown — better risk-adjusted
  returns than fixed allocation or market-cap weighting.

IMPLEMENTATION:
  For a single-stock trading agent, this strategy provides:
  1. Position sizing recommendations based on inverse-vol (risk parity)
  2. Dynamic confidence scaling based on recent strategy performance
  3. Regime-adaptive weight suggestions that blend momentum and mean-reversion
     signals based on current market conditions

INTEGRATION:
  Acts as the "portfolio optimizer" layer that sits above individual
  strategy signals. Takes raw signals and adjusts their weights/sizes
  for risk-efficiency.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from strategies.mean_reversion import Signal


class AdaptiveAllocationStrategy:
    """
    Adaptive Asset Allocation with risk parity sizing.

    Provides portfolio-level optimization:
      1. Risk parity position sizing
      2. Momentum-based strategy weight tilting
      3. Drawdown-aware dynamic de-risking
    """

    name = "adaptive_allocation"

    def __init__(
        self,
        vol_target: float = 0.12,
        lookback: int = 63,
        rebalance_period: int = 21,
        max_position_vol: float = 0.20,
        drawdown_threshold: float = 0.05,
        severe_drawdown: float = 0.10,
    ):
        self.vol_target = vol_target
        self.lookback = lookback
        self.rebalance_period = rebalance_period
        self.max_position_vol = max_position_vol
        self.drawdown_threshold = drawdown_threshold
        self.severe_drawdown = severe_drawdown

        # Track strategy performance for adaptive weighting
        self._strategy_returns: Dict[str, List[float]] = {}
        self._day_counter = 0

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Signal:
        """
        Generate an allocation-aware signal.

        This strategy doesn't directly generate buy/sell signals.
        Instead, it provides a position-sizing multiplier based on:
        1. Current volatility vs target
        2. Recent drawdown level
        3. Momentum of the asset
        """
        if len(df) < self.lookback + 20:
            return Signal(symbol, "HOLD", 0.0, self.name, "Insufficient data")

        close = df["Close"].values
        returns_series = df["returns"].values if "returns" in df.columns else np.diff(np.log(close))

        # Risk parity sizing
        vol_scale = self._risk_parity_scale(returns_series)

        # Drawdown awareness
        dd_scale = self._drawdown_scale(close)

        # Momentum tilt
        mom_tilt = self._momentum_tilt(close)

        # Combined signal
        composite = vol_scale * dd_scale * mom_tilt
        composite = np.clip(composite, -1.0, 1.0)

        if composite > 0.15:
            action = "BUY"
        elif composite < -0.15:
            action = "SELL"
        else:
            action = "HOLD"

        reason = (
            f"AdaptiveAlloc: vol_scale={vol_scale:.3f}, "
            f"dd_scale={dd_scale:.3f}, mom_tilt={mom_tilt:.3f}"
        )

        return Signal(symbol, action, composite, self.name, reason)

    def compute_strategy_weights(
        self,
        strategy_signals: Dict[str, float],
        strategy_recent_returns: Dict[str, List[float]],
        current_regime: str = "normal",
    ) -> Dict[str, float]:
        """
        Compute adaptive strategy weights based on recent performance.

        Uses exponential weighting of recent strategy returns to tilt
        toward strategies that have been performing well recently.
        """
        base_weights = {
            "momentum": 0.25,
            "mean_reversion": 0.20,
            "sentiment": 0.15,
            "pattern_recognition": 0.15,
            "dual_momentum": 0.15,
            "cross_asset": 0.10,
        }

        # Adjust for regime
        if current_regime == "crisis":
            base_weights["momentum"] *= 0.5
            base_weights["mean_reversion"] *= 1.3
            base_weights["dual_momentum"] *= 1.5
            base_weights["cross_asset"] *= 1.5
        elif current_regime in ("trending_up", "trending_down"):
            base_weights["momentum"] *= 1.3
            base_weights["dual_momentum"] *= 1.3
            base_weights["mean_reversion"] *= 0.7

        # Performance-based tilting
        if strategy_recent_returns:
            perf_scores = {}
            for name, rets in strategy_recent_returns.items():
                if len(rets) >= 5:
                    # Risk-adjusted recent performance
                    avg = np.mean(rets)
                    std = np.std(rets) + 1e-8
                    perf_scores[name] = avg / std
                else:
                    perf_scores[name] = 0.0

            if perf_scores:
                # Softmax for performance weights
                scores = np.array(list(perf_scores.values()))
                scores_shifted = scores - np.max(scores)
                exp_scores = np.exp(np.clip(scores_shifted, -10, 10))
                perf_weights = exp_scores / exp_scores.sum()

                # Blend: 70% base + 30% performance
                for i, name in enumerate(perf_scores.keys()):
                    if name in base_weights:
                        base_weights[name] = (
                            0.7 * base_weights[name] + 0.3 * perf_weights[i]
                        )

        # Normalize
        total = sum(base_weights.values())
        if total > 0:
            base_weights = {k: v / total for k, v in base_weights.items()}

        return base_weights

    def get_position_size_multiplier(
        self, df: pd.DataFrame, signal_strength: float
    ) -> float:
        """
        Compute position size multiplier using risk parity principles.

        Scales position inversely with volatility so each position
        contributes roughly equal risk to the portfolio.
        """
        if len(df) < 21:
            return abs(signal_strength)

        returns = df["returns"].values if "returns" in df.columns else np.diff(np.log(df["Close"].values))

        recent_vol = np.std(returns[-21:]) * np.sqrt(252)
        if recent_vol < 0.01:
            recent_vol = 0.15  # Default to market-average vol

        # Inverse vol sizing
        vol_multiplier = self.vol_target / recent_vol
        vol_multiplier = np.clip(vol_multiplier, 0.3, 2.5)

        # Signal-scaled
        size = abs(signal_strength) * vol_multiplier

        # Cap at max position vol contribution
        max_size = self.max_position_vol / recent_vol
        return min(size, max_size)

    def _risk_parity_scale(self, returns: np.ndarray) -> float:
        """Inverse-volatility scaling for risk parity."""
        if len(returns) < 21:
            return 0.5

        recent_vol = np.std(returns[-21:]) * np.sqrt(252)
        if recent_vol < 0.01:
            return 0.5

        scale = self.vol_target / recent_vol
        return np.clip(scale, 0.2, 2.0)

    def _drawdown_scale(self, close: np.ndarray) -> float:
        """
        Reduce position size based on recent drawdown.

        Implements dynamic de-risking: as drawdown increases,
        position sizes decrease to protect capital.
        """
        if len(close) < 50:
            return 1.0

        recent = close[-50:]
        peak = np.maximum.accumulate(recent)
        current_dd = (peak[-1] - close[-1]) / peak[-1] if peak[-1] > 0 else 0

        if current_dd < self.drawdown_threshold:
            return 1.0  # No de-risking needed
        elif current_dd < self.severe_drawdown:
            # Linear reduction
            pct = (current_dd - self.drawdown_threshold) / (
                self.severe_drawdown - self.drawdown_threshold
            )
            return 1.0 - 0.5 * pct  # Reduce to 50% at severe threshold
        else:
            # Severe: reduce to 20-30%
            excess = (current_dd - self.severe_drawdown) / self.severe_drawdown
            return max(0.2, 0.5 - 0.3 * excess)

    def _momentum_tilt(self, close: np.ndarray) -> float:
        """
        Momentum-based directional tilt.

        Positive when recent returns are positive and accelerating.
        Negative when recent returns are negative.
        """
        if len(close) < self.lookback:
            return 0.0

        ret_1m = (close[-1] - close[-21]) / close[-21] if len(close) >= 21 else 0
        ret_3m = (close[-1] - close[-63]) / close[-63] if len(close) >= 63 else 0

        # Combine short and medium momentum
        tilt = 0.6 * np.clip(ret_1m / 0.05, -1, 1) + 0.4 * np.clip(ret_3m / 0.10, -1, 1)

        return np.clip(tilt, -1.0, 1.0)
