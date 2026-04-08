"""
Dual Momentum Strategy (Gary Antonacci's GEM + Keller's VAA enhancements).

Based on:
  - Antonacci (2014): "Dual Momentum Investing" — documented 15-17% CAGR, <20% max DD
  - Keller & Keuning (2017): Vigilant Asset Allocation (VAA) breadth momentum
  - Accelerating Dual Momentum (ADM) — faster signals via weighted lookbacks

WHY IT WORKS:
  Combines two types of momentum:
  1. Absolute Momentum (time-series): Compare asset return to cash/T-bills.
     If asset underperforms cash, move to safety (bonds). This avoids major
     drawdowns by sidestepping recessions.
  2. Relative Momentum (cross-sectional): Among risky assets, invest in the
     one with the best recent return. This captures cross-asset trends.

  The dual filter (absolute + relative) historically turned 7-8% CAGR into
  16-18% by avoiding the worst of each bear market while capturing upside.

IMPLEMENTATION:
  - Multi-period lookback: 1, 3, 6, 12 months with recency weighting (VAA-style)
  - Breadth momentum: count how many offensive assets have positive momentum
  - Crash protection: if breadth is low, allocate to defensive assets
  - Signal for single-stock use: combines absolute momentum filter with
    relative strength ranking vs. broad market proxy

INTEGRATION:
  Used as an overlay on top of existing momentum strategy. When dual momentum
  says "risk-off", it reduces the weight of all offensive strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from strategies.mean_reversion import Signal


class DualMomentumStrategy:
    """
    Dual Momentum with VAA-style breadth momentum overlay.

    Provides:
      1. Asset-level signal: Should we be long this asset or in cash?
      2. Portfolio-level risk signal: What fraction should be in offense vs defense?
    """

    name = "dual_momentum"

    def __init__(
        self,
        lookback_periods: Optional[Dict[str, int]] = None,
        lookback_weights: Optional[Dict[str, float]] = None,
        breadth_threshold: float = 0.5,
        absolute_threshold: float = 0.0,
    ):
        # VAA-style weighted lookbacks (recent months weighted more heavily)
        self.lookback_periods = lookback_periods or {
            "1m": 21,
            "3m": 63,
            "6m": 126,
            "12m": 252,
        }
        # Keller's 12-4-2-1 weighting scheme for VAA
        self.lookback_weights = lookback_weights or {
            "1m": 12.0,
            "3m": 4.0,
            "6m": 2.0,
            "12m": 1.0,
        }
        self.breadth_threshold = breadth_threshold
        self.absolute_threshold = absolute_threshold

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Signal:
        """
        Generate a dual momentum signal for a single asset.

        Steps:
          1. Compute weighted momentum score (VAA-style)
          2. Check absolute momentum (vs. zero / cash proxy)
          3. Determine signal strength based on both filters
        """
        max_lookback = max(self.lookback_periods.values())
        if len(df) < max_lookback + 10:
            return Signal(symbol, "HOLD", 0.0, self.name, "Insufficient data")

        close = df["Close"].values

        # Weighted momentum score (VAA breadth momentum approach)
        mom_score = self._weighted_momentum_score(close)

        # Absolute momentum filter
        abs_mom = self._absolute_momentum(close)

        # Relative strength vs. simple moving average trend
        trend_aligned = self._trend_alignment(close)

        # Combine: both filters must agree for strong signal
        if mom_score > 0 and abs_mom > self.absolute_threshold:
            # Both positive: strong buy
            strength = min(1.0, mom_score * 0.6 + abs_mom * 0.4)
            if trend_aligned:
                strength = min(1.0, strength * 1.15)
            action = "BUY" if strength > 0.2 else "HOLD"
        elif mom_score < 0 or abs_mom < self.absolute_threshold:
            # Either negative: defensive / sell
            strength = max(-1.0, mom_score * 0.5 + abs_mom * 0.5)
            if not trend_aligned:
                strength = max(-1.0, strength * 1.15)
            action = "SELL" if strength < -0.2 else "HOLD"
        else:
            strength = 0.0
            action = "HOLD"

        reason = (
            f"DualMom: weighted_score={mom_score:.3f}, "
            f"abs_mom={abs_mom:.3f}, trend_aligned={trend_aligned}"
        )
        return Signal(symbol, action, strength, self.name, reason)

    def get_risk_allocation(self, asset_data: Dict[str, pd.DataFrame]) -> float:
        """
        Portfolio-level breadth momentum signal.

        Counts what fraction of assets have positive weighted momentum.
        Returns fraction that should be in offensive assets (0.0 to 1.0).
        If breadth is low, shift to defensive allocation.
        """
        if not asset_data:
            return 0.5

        positive_count = 0
        total_count = 0

        for sym, df in asset_data.items():
            if len(df) < max(self.lookback_periods.values()) + 5:
                continue
            close = df["Close"].values
            score = self._weighted_momentum_score(close)
            if score > 0:
                positive_count += 1
            total_count += 1

        if total_count == 0:
            return 0.5

        breadth = positive_count / total_count

        # VAA-style: if any asset has negative momentum, start shifting defensive
        if breadth >= self.breadth_threshold:
            return 1.0  # Full offense
        else:
            # Proportional reduction
            return max(0.0, breadth / self.breadth_threshold)

    def _weighted_momentum_score(self, close: np.ndarray) -> float:
        """
        Compute VAA-style weighted momentum score.

        Uses Keller's 12-4-2-1 weighting that heavily favors recent returns.
        This makes the strategy more responsive to regime changes.
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for name, period in self.lookback_periods.items():
            if len(close) <= period:
                continue
            ret = (close[-1] - close[-period]) / close[-period]
            weight = self.lookback_weights.get(name, 1.0)
            weighted_sum += weight * ret
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight

    def _absolute_momentum(self, close: np.ndarray, period: int = 252) -> float:
        """
        Absolute momentum: 12-month return compared to zero (cash proxy).

        In Antonacci's GEM, this is compared to T-bill returns.
        We use zero as a simplified threshold (equivalent to ~0% cash rate).
        """
        if len(close) <= period:
            return 0.0
        return (close[-1] - close[-period]) / close[-period]

    def _trend_alignment(self, close: np.ndarray, sma_period: int = 200) -> bool:
        """Check if price is above its 200-day SMA (long-term uptrend)."""
        if len(close) < sma_period:
            return True  # Default to aligned if insufficient data
        sma = np.mean(close[-sma_period:])
        return close[-1] > sma
