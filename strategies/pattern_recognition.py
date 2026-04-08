"""
Volatility-Regime Detection Strategy.

Redesigned based on:
  - Hidden Markov Models for market regime detection (Hamilton, 1989)
  - Regime-switching factor investing (Nystrup et al., 2020)
  - Renaissance Technologies' regime-aware signal processing
  - Conditional volatility targeting (Moreira & Muir, 2017)

WHY IT WORKS:
  Markets alternate between distinct regimes: low-vol trending, high-vol
  mean-reverting, and crisis/crash states. Each regime has different return
  distributions (mean, variance, skewness). A 2-state HMM on daily returns
  reproduces fat tails, negative skewness, and volatility clustering.
  By detecting the current regime, we can:
    1. Choose which strategies to activate (momentum in trends, MR in ranges)
    2. Scale position sizes (reduce in high-vol)
    3. Adjust stop-losses and take-profits dynamically
    4. Avoid the biggest drawdowns by going defensive in crash regimes

IMPLEMENTATION:
  Uses a simplified regime detector based on rolling statistics (no heavy ML
  libraries required for the core logic). Optionally uses HMM from hmmlearn
  if available. Classifies markets into 3 states:
    - TRENDING: Low vol, positive drift, momentum works
    - MEAN_REVERTING: Normal vol, no drift, MR works
    - CRISIS: High vol, negative drift, go defensive

WHEN IT WORKS BEST:
  - Transitions between regimes (early detection = biggest edge)
  - High-volatility periods (correctly identifying crisis saves capital)
  - Post-crisis recovery (switching back to risk-on early)

WEAKNESSES:
  - Regime detection inherently lags (needs several days of data)
  - Transitions are noisy (whipsaws during regime shifts)
  - HMM states may not map cleanly to intuitive regimes
  - Overfitting risk if too many states or features
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.mean_reversion import Signal


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    CRISIS = "crisis"


class PatternRecognitionStrategy:
    """
    Volatility-Regime Detection strategy.

    Detects the current market regime using rolling statistics and
    generates signals appropriate for each regime:
      - TRENDING_UP: Go long (momentum)
      - TRENDING_DOWN: Go short / exit longs
      - MEAN_REVERTING: Trade reversions (buy dips, sell rips)
      - CRISIS: Reduce exposure, tighten stops

    Also provides regime classification and recommended strategy weights
    to the agent for dynamic strategy allocation.
    """

    name = "pattern_recognition"

    def __init__(
        self,
        regime_window: int = 60,
        vol_window: int = 20,
        trend_threshold: float = 0.0005,
        vol_percentile_high: float = 0.75,
        vol_percentile_low: float = 0.25,
        use_hmm: bool = False,
    ):
        self.regime_window = regime_window
        self.vol_window = vol_window
        self.trend_threshold = trend_threshold
        self.vol_pct_high = vol_percentile_high
        self.vol_pct_low = vol_percentile_low
        self.use_hmm = use_hmm

        # Regime history for smoothing
        self._regime_history: List[MarketRegime] = []
        self._regime_probs: Dict[str, float] = {}

        # Try to import hmmlearn (optional)
        self._hmm_model = None
        if use_hmm:
            try:
                from hmmlearn.hmm import GaussianHMM
                self._hmm_class = GaussianHMM
            except ImportError:
                self.use_hmm = False

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Signal:
        """
        Detect current market regime and generate an appropriate signal.

        The signal encodes:
          - Direction based on regime (long in uptrend, defensive in crisis)
          - Strength based on regime confidence
          - Recommended strategy weights in the reason field
        """
        if len(df) < self.regime_window + 20:
            return Signal(symbol, "HOLD", 0.0, self.name, "Insufficient data")

        # -- Detect regime --
        regime = self._detect_regime(df)
        regime_confidence = self._regime_confidence(df, regime)

        # -- Regime-specific signal --
        regime_signal = self._regime_to_signal(df, regime)

        # -- Volatility scaling --
        vol_scalar = self._vol_position_scalar(df)

        # -- Composite --
        composite = regime_signal * regime_confidence * vol_scalar
        composite = np.clip(composite, -1.0, 1.0)

        # -- Recommended strategy weights for this regime --
        rec_weights = self.get_regime_weights(regime)

        # -- Decision --
        if composite > 0.15:
            action = "BUY"
        elif composite < -0.15:
            action = "SELL"
        else:
            action = "HOLD"

        reason = (
            f"Regime={regime.value} (conf={regime_confidence:.2f}), "
            f"vol_scalar={vol_scalar:.2f}, "
            f"rec_weights={rec_weights}"
        )

        return Signal(symbol, action, composite, self.name, reason)

    def detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Public interface to get current regime."""
        return self._detect_regime(df)

    def get_regime_weights(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Get recommended strategy weights for a given regime.

        This is used by the agent to dynamically re-weight strategies
        based on the current market environment.
        """
        if regime == MarketRegime.TRENDING_UP:
            return {
                "momentum": 0.45,
                "mean_reversion": 0.10,
                "sentiment": 0.25,
                "pattern_recognition": 0.20,
            }
        elif regime == MarketRegime.TRENDING_DOWN:
            return {
                "momentum": 0.40,
                "mean_reversion": 0.15,
                "sentiment": 0.25,
                "pattern_recognition": 0.20,
            }
        elif regime == MarketRegime.MEAN_REVERTING:
            return {
                "momentum": 0.15,
                "mean_reversion": 0.45,
                "sentiment": 0.20,
                "pattern_recognition": 0.20,
            }
        else:  # CRISIS
            return {
                "momentum": 0.10,
                "mean_reversion": 0.25,
                "sentiment": 0.30,
                "pattern_recognition": 0.35,
            }

    # -- Internal Methods --

    def _detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        """
        Classify current market regime using rolling statistics.

        Features used:
          1. Mean of recent returns (drift direction)
          2. Volatility ratio (recent vs historical)
          3. Return distribution shape (skewness)
          4. Maximum drawdown in recent window
        """
        if "returns" not in df.columns or len(df) < self.regime_window:
            return MarketRegime.MEAN_REVERTING

        returns = df["returns"].iloc[-self.regime_window:].dropna()
        if len(returns) < 20:
            return MarketRegime.MEAN_REVERTING

        # Feature 1: Mean return (drift)
        mean_ret = returns.mean()

        # Feature 2: Volatility ratio
        recent_vol = returns.iloc[-self.vol_window:].std() if len(returns) >= self.vol_window else returns.std()
        hist_vol = df["returns"].dropna().std() if "returns" in df.columns else recent_vol
        vol_ratio = recent_vol / max(hist_vol, 1e-8)

        # Feature 3: Skewness
        skew = float(returns.skew()) if len(returns) > 10 else 0.0

        # Feature 4: Recent max drawdown
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.cummax()
        drawdown = ((peak - cum_returns) / peak).max()

        # -- Classification Rules --

        # Crisis: Very high vol + large drawdown + negative skew
        if vol_ratio > 1.8 and drawdown > 0.05 and skew < -0.5:
            regime = MarketRegime.CRISIS
        elif vol_ratio > 2.0 and drawdown > 0.08:
            regime = MarketRegime.CRISIS
        # Trending up: Positive drift + not extreme vol
        elif mean_ret > self.trend_threshold and vol_ratio < 1.5:
            regime = MarketRegime.TRENDING_UP
        # Trending down: Negative drift + not extreme vol
        elif mean_ret < -self.trend_threshold and vol_ratio < 1.5:
            regime = MarketRegime.TRENDING_DOWN
        # Mean-reverting: Low drift and/or moderate vol
        else:
            regime = MarketRegime.MEAN_REVERTING

        # -- Smooth with history (avoid rapid switching) --
        self._regime_history.append(regime)
        self._regime_history = self._regime_history[-10:]

        # If regime just changed, require 3 consecutive days for confirmation
        if len(self._regime_history) >= 3:
            last_3 = self._regime_history[-3:]
            if all(r == regime for r in last_3):
                return regime
            # Otherwise, stay with the previous confirmed regime
            # (unless it's CRISIS, which should switch immediately)
            if regime == MarketRegime.CRISIS:
                return regime
            # Return the most common regime in the last 5
            from collections import Counter
            recent = self._regime_history[-5:]
            most_common = Counter(recent).most_common(1)[0][0]
            return most_common

        return regime

    def _regime_confidence(
        self, df: pd.DataFrame, regime: MarketRegime
    ) -> float:
        """
        How confident are we in the regime classification?

        Uses the strength of the classification features.
        Returns 0.3 (low) to 1.0 (high).
        """
        if "returns" not in df.columns or len(df) < self.regime_window:
            return 0.5

        returns = df["returns"].iloc[-self.regime_window:].dropna()
        if len(returns) < 20:
            return 0.5

        mean_ret = returns.mean()
        recent_vol = returns.iloc[-self.vol_window:].std() if len(returns) >= self.vol_window else returns.std()
        hist_vol = df["returns"].dropna().std()
        vol_ratio = recent_vol / max(hist_vol, 1e-8)

        if regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            # Confidence from trend strength
            trend_strength = abs(mean_ret) / max(returns.std(), 1e-8)
            return np.clip(0.3 + trend_strength * 2, 0.3, 1.0)
        elif regime == MarketRegime.CRISIS:
            # High vol = high confidence in crisis
            return np.clip(0.5 + (vol_ratio - 1.5) * 0.5, 0.5, 1.0)
        else:
            # Mean-reverting: moderate confidence
            return 0.6

    def _regime_to_signal(
        self, df: pd.DataFrame, regime: MarketRegime
    ) -> float:
        """
        Convert regime classification into a directional signal.

        TRENDING_UP: Positive (go with trend)
        TRENDING_DOWN: Negative (go with trend / exit longs)
        MEAN_REVERTING: Slight contrarian based on recent move
        CRISIS: Defensive (negative / reduce exposure)
        """
        if regime == MarketRegime.TRENDING_UP:
            return 0.6

        elif regime == MarketRegime.TRENDING_DOWN:
            return -0.5

        elif regime == MarketRegime.CRISIS:
            return -0.7  # Defensive

        else:  # MEAN_REVERTING
            # Slight contrarian: buy recent dips, sell recent rallies
            if "returns" in df.columns and len(df) >= 5:
                recent_ret = df["returns"].iloc[-5:].sum()
                return np.clip(-recent_ret * 5, -0.4, 0.4)
            return 0.0

    def _vol_position_scalar(self, df: pd.DataFrame) -> float:
        """
        Position scaling based on current volatility regime.

        Conditional vol targeting: only adjust in extreme vol percentiles.
        Normal vol -> no adjustment.
        High vol (>75th percentile) -> reduce to 50-70%.
        Low vol (<25th percentile) -> can slightly increase.
        """
        if "returns" not in df.columns or len(df) < 252:
            return 1.0

        returns = df["returns"].dropna()
        if len(returns) < 252:
            return 1.0

        current_vol = returns.iloc[-20:].std()
        vol_series = returns.rolling(20).std().dropna()

        if len(vol_series) < 50:
            return 1.0

        percentile = float((vol_series < current_vol).mean())

        if percentile > self.vol_pct_high:
            # High vol: reduce position size
            return np.clip(1.0 - (percentile - self.vol_pct_high) * 2, 0.5, 0.9)
        elif percentile < self.vol_pct_low:
            # Low vol: slight increase
            return np.clip(1.0 + (self.vol_pct_low - percentile) * 0.5, 1.0, 1.3)
        return 1.0

    def get_key_levels(
        self, df: pd.DataFrame, window: int = 50
    ) -> Tuple[float, float]:
        """Return (support, resistance) levels."""
        if len(df) < window:
            window = len(df)
        recent = df.iloc[-window:]
        return float(recent["Low"].min()), float(recent["High"].max())
