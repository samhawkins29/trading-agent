"""
Cross-Asset Signal Strategy.

Based on:
  - Cross-asset momentum (equities + bonds + volatility)
  - Ilmanen (2011): "Expected Returns" — multi-asset factor approach
  - AQR factor portfolios across asset classes
  - Volatility regime signals (VIX-based risk management)

WHY IT WORKS:
  Asset classes are interconnected. Bond yields signal economic expectations.
  Volatility signals risk appetite. Credit spreads indicate financial stress.
  By monitoring cross-asset signals, we can:
  1. Detect risk-on/risk-off shifts before they show in equity prices
  2. Use bond momentum as a leading indicator for equity allocation
  3. Use volatility regime to scale positions dynamically

IMPLEMENTATION:
  Since we're working with single-stock data, we generate synthetic
  cross-asset proxies from the equity data itself:
  - Bond proxy: inverse of equity momentum (when stocks fall, bonds rally)
  - Volatility proxy: realized vol and vol-of-vol
  - Regime signal: composite of vol level, vol trend, and equity trend

INTEGRATION:
  Provides a "risk multiplier" (0.0 to 1.5) that scales all other strategy
  signals. In risk-off regimes, the multiplier reduces exposure. In calm
  trending markets, it can slightly increase exposure.
"""

import numpy as np
import pandas as pd
from typing import Optional
from strategies.mean_reversion import Signal


class CrossAssetSignalStrategy:
    """
    Cross-asset signal generator that creates a risk environment overlay.

    Uses equity-derived proxies for bond and volatility signals to create
    a risk multiplier for the portfolio.
    """

    name = "cross_asset"

    def __init__(
        self,
        vol_window: int = 21,
        vol_long_window: int = 63,
        trend_window: int = 50,
        vix_proxy_window: int = 21,
        risk_off_vol_threshold: float = 0.25,
        risk_on_vol_threshold: float = 0.12,
    ):
        self.vol_window = vol_window
        self.vol_long_window = vol_long_window
        self.trend_window = trend_window
        self.vix_proxy_window = vix_proxy_window
        self.risk_off_vol_threshold = risk_off_vol_threshold
        self.risk_on_vol_threshold = risk_on_vol_threshold

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Signal:
        """
        Generate a cross-asset informed signal.

        Combines:
          1. Volatility regime (current vol vs historical)
          2. Vol trend (is vol increasing or decreasing?)
          3. Equity-bond rotation signal (inverse momentum)
          4. Risk appetite score
        """
        if len(df) < self.vol_long_window + 50:
            return Signal(symbol, "HOLD", 0.0, self.name, "Insufficient data")

        close = df["Close"].values
        returns = df["returns"].values if "returns" in df.columns else np.diff(np.log(close))

        # 1. Volatility regime
        vol_regime_score = self._vol_regime_score(returns)

        # 2. Vol trend (is volatility increasing or decreasing?)
        vol_trend = self._vol_trend(returns)

        # 3. Equity trend strength
        trend_score = self._trend_score(close)

        # 4. Composite risk score
        # Positive = risk-on (favorable for longs)
        # Negative = risk-off (reduce exposure)
        risk_score = (
            0.40 * vol_regime_score +
            0.25 * vol_trend +
            0.35 * trend_score
        )

        risk_score = np.clip(risk_score, -1.0, 1.0)

        if risk_score > 0.2:
            action = "BUY"
        elif risk_score < -0.2:
            action = "SELL"
        else:
            action = "HOLD"

        reason = (
            f"CrossAsset: vol_regime={vol_regime_score:.3f}, "
            f"vol_trend={vol_trend:.3f}, trend={trend_score:.3f}"
        )

        return Signal(symbol, action, risk_score, self.name, reason)

    def get_risk_multiplier(self, df: pd.DataFrame) -> float:
        """
        Get a portfolio-wide risk multiplier based on cross-asset signals.

        Returns:
          0.0 - 0.5: Risk-off (reduce all positions)
          0.5 - 1.0: Normal (standard position sizing)
          1.0 - 1.5: Risk-on (can slightly increase positions)
        """
        if len(df) < self.vol_long_window + 50:
            return 1.0

        returns = df["returns"].values if "returns" in df.columns else np.diff(np.log(df["Close"].values))
        close = df["Close"].values

        vol_regime = self._vol_regime_score(returns)
        vol_trend = self._vol_trend(returns)
        trend = self._trend_score(close)

        composite = 0.40 * vol_regime + 0.25 * vol_trend + 0.35 * trend

        # Map [-1, 1] to [0.2, 1.5]
        multiplier = 0.85 + composite * 0.65
        return np.clip(multiplier, 0.2, 1.5)

    def _vol_regime_score(self, returns: np.ndarray) -> float:
        """
        Score current volatility regime.

        Low vol = positive (risk-on), high vol = negative (risk-off).
        Uses annualized realized volatility compared to thresholds.
        """
        if len(returns) < self.vol_window:
            return 0.0

        recent_vol = np.std(returns[-self.vol_window:]) * np.sqrt(252)

        if recent_vol < self.risk_on_vol_threshold:
            return 1.0  # Very low vol: strong risk-on
        elif recent_vol > self.risk_off_vol_threshold:
            # Scale negative score by how much vol exceeds threshold
            excess = (recent_vol - self.risk_off_vol_threshold) / self.risk_off_vol_threshold
            return max(-1.0, -excess)
        else:
            # Linear interpolation between thresholds
            range_size = self.risk_off_vol_threshold - self.risk_on_vol_threshold
            position = (recent_vol - self.risk_on_vol_threshold) / range_size
            return 1.0 - 2.0 * position  # Maps [0,1] to [1,-1]

    def _vol_trend(self, returns: np.ndarray) -> float:
        """
        Is volatility increasing or decreasing?

        Decreasing vol = positive (improving conditions).
        Increasing vol = negative (deteriorating conditions).
        """
        if len(returns) < self.vol_long_window:
            return 0.0

        short_vol = np.std(returns[-self.vol_window:])
        long_vol = np.std(returns[-self.vol_long_window:])

        if long_vol < 1e-8:
            return 0.0

        ratio = short_vol / long_vol

        if ratio < 0.8:
            return 0.8  # Vol decreasing: positive
        elif ratio > 1.3:
            return -0.8  # Vol increasing: negative
        else:
            return -(ratio - 1.0) * 2.0  # Linear around 1.0

    def _trend_score(self, close: np.ndarray) -> float:
        """
        Equity trend strength using multiple moving averages.

        Positive when price is above key MAs and MAs are properly stacked.
        """
        if len(close) < 200:
            return 0.0

        ma_50 = np.mean(close[-50:])
        ma_100 = np.mean(close[-100:])
        ma_200 = np.mean(close[-200:])

        score = 0.0

        # Price above/below MAs
        if close[-1] > ma_50:
            score += 0.3
        else:
            score -= 0.3

        if close[-1] > ma_200:
            score += 0.3
        else:
            score -= 0.3

        # MA stacking (bullish: 50 > 100 > 200)
        if ma_50 > ma_100 > ma_200:
            score += 0.4  # Perfect bull stack
        elif ma_50 < ma_100 < ma_200:
            score -= 0.4  # Perfect bear stack

        return np.clip(score, -1.0, 1.0)
