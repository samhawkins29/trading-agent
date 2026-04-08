"""
Factor Momentum Strategy (replaces Sentiment strategy).

Redesigned based on:
  - Fama-French five-factor model (market, value, size, profitability, investment)
  - Carhart four-factor model (adding UMD momentum factor)
  - Research showing multi-factor portfolios outperform 75-82% of time
  - Factor momentum documented by Gupta and Kelly (2019)

WHY IT WORKS:
  Academic research spanning 60+ years shows that certain stock characteristics
  ("factors") predict future returns. Value stocks (low price-to-book) earn a
  premium because they are riskier or because investors systematically misjudge
  them. Momentum works due to behavioral underreaction. Quality (profitable firms)
  earns a premium because the market undervalues stable earnings. Combining
  multiple factors provides diversification since different factors work in
  different market environments.

IMPLEMENTATION:
  Since we don't have fundamental data (balance sheet, earnings), we construct
  PRICE-BASED PROXIES for the main factors:
  - Momentum (UMD): 12-month return minus last month (skip 1 month)
  - Value proxy: 52-week low ratio (how close to 52-week low = "cheap")
  - Quality proxy: Earnings stability via price volatility (low vol = stable)
  - Size proxy: Not applicable for single-stock signals

WHEN IT WORKS BEST:
  - Factor momentum strongest in normal volatility environments
  - Value works in rising rate environments
  - Momentum works in trending markets
  - Quality works as defensive factor in downturns

WEAKNESSES:
  - Value has underperformed for extended periods (1998-2020)
  - Factor crowding reduces future returns
  - All factors can crash simultaneously in liquidity crises
  - Price-based proxies are imperfect substitutes for fundamental factors
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.mean_reversion import Signal


class SentimentStrategy:
    """
    Factor Momentum strategy (renamed from Sentiment for backward compatibility).

    Constructs price-based factor signals and combines them into a composite
    multi-factor score. Uses factor momentum (recent factor performance predicts
    near-term factor performance) to dynamically weight factors.

    Factors computed from price data alone:
      1. Cross-sectional Momentum (12-1 month return)
      2. Value proxy (distance from 52-week low / 52-week range)
      3. Quality proxy (inverse volatility as stability measure)
      4. Short-term reversal (1-week return, contrarian)
    """

    name = "sentiment"  # Keep name for backward compatibility

    def __init__(
        self,
        momentum_lookback: int = 252,
        momentum_skip: int = 21,
        value_lookback: int = 252,
        quality_vol_window: int = 60,
        reversal_window: int = 5,
        factor_weights: Optional[Dict[str, float]] = None,
    ):
        self.momentum_lookback = momentum_lookback
        self.momentum_skip = momentum_skip
        self.value_lookback = value_lookback
        self.quality_vol_window = quality_vol_window
        self.reversal_window = reversal_window

        # Default factor weights (can be overridden)
        self.factor_weights = factor_weights or {
            "momentum": 0.35,
            "value": 0.25,
            "quality": 0.20,
            "reversal": 0.20,
        }

        # Track recent factor performance for factor momentum
        self._factor_history: Dict[str, List[float]] = {
            k: [] for k in self.factor_weights
        }

    def generate_signal(
        self, symbol: str, df_or_articles, **kwargs
    ) -> Signal:
        """
        Generate a multi-factor signal from price data.

        Accepts either a DataFrame (new interface) or a list of articles
        (legacy interface, returns neutral signal).
        """
        # Handle legacy interface (list of articles)
        if isinstance(df_or_articles, list):
            return Signal(
                symbol, "HOLD", 0.0, self.name,
                "Factor strategy requires price data (DataFrame)"
            )

        df = df_or_articles
        if not isinstance(df, pd.DataFrame) or len(df) < self.momentum_lookback + 10:
            return Signal(symbol, "HOLD", 0.0, self.name, "Insufficient data")

        close = df["Close"].values

        # -- Compute individual factor signals --
        factors = {}

        # 1. Momentum factor (12-month return, skip last month)
        factors["momentum"] = self._momentum_factor(close)

        # 2. Value proxy (position within 52-week range)
        factors["value"] = self._value_proxy(close)

        # 3. Quality proxy (inverse volatility = earnings stability proxy)
        factors["quality"] = self._quality_proxy(df)

        # 4. Short-term reversal (1-week contrarian)
        factors["reversal"] = self._reversal_factor(close)

        # -- Dynamic factor weighting via factor momentum --
        dynamic_weights = self._compute_dynamic_weights(factors)

        # -- Composite signal --
        composite = sum(
            dynamic_weights.get(name, 0) * signal
            for name, signal in factors.items()
        )
        composite = np.clip(composite, -1.0, 1.0)

        # -- Decision --
        factor_strs = [f"{n}={v:.3f}" for n, v in factors.items()]
        if composite > 0.2:
            action = "BUY"
            reason = (
                f"Factor BUY: {', '.join(factor_strs)}, "
                f"composite={composite:.3f}"
            )
        elif composite < -0.2:
            action = "SELL"
            reason = (
                f"Factor SELL: {', '.join(factor_strs)}, "
                f"composite={composite:.3f}"
            )
        else:
            action = "HOLD"
            reason = f"Neutral factors: composite={composite:.3f}"

        return Signal(symbol, action, composite, self.name, reason)

    def _momentum_factor(self, close: np.ndarray) -> float:
        """
        12-1 month momentum factor (Carhart UMD).

        Returns the 12-month return excluding the most recent month,
        scaled to [-1, 1]. The 1-month skip avoids short-term reversal
        contaminating the momentum signal.
        """
        if len(close) < self.momentum_lookback + self.momentum_skip:
            return 0.0

        # 12-month return ending 1 month ago
        end_idx = -self.momentum_skip
        start_idx = end_idx - self.momentum_lookback
        if close[start_idx] <= 0:
            return 0.0

        ret_12_1 = (close[end_idx] - close[start_idx]) / close[start_idx]

        # Scale: 15% return = full signal
        return np.clip(ret_12_1 / 0.15, -1.0, 1.0)

    def _value_proxy(self, close: np.ndarray) -> float:
        """
        Value proxy: position within 52-week range.

        Stocks near 52-week lows are "cheap" (value), stocks near
        52-week highs are "expensive" (growth/momentum).
        Returns signal in [-1, 1]: positive = cheap (buy), negative = expensive.
        """
        if len(close) < self.value_lookback:
            return 0.0

        window = close[-self.value_lookback:]
        low_52w = np.min(window)
        high_52w = np.max(window)
        range_52w = high_52w - low_52w

        if range_52w <= 0:
            return 0.0

        # Position: 0 = at low, 1 = at high
        position = (close[-1] - low_52w) / range_52w

        # Invert: near low = positive (value buy), near high = negative
        value_signal = -(position - 0.5) * 2

        # Non-linear: stronger signal at extremes
        if abs(value_signal) > 0.7:
            value_signal *= 1.2

        return np.clip(value_signal, -1.0, 1.0)

    def _quality_proxy(self, df: pd.DataFrame) -> float:
        """
        Quality proxy: inverse realized volatility.

        Low-volatility stocks tend to be higher quality (stable earnings,
        strong balance sheets). This is a price-based proxy for the
        Fama-French RMW (profitability) factor.

        Returns signal in [-1, 1]: positive = low vol (quality), negative = high vol.
        """
        if "returns" not in df.columns or len(df) < self.quality_vol_window:
            return 0.0

        returns = df["returns"].iloc[-self.quality_vol_window:].dropna()
        if len(returns) < 20:
            return 0.0

        vol = returns.std() * np.sqrt(252)

        # Low vol = quality (positive signal)
        # Typical stock vol: 15-40% annualized
        # Below 20% = very low vol (quality), above 35% = high vol (speculative)
        quality_signal = np.clip((0.25 - vol) / 0.15, -1.0, 1.0)

        return quality_signal

    def _reversal_factor(self, close: np.ndarray) -> float:
        """
        Short-term reversal factor (1-week contrarian).

        Stocks that dropped sharply in the past week tend to bounce,
        and stocks that surged tend to pull back. This is the flip side
        of momentum at very short horizons.
        """
        if len(close) < self.reversal_window + 1:
            return 0.0

        week_return = (close[-1] - close[-self.reversal_window - 1]) / close[-self.reversal_window - 1]

        # Contrarian: negative return = buy, positive return = sell
        # Scale: 3% weekly move = moderate signal
        reversal_signal = np.clip(-week_return / 0.03, -1.0, 1.0)

        return reversal_signal

    def _compute_dynamic_weights(
        self, current_factors: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Factor momentum: tilt weights toward recently-performing factors.

        If a factor has been generating good signals recently, increase
        its weight. Uses exponential moving average of factor signals.
        """
        # Store current factor values
        for name, value in current_factors.items():
            if name in self._factor_history:
                self._factor_history[name].append(value)
                # Keep last 60 observations
                self._factor_history[name] = self._factor_history[name][-60:]

        # If not enough history, use static weights
        if any(len(v) < 10 for v in self._factor_history.values()):
            return dict(self.factor_weights)

        # Compute recent "momentum of factors"
        factor_scores = {}
        for name, history in self._factor_history.items():
            recent = np.array(history[-20:])
            # Consistency: are signals mostly positive or negative?
            consistency = abs(np.mean(recent)) / max(np.std(recent), 1e-6)
            factor_scores[name] = consistency

        # Softmax to convert scores to weights
        if all(s == 0 for s in factor_scores.values()):
            return dict(self.factor_weights)

        base = np.array([self.factor_weights[n] for n in self.factor_weights])
        scores = np.array([factor_scores.get(n, 0) for n in self.factor_weights])

        # Blend: 70% base weights + 30% score-adjusted weights
        # Stabilize softmax to prevent overflow
        scores_shifted = scores - np.max(scores)
        exp_scores = np.exp(np.clip(scores_shifted, -10, 10))
        score_weights = exp_scores / np.sum(exp_scores)
        blended = 0.7 * base + 0.3 * score_weights
        blended /= blended.sum()

        return dict(zip(self.factor_weights.keys(), blended))

    def get_cached_sentiment(self, symbol: str) -> float:
        """Legacy compatibility: returns 0 (no sentiment cache)."""
        return 0.0
