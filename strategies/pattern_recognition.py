"""
Price Pattern Recognition Strategy.

Inspired by:
  - Renaissance Technologies' pattern detection across vast datasets
  - Hidden Markov Model regime detection (Baum-Welch origins at RenTech)
  - Candlestick pattern recognition and price action analysis
  - DE Shaw's computational pattern search

Concept: Price action forms repeating patterns that precede predictable
moves. We detect candlestick formations, support/resistance levels,
and regime changes using statistical methods.
"""

from enum import Enum
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from strategies.mean_reversion import Signal


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"


class PatternRecognitionStrategy:
    """
    Price pattern recognition.

    Detects:
      1. Candlestick patterns (hammer, engulfing, doji)
      2. Support / resistance levels
      3. Market regime (trend vs range)
      4. Volatility regime shifts
    """

    name = "pattern_recognition"

    def __init__(self, regime_window: int = 30, vol_window: int = 20):
        self.regime_window = regime_window
        self.vol_window = vol_window

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Signal:
        """
        Analyze price patterns and return a Signal.

        Expects standard OHLCV DataFrame with indicators.
        """
        if len(df) < self.regime_window + 10:
            return Signal(symbol, "HOLD", 0.0, self.name, "Insufficient data")

        # ── Component Signals ────────────────────────────────────────
        candle_sig = self._candlestick_signal(df)
        sr_sig = self._support_resistance_signal(df)
        regime = self._detect_regime(df)
        regime_sig = self._regime_signal(df, regime)
        vol_sig = self._volatility_signal(df)

        # ── Composite ────────────────────────────────────────────────
        composite = (
            0.30 * candle_sig
            + 0.25 * sr_sig
            + 0.25 * regime_sig
            + 0.20 * vol_sig
        )
        composite = np.clip(composite, -1.0, 1.0)

        # ── Decision ─────────────────────────────────────────────────
        if composite > 0.25:
            action = "BUY"
            reason = (
                f"Pattern BUY: candle={candle_sig:.2f}, "
                f"S/R={sr_sig:.2f}, regime={regime.value}"
            )
        elif composite < -0.25:
            action = "SELL"
            reason = (
                f"Pattern SELL: candle={candle_sig:.2f}, "
                f"S/R={sr_sig:.2f}, regime={regime.value}"
            )
        else:
            action = "HOLD"
            reason = (
                f"No clear pattern: composite={composite:.3f}, "
                f"regime={regime.value}"
            )

        return Signal(symbol, action, composite, self.name, reason)

    # ── Candlestick Patterns ─────────────────────────────────────────
    def _candlestick_signal(self, df: pd.DataFrame) -> float:
        """Detect recent candlestick patterns. Returns -1 to +1."""
        if len(df) < 3:
            return 0.0

        o, h, l, c = (
            df["Open"].values,
            df["High"].values,
            df["Low"].values,
            df["Close"].values,
        )
        signal = 0.0

        # Hammer / Hanging Man (last candle)
        body = abs(c[-1] - o[-1])
        lower_shadow = min(o[-1], c[-1]) - l[-1]
        upper_shadow = h[-1] - max(o[-1], c[-1])
        total_range = h[-1] - l[-1]

        if total_range > 0:
            if lower_shadow > 2 * body and upper_shadow < body * 0.5:
                # Hammer (bullish if after downtrend)
                if c[-1] < c[-5] if len(c) > 5 else False:
                    signal += 0.5  # bullish hammer
                else:
                    signal -= 0.3  # hanging man (bearish)

        # Bullish Engulfing
        if len(df) >= 2:
            prev_body = c[-2] - o[-2]
            curr_body = c[-1] - o[-1]
            if prev_body < 0 and curr_body > 0:
                if abs(curr_body) > abs(prev_body) * 1.2:
                    signal += 0.6  # bullish engulfing

            # Bearish Engulfing
            if prev_body > 0 and curr_body < 0:
                if abs(curr_body) > abs(prev_body) * 1.2:
                    signal -= 0.6  # bearish engulfing

        # Doji (indecision)
        if total_range > 0 and body / total_range < 0.1:
            signal *= 0.5  # reduce conviction on doji

        return np.clip(signal, -1.0, 1.0)

    # ── Support / Resistance ─────────────────────────────────────────
    def _support_resistance_signal(
        self, df: pd.DataFrame, window: int = 20
    ) -> float:
        """
        Signal based on proximity to support/resistance levels.
        Near support → bullish; near resistance → bearish.
        """
        if len(df) < window:
            return 0.0

        recent = df.iloc[-window:]
        support = recent["Low"].min()
        resistance = recent["High"].max()
        current = df["Close"].iloc[-1]

        sr_range = resistance - support
        if sr_range <= 0:
            return 0.0

        position = (current - support) / sr_range  # 0 = at support, 1 = at resistance

        # Near support → buy signal; near resistance → sell signal
        # But also consider breakout potential
        if position < 0.2:
            return 0.5  # near support, potential bounce
        elif position > 0.8:
            return -0.5  # near resistance, potential rejection
        else:
            return 0.0  # mid-range, no edge

    # ── Regime Detection ─────────────────────────────────────────────
    def _detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        """
        Classify current market regime.
        Simplified Hidden Markov Model approach using returns distribution.
        """
        if "returns" not in df.columns or len(df) < self.regime_window:
            return MarketRegime.RANGING

        returns = df["returns"].iloc[-self.regime_window:].dropna()
        if len(returns) < 10:
            return MarketRegime.RANGING

        mean_ret = returns.mean()
        vol = returns.std()
        historical_vol = df["returns"].std() if "returns" in df.columns else vol

        # High-volatility regime
        if vol > 1.5 * historical_vol:
            return MarketRegime.HIGH_VOLATILITY

        # Trending regimes
        trend_threshold = 0.001  # daily return threshold
        if mean_ret > trend_threshold:
            return MarketRegime.TRENDING_UP
        elif mean_ret < -trend_threshold:
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.RANGING

    def _regime_signal(
        self, df: pd.DataFrame, regime: MarketRegime
    ) -> float:
        """Convert regime into a directional signal."""
        if regime == MarketRegime.TRENDING_UP:
            return 0.5  # go with the trend
        elif regime == MarketRegime.TRENDING_DOWN:
            return -0.5
        elif regime == MarketRegime.HIGH_VOLATILITY:
            return 0.0  # reduce exposure in high vol
        else:
            return 0.0  # ranging — no directional edge

    # ── Volatility Regime ────────────────────────────────────────────
    def _volatility_signal(self, df: pd.DataFrame) -> float:
        """
        Signal from volatility regime shifts.
        Low vol → expect breakout; high vol → expect mean reversion.
        """
        if "ATR" not in df.columns or len(df) < self.vol_window + 5:
            return 0.0

        atr_current = df["ATR"].iloc[-1]
        atr_mean = df["ATR"].iloc[-self.vol_window:].mean()

        if atr_mean <= 0:
            return 0.0

        vol_ratio = atr_current / atr_mean

        if vol_ratio < 0.7:
            # Low vol — coiled spring, could break either way
            # Slight bullish bias (vol compression often precedes upside)
            return 0.2
        elif vol_ratio > 1.5:
            # High vol — mean-revert bias (sell if up, buy if down)
            recent_return = df["returns"].iloc[-5:].sum() if "returns" in df.columns else 0
            return -0.3 if recent_return > 0 else 0.3
        return 0.0

    def get_key_levels(
        self, df: pd.DataFrame, window: int = 50
    ) -> Tuple[float, float]:
        """Return (support, resistance) levels."""
        if len(df) < window:
            window = len(df)
        recent = df.iloc[-window:]
        return float(recent["Low"].min()), float(recent["High"].max())
