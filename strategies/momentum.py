"""
Time Series Momentum Strategy.

Redesigned based on:
  - Moskowitz, Ooi, Pedersen (2012): "Time Series Momentum"
  - AQR's "A Century of Evidence on Trend-Following Investing"
  - Managed futures CTA strategies (AQR, Man AHL) with 10-15% annualized

WHY IT WORKS:
  Assets that have gone up tend to continue going up (and vice versa) over
  1-12 month horizons. This is driven by: (1) behavioral underreaction to
  new information, (2) herding and positive feedback loops, (3) central bank
  policy persistence, and (4) corporate earnings momentum. The effect has been
  documented across 58 liquid instruments over 140+ years with Sharpe ~1.0.

IMPLEMENTATION:
  Uses multiple lookback windows (1, 3, 6, 12 months) and combines them
  into a composite signal. Each window captures different aspects of
  momentum: short-term (1m) is noisier but faster, long-term (12m) is
  smoother but slower. Equal-weighted combination reduces timing risk.

WHEN IT WORKS BEST:
  - Trending markets (strong directional moves)
  - Low-to-normal volatility regimes
  - Post-earnings announcements and macro events

WEAKNESSES:
  - Suffers badly during trend reversals ("momentum crashes")
  - Whipsawed in range-bound/choppy markets
  - Reduced profitability when volatility is very high
  - 2009 and 2020 V-shaped recoveries caused momentum crashes
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from strategies.mean_reversion import Signal


class MomentumStrategy:
    """
    Time Series Momentum (TSMOM) strategy.

    Core idea from Moskowitz et al.: Look at past returns over multiple
    horizons. If returns were positive, go long; if negative, go short
    (or exit). Combine multiple lookback windows for robustness.

    Enhanced with:
      - Volatility scaling (position size inversely proportional to vol)
      - Volume confirmation (higher conviction when volume confirms)
      - Acceleration detection (momentum of momentum)
    """

    name = "momentum"

    def __init__(
        self,
        lookback_windows: Optional[Dict[str, int]] = None,
        vol_target: float = 0.12,
        min_trend_strength: float = 0.15,
    ):
        # Multiple lookback windows (trading days)
        self.lookback_windows = lookback_windows or {
            "1m": 21,
            "3m": 63,
            "6m": 126,
            "12m": 252,
        }
        self.vol_target = vol_target
        self.min_trend_strength = min_trend_strength

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Signal:
        """
        Generate a time-series momentum signal.

        For each lookback window, compute the return over that period.
        Combine windows with equal weight, scale by inverse volatility,
        and confirm with volume.
        """
        max_lookback = max(self.lookback_windows.values())
        if len(df) < max_lookback + 10:
            return Signal(symbol, "HOLD", 0.0, self.name, "Insufficient data")

        close = df["Close"].values

        # -- Multi-window momentum signals --
        window_signals = {}
        window_returns = {}

        for name, period in self.lookback_windows.items():
            if len(close) <= period:
                continue

            # Raw return over lookback period
            ret = (close[-1] - close[-period]) / close[-period]
            window_returns[name] = ret

            # Normalize return: sign * magnitude (capped)
            # Positive return -> buy signal, negative -> sell signal
            signal = np.clip(ret / self.min_trend_strength, -1.0, 1.0)
            window_signals[name] = signal

        if not window_signals:
            return Signal(symbol, "HOLD", 0.0, self.name, "No valid windows")

        # -- Equal-weight combination of lookback windows --
        raw_composite = np.mean(list(window_signals.values()))

        # -- Volatility scaling --
        # Scale signal strength by inverse of realized volatility
        vol_scalar = self._volatility_scalar(df)
        scaled_composite = raw_composite * vol_scalar

        # -- Momentum acceleration (2nd derivative) --
        # Is momentum getting stronger or weaker?
        acceleration = self._momentum_acceleration(close)
        if acceleration > 0 and raw_composite > 0:
            scaled_composite *= 1.15  # Boost when accelerating in trend direction
        elif acceleration < 0 and raw_composite > 0:
            scaled_composite *= 0.85  # Reduce when decelerating

        # -- Volume confirmation --
        vol_confirm = self._volume_confirmation(df)
        scaled_composite *= vol_confirm

        # -- Moving average trend filter --
        # Only take momentum trades in direction of longer-term trend
        trend_filter = self._trend_filter(df)
        if trend_filter * scaled_composite < 0:
            # Momentum opposes the longer-term trend -> reduce conviction
            scaled_composite *= 0.5

        composite = np.clip(scaled_composite, -1.0, 1.0)

        # -- Decision --
        ret_strs = [f"{n}={r:.3f}" for n, r in window_returns.items()]
        if composite > 0.2:
            action = "BUY"
            reason = (
                f"TSMOM BUY: {', '.join(ret_strs)}, "
                f"vol_scalar={vol_scalar:.2f}, accel={acceleration:.4f}"
            )
        elif composite < -0.2:
            action = "SELL"
            reason = (
                f"TSMOM SELL: {', '.join(ret_strs)}, "
                f"vol_scalar={vol_scalar:.2f}, accel={acceleration:.4f}"
            )
        else:
            action = "HOLD"
            reason = f"Weak TSMOM: composite={composite:.3f}"

        return Signal(symbol, action, composite, self.name, reason)

    def _volatility_scalar(self, df: pd.DataFrame) -> float:
        """
        Compute inverse-volatility scalar.

        Position sizes are scaled so that the risk contribution is roughly
        constant across different volatility regimes. Target = vol_target.
        """
        if "returns" not in df.columns or len(df) < 21:
            return 1.0

        recent_vol = df["returns"].iloc[-21:].std() * np.sqrt(252)
        if recent_vol < 0.01:
            return 1.0

        scalar = self.vol_target / recent_vol
        # Cap the scalar to avoid extreme leverage
        return np.clip(scalar, 0.3, 2.0)

    def _momentum_acceleration(
        self, close: np.ndarray, short: int = 5, long: int = 21
    ) -> float:
        """
        Measure momentum acceleration: is the trend speeding up or slowing?

        Compares short-term momentum to medium-term momentum.
        Positive = accelerating, Negative = decelerating.
        """
        if len(close) < long + 1:
            return 0.0

        short_mom = (close[-1] - close[-short]) / close[-short]
        long_mom = (close[-1] - close[-long]) / close[-long]
        long_normalized = long_mom * (short / long)  # Normalize to same timeframe

        return short_mom - long_normalized

    def _volume_confirmation(self, df: pd.DataFrame) -> float:
        """
        Volume confirmation: higher conviction when volume supports the move.

        Returns a multiplier between 0.7 (low volume, less conviction)
        and 1.3 (high volume, more conviction).
        """
        if "Vol_ratio" not in df.columns:
            return 1.0

        vr = df["Vol_ratio"].iloc[-1]
        if np.isnan(vr):
            return 1.0

        # Above-average volume = more conviction
        if vr > 1.5:
            return 1.2
        elif vr > 1.0:
            return 1.0 + (vr - 1.0) * 0.4
        elif vr < 0.5:
            return 0.7
        else:
            return 0.7 + vr * 0.6

    def _trend_filter(self, df: pd.DataFrame) -> float:
        """
        Longer-term trend filter using 200-day SMA slope.

        Returns +1 if above 200 SMA (uptrend), -1 if below (downtrend).
        Used to avoid counter-trend momentum trades.
        """
        close = df["Close"].values
        if len(close) < 200:
            return 0.0

        sma_200 = np.mean(close[-200:])
        if sma_200 <= 0:
            return 0.0

        return 1.0 if close[-1] > sma_200 else -1.0
