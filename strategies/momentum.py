"""
Trend Following / Momentum Strategy.

Inspired by:
  - Two Sigma's systematic trend-following with alternative data overlays
  - Classic CTA (Commodity Trading Advisor) dual moving-average crossovers
  - Citadel's real-time momentum capture

Concept: Assets in motion tend to stay in motion. We identify strong
trends using moving-average crossovers, MACD, and rate-of-change,
then ride the trend until momentum fades.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategies.mean_reversion import Signal


class MomentumStrategy:
    """
    Trend following / momentum.

    Signals:
      BUY  when multiple momentum indicators align upward
      SELL when momentum indicators align downward
    """

    name = "momentum"

    def __init__(
        self,
        fast_ma: int = 20,
        slow_ma: int = 50,
        roc_period: int = 10,
        adx_threshold: float = 25.0,
    ):
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.roc_period = roc_period
        self.adx_threshold = adx_threshold

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Signal:
        """
        Analyze price data and return a momentum Signal.

        Expects df to have: Close, SMA_20, SMA_50, MACD, MACD_signal,
        Volume, Vol_ratio (from DataFetcher.compute_indicators).
        """
        if len(df) < self.slow_ma + 10:
            return Signal(symbol, "HOLD", 0.0, self.name, "Insufficient data")

        close = df["Close"].values

        # ── Moving Average Crossover ─────────────────────────────────
        sma_fast = df["SMA_20"].iloc[-1] if "SMA_20" in df.columns else np.mean(close[-self.fast_ma:])
        sma_slow = df["SMA_50"].iloc[-1] if "SMA_50" in df.columns else np.mean(close[-self.slow_ma:])

        ma_signal = 0.0
        if sma_slow > 0:
            ma_ratio = (sma_fast - sma_slow) / sma_slow
            ma_signal = np.clip(ma_ratio * 20, -1, 1)  # scale

        # ── MACD Histogram ───────────────────────────────────────────
        macd_signal_val = 0.0
        if "MACD_hist" in df.columns:
            hist = df["MACD_hist"].iloc[-1]
            prev_hist = df["MACD_hist"].iloc[-2] if len(df) > 1 else 0
            # Acceleration: is histogram growing?
            macd_signal_val = np.clip(hist * 10, -1, 1)
            if hist > 0 and hist > prev_hist:
                macd_signal_val = min(macd_signal_val + 0.2, 1.0)
            elif hist < 0 and hist < prev_hist:
                macd_signal_val = max(macd_signal_val - 0.2, -1.0)

        # ── Rate of Change (ROC) ─────────────────────────────────────
        roc = 0.0
        if len(close) > self.roc_period:
            past = close[-self.roc_period - 1]
            roc = (close[-1] - past) / past if past > 0 else 0
        roc_signal = np.clip(roc * 10, -1, 1)

        # ── ADX-like Trend Strength ──────────────────────────────────
        # Simplified: use rolling std of returns as proxy for directional move
        if "returns" in df.columns:
            recent_returns = df["returns"].iloc[-14:].dropna()
            mean_ret = recent_returns.mean()
            trend_strength = abs(mean_ret) / max(recent_returns.std(), 1e-8)
        else:
            trend_strength = 1.0

        # Only generate strong signals when trend is clear
        trend_scalar = min(trend_strength / 2.0, 1.0)

        # ── Volume Confirmation ──────────────────────────────────────
        vol_confirm = 1.0
        if "Vol_ratio" in df.columns:
            vr = df["Vol_ratio"].iloc[-1]
            if not np.isnan(vr):
                vol_confirm = min(vr, 2.0) / 2.0  # normalize 0-1

        # ── Composite Signal ─────────────────────────────────────────
        raw = (
            0.35 * ma_signal
            + 0.30 * macd_signal_val
            + 0.20 * roc_signal
            + 0.15 * (ma_signal * vol_confirm)  # volume-confirmed trend
        )
        composite = np.clip(raw * trend_scalar, -1.0, 1.0)

        # ── Decision ─────────────────────────────────────────────────
        if composite > 0.25:
            action = "BUY"
            reason = (
                f"Momentum BUY: MA_sig={ma_signal:.2f}, "
                f"MACD={macd_signal_val:.2f}, ROC={roc:.4f}"
            )
        elif composite < -0.25:
            action = "SELL"
            reason = (
                f"Momentum SELL: MA_sig={ma_signal:.2f}, "
                f"MACD={macd_signal_val:.2f}, ROC={roc:.4f}"
            )
        else:
            action = "HOLD"
            reason = f"Weak trend: composite={composite:.3f}"

        return Signal(symbol, action, composite, self.name, reason)

    def detect_breakout(
        self, df: pd.DataFrame, lookback: int = 20
    ) -> float:
        """
        Detect price breakout above/below recent range.
        Returns breakout strength: >0 bullish, <0 bearish.
        """
        if len(df) < lookback + 1:
            return 0.0
        recent_high = df["High"].iloc[-lookback - 1:-1].max()
        recent_low = df["Low"].iloc[-lookback - 1:-1].min()
        current = df["Close"].iloc[-1]
        range_size = recent_high - recent_low
        if range_size <= 0:
            return 0.0

        if current > recent_high:
            return min((current - recent_high) / range_size, 1.0)
        elif current < recent_low:
            return max((current - recent_low) / range_size, -1.0)
        return 0.0
