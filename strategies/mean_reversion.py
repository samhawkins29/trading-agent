"""
Mean Reversion / Statistical Arbitrage Strategy.

Inspired by:
  - Renaissance Technologies' core stat-arb approach
  - Pairs trading and Ornstein-Uhlenbeck mean-reversion models
  - Bollinger Band mean-reversion signals

Concept: Prices deviate from their statistical "fair value" and tend to
revert. We identify overbought/oversold conditions using z-scores,
Bollinger Band position, and RSI extremes, then trade the reversion.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Signal:
    """A trading signal from a strategy."""
    symbol: str
    action: str          # "BUY", "SELL", or "HOLD"
    strength: float      # -1.0 (strong sell) to +1.0 (strong buy)
    strategy: str
    reason: str


class MeanReversionStrategy:
    """
    Statistical arbitrage / mean reversion.

    Signals:
      BUY  when price is significantly below its rolling mean (oversold)
      SELL when price is significantly above its rolling mean (overbought)
    """

    name = "mean_reversion"

    def __init__(
        self,
        z_score_window: int = 20,
        z_entry_threshold: float = 2.0,
        z_exit_threshold: float = 0.5,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
    ):
        self.z_window = z_score_window
        self.z_entry = z_entry_threshold
        self.z_exit = z_exit_threshold
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Signal:
        """
        Analyze price data and return a Signal.

        Expects df to have columns: Close, RSI, BB_pct, SMA_20
        (as produced by DataFetcher.compute_indicators).
        """
        if len(df) < self.z_window + 5:
            return Signal(symbol, "HOLD", 0.0, self.name, "Insufficient data")

        close = df["Close"].values
        current = close[-1]

        # ── Z-Score of price relative to rolling mean ────────────────
        rolling_mean = np.mean(close[-self.z_window:])
        rolling_std = np.std(close[-self.z_window:], ddof=1)
        z_score = (
            (current - rolling_mean) / rolling_std
            if rolling_std > 0
            else 0.0
        )

        # ── Bollinger Band position (0 = lower band, 1 = upper band) ─
        bb_pct = df["BB_pct"].iloc[-1] if "BB_pct" in df.columns else 0.5

        # ── RSI ──────────────────────────────────────────────────────
        rsi = df["RSI"].iloc[-1] if "RSI" in df.columns else 50.0

        # ── Composite Signal ─────────────────────────────────────────
        # Each component votes between -1 (sell) and +1 (buy)
        z_signal = np.clip(-z_score / self.z_entry, -1, 1)

        bb_signal = 0.0
        if not np.isnan(bb_pct):
            bb_signal = np.clip(-(bb_pct - 0.5) * 2, -1, 1)

        rsi_signal = 0.0
        if not np.isnan(rsi):
            if rsi < self.rsi_oversold:
                rsi_signal = (self.rsi_oversold - rsi) / self.rsi_oversold
            elif rsi > self.rsi_overbought:
                rsi_signal = -(rsi - self.rsi_overbought) / (
                    100 - self.rsi_overbought
                )

        # Weighted combination
        composite = 0.50 * z_signal + 0.25 * bb_signal + 0.25 * rsi_signal
        composite = np.clip(composite, -1.0, 1.0)

        # ── Decision ─────────────────────────────────────────────────
        if composite > 0.3:
            action = "BUY"
            reason = (
                f"Mean reversion BUY: z={z_score:.2f}, "
                f"BB%={bb_pct:.2f}, RSI={rsi:.1f}"
            )
        elif composite < -0.3:
            action = "SELL"
            reason = (
                f"Mean reversion SELL: z={z_score:.2f}, "
                f"BB%={bb_pct:.2f}, RSI={rsi:.1f}"
            )
        else:
            action = "HOLD"
            reason = f"No clear reversion signal: composite={composite:.3f}"

        return Signal(symbol, action, composite, self.name, reason)

    def detect_pairs_opportunity(
        self, df_a: pd.DataFrame, df_b: pd.DataFrame, window: int = 60
    ) -> Optional[float]:
        """
        Simple pairs-trading z-score between two price series.
        Returns the spread z-score (positive = A overvalued vs B).
        """
        if len(df_a) < window or len(df_b) < window:
            return None

        # Log-price spread
        spread = np.log(df_a["Close"].values[-window:]) - np.log(
            df_b["Close"].values[-window:]
        )
        z = (spread[-1] - np.mean(spread)) / max(np.std(spread, ddof=1), 1e-8)
        return float(z)
