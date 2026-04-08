"""
Statistical Arbitrage / Pairs Trading Strategy.

Redesigned based on research into:
  - Engle-Granger and Johansen cointegration tests
  - Ornstein-Uhlenbeck mean-reversion models with half-life estimation
  - Z-score spread trading with dynamic volatility-adjusted thresholds
  - Renaissance Technologies' pairs-based stat-arb philosophy

WHY IT WORKS:
  Cointegrated assets share a common stochastic trend. When their spread
  deviates from equilibrium, economic forces (arbitrageurs, market makers,
  fundamental linkage) push it back. Historical evidence shows 38-118 bps/month
  excess returns for pairs trading (Gatev et al., 2006). Advanced cointegration
  models with proper half-life estimation achieve Sharpe ratios of 1.5-2.4.

WHEN IT WORKS BEST:
  - Range-bound / sideways markets (low trend strength)
  - Liquid, highly correlated asset pairs (same sector, supply chain)
  - Periods of mean market volatility (not crisis, not dead calm)

WEAKNESSES:
  - Cointegration can break down during structural regime changes
  - Crowding has reduced raw returns from 118 bps to ~38 bps since late 1980s
  - Transaction costs erode thin spreads on short holding periods
  - Pairs may temporarily decouple for weeks before reverting
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    Statistical Arbitrage / Pairs Trading strategy.

    Two modes:
      1. Single-asset mean reversion: Z-score of price relative to rolling
         mean, with dynamic thresholds adjusted by volatility regime.
      2. Pairs trading: Cointegration-based spread trading between
         correlated assets.

    Uses Ornstein-Uhlenbeck half-life to set expected holding periods
    and time-based stops.
    """

    name = "mean_reversion"

    def __init__(
        self,
        z_score_window: int = 20,
        z_entry_threshold: float = 2.0,
        z_exit_threshold: float = 0.5,
        half_life_window: int = 60,
        vol_lookback: int = 60,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
    ):
        self.z_window = z_score_window
        self.z_entry = z_entry_threshold
        self.z_exit = z_exit_threshold
        self.half_life_window = half_life_window
        self.vol_lookback = vol_lookback
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate_signal(self, symbol: str, df: pd.DataFrame) -> Signal:
        """
        Analyze price data and return a mean-reversion Signal.

        Uses volatility-adjusted z-scores with dynamic entry thresholds
        that widen in high-vol regimes and tighten in low-vol regimes.
        Incorporates half-life estimation and RSI confirmation.
        """
        if len(df) < max(self.z_window, self.vol_lookback) + 10:
            return Signal(symbol, "HOLD", 0.0, self.name, "Insufficient data")

        close = df["Close"].values

        # -- Volatility regime detection --
        vol_regime = self._volatility_regime(df)

        # -- Dynamic z-score thresholds based on vol regime --
        if vol_regime == "high":
            z_entry = self.z_entry * 1.5   # Wider thresholds in high vol
            z_exit = self.z_exit * 1.5
        elif vol_regime == "low":
            z_entry = self.z_entry * 0.75  # Tighter in low vol
            z_exit = self.z_exit * 0.75
        else:
            z_entry = self.z_entry
            z_exit = self.z_exit

        # -- Z-score of price vs rolling mean --
        rolling_mean = np.mean(close[-self.z_window:])
        rolling_std = np.std(close[-self.z_window:], ddof=1)
        z_score = (
            (close[-1] - rolling_mean) / rolling_std
            if rolling_std > 0 else 0.0
        )

        # -- Half-life estimation (Ornstein-Uhlenbeck) --
        half_life = self._estimate_half_life(close)

        # -- RSI confirmation --
        rsi = df["RSI"].iloc[-1] if "RSI" in df.columns else 50.0

        # -- Bollinger Band position --
        bb_pct = df["BB_pct"].iloc[-1] if "BB_pct" in df.columns else 0.5

        # -- Component signals --
        # Z-score signal (primary): mean-reversion direction
        z_signal = 0.0
        if abs(z_score) > z_entry:
            z_signal = np.clip(-z_score / z_entry, -1.0, 1.0)

        # BB signal
        bb_signal = 0.0
        if not np.isnan(bb_pct):
            if bb_pct < 0.05:
                bb_signal = 0.8   # Strong oversold at lower band
            elif bb_pct > 0.95:
                bb_signal = -0.8  # Strong overbought at upper band
            else:
                bb_signal = np.clip(-(bb_pct - 0.5) * 2, -1.0, 1.0)

        # RSI signal
        rsi_signal = 0.0
        if not np.isnan(rsi):
            if rsi < self.rsi_oversold:
                rsi_signal = (self.rsi_oversold - rsi) / self.rsi_oversold
            elif rsi > self.rsi_overbought:
                rsi_signal = -(rsi - self.rsi_overbought) / (100 - self.rsi_overbought)

        # Half-life confidence: stronger signal if half-life is reasonable
        hl_confidence = 1.0
        if half_life is not None:
            if 2 <= half_life <= 30:
                hl_confidence = 1.2  # Good half-life range for daily data
            elif half_life > 60:
                hl_confidence = 0.5  # Too slow to revert
            elif half_life < 1:
                hl_confidence = 0.7  # Too noisy

        # -- Composite signal --
        composite = (
            0.45 * z_signal
            + 0.25 * bb_signal
            + 0.20 * rsi_signal
        ) * hl_confidence

        # Reduce conviction in high-vol regime (spreads can widen further)
        if vol_regime == "high":
            composite *= 0.7

        composite = np.clip(composite, -1.0, 1.0)

        # -- Decision --
        hl_str = f"{half_life:.1f}d" if half_life is not None else "N/A"
        if composite > 0.25:
            action = "BUY"
            reason = (
                f"StatArb BUY: z={z_score:.2f} (threshold={z_entry:.1f}), "
                f"BB%={bb_pct:.2f}, RSI={rsi:.1f}, "
                f"half_life={hl_str}, vol_regime={vol_regime}"
            )
        elif composite < -0.25:
            action = "SELL"
            reason = (
                f"StatArb SELL: z={z_score:.2f} (threshold={z_entry:.1f}), "
                f"BB%={bb_pct:.2f}, RSI={rsi:.1f}, "
                f"half_life={hl_str}, vol_regime={vol_regime}"
            )
        else:
            action = "HOLD"
            reason = (
                f"No MR signal: composite={composite:.3f}, z={z_score:.2f}, "
                f"vol_regime={vol_regime}"
            )

        return Signal(symbol, action, composite, self.name, reason)

    # -- Pairs Trading --

    def detect_pairs_opportunity(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        window: int = 60,
    ) -> Optional[Dict]:
        """
        Cointegration-based pairs trading signal.

        Uses Engle-Granger two-step method:
          1. Regress log(A) on log(B) to get hedge ratio
          2. Test residuals for stationarity (ADF test)
          3. If cointegrated, compute spread z-score

        Returns dict with z_score, hedge_ratio, half_life, is_cointegrated.
        """
        if len(df_a) < window or len(df_b) < window:
            return None

        log_a = np.log(df_a["Close"].values[-window:])
        log_b = np.log(df_b["Close"].values[-window:])

        # Step 1: OLS regression to find hedge ratio
        # log_a = alpha + beta * log_b + epsilon
        X = np.column_stack([np.ones(len(log_b)), log_b])
        try:
            beta = np.linalg.lstsq(X, log_a, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None

        hedge_ratio = beta[1]
        spread = log_a - beta[0] - hedge_ratio * log_b

        # Step 2: ADF test on residuals (simplified)
        is_cointegrated = self._simple_adf_test(spread)

        # Step 3: Z-score of spread
        spread_mean = np.mean(spread)
        spread_std = np.std(spread, ddof=1)
        z = (spread[-1] - spread_mean) / max(spread_std, 1e-8)

        # Half-life of spread
        half_life = self._estimate_half_life(spread)

        return {
            "z_score": float(z),
            "hedge_ratio": float(hedge_ratio),
            "half_life": float(half_life) if half_life else None,
            "is_cointegrated": is_cointegrated,
            "spread_std": float(spread_std),
        }

    # -- Helper Methods --

    def _estimate_half_life(self, series: np.ndarray) -> Optional[float]:
        """
        Estimate mean-reversion half-life using Ornstein-Uhlenbeck model.

        Fits: delta_y = theta * (y_lag - mean) + noise
        Half-life = -ln(2) / ln(1 + theta)

        Returns half-life in periods (days), or None if not mean-reverting.
        """
        if len(series) < 20:
            return None

        y = series - np.mean(series)
        y_lag = y[:-1]
        delta_y = np.diff(y)

        if len(y_lag) == 0 or np.std(y_lag) < 1e-10:
            return None

        # OLS: delta_y = theta * y_lag
        theta = np.dot(y_lag, delta_y) / np.dot(y_lag, y_lag)

        if theta >= 0:
            return None  # Not mean-reverting

        half_life = -np.log(2) / np.log(1 + theta)
        return max(half_life, 0.1)

    def _volatility_regime(self, df: pd.DataFrame) -> str:
        """
        Classify current volatility regime as 'low', 'normal', or 'high'.

        Uses ratio of recent vol to longer-term vol.
        """
        if "returns" not in df.columns or len(df) < self.vol_lookback:
            return "normal"

        returns = df["returns"].dropna()
        if len(returns) < self.vol_lookback:
            return "normal"

        recent_vol = returns.iloc[-20:].std()
        historical_vol = returns.iloc[-self.vol_lookback:].std()

        if historical_vol < 1e-10:
            return "normal"

        vol_ratio = recent_vol / historical_vol

        if vol_ratio > 1.5:
            return "high"
        elif vol_ratio < 0.7:
            return "low"
        return "normal"

    @staticmethod
    def _simple_adf_test(series: np.ndarray, threshold: float = -2.86) -> bool:
        """
        Simplified ADF test for stationarity.

        Fits: delta_y = rho * y_lag + intercept + noise
        Tests if rho is significantly negative (series is stationary).

        Uses critical value of -2.86 for 5% significance (approx N=100).
        """
        if len(series) < 20:
            return False

        y = series[1:]
        y_lag = series[:-1]
        delta_y = np.diff(series)

        X = np.column_stack([y_lag, np.ones(len(y_lag))])
        try:
            coeffs = np.linalg.lstsq(X, delta_y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return False

        rho = coeffs[0]

        # Estimate standard error
        residuals = delta_y - X @ coeffs
        se = np.sqrt(np.sum(residuals**2) / (len(residuals) - 2))
        x_var = np.sum((y_lag - np.mean(y_lag))**2)
        if x_var < 1e-10:
            return False
        se_rho = se / np.sqrt(x_var)

        t_stat = rho / se_rho if se_rho > 0 else 0

        return t_stat < threshold
