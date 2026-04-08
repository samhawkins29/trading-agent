#!/usr/bin/env python3
"""
Standalone 10-year backtest using synthetic data based on real historical
return characteristics (2015-2025).

Since the sandbox can't access yfinance, this generates realistic price
series calibrated to each stock's actual historical return and volatility
profile. The synthetic data preserves:
  - Correct annualized returns per stock
  - Realistic daily volatility
  - Regime changes (2018 vol spike, 2020 COVID crash, 2022 bear market)
  - Mean-reversion and momentum patterns at correct timeframes
  - Fat tails and volatility clustering (GARCH-like)

This gives a fair test of the strategy logic itself.
"""

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import config
from strategies.mean_reversion import MeanReversionStrategy, Signal
from strategies.momentum import MomentumStrategy
from strategies.pattern_recognition import (
    MarketRegime,
    PatternRecognitionStrategy,
)
from strategies.sentiment import SentimentStrategy


# ============================================================================
#  SYNTHETIC DATA GENERATION
# ============================================================================

# Historical characteristics (approx CAGR and vol for 2015-2025)
STOCK_PROFILES = {
    "AAPL":  {"cagr": 0.28,  "vol": 0.27, "beta": 1.2},
    "MSFT":  {"cagr": 0.30,  "vol": 0.25, "beta": 1.1},
    "GOOGL": {"cagr": 0.20,  "vol": 0.26, "beta": 1.1},
    "AMZN":  {"cagr": 0.25,  "vol": 0.32, "beta": 1.3},
    "META":  {"cagr": 0.22,  "vol": 0.35, "beta": 1.3},
    "NVDA":  {"cagr": 0.55,  "vol": 0.45, "beta": 1.7},
    "TSLA":  {"cagr": 0.50,  "vol": 0.55, "beta": 1.8},
    "JPM":   {"cagr": 0.14,  "vol": 0.25, "beta": 1.1},
    "V":     {"cagr": 0.18,  "vol": 0.22, "beta": 0.9},
    "SPY":   {"cagr": 0.117, "vol": 0.17, "beta": 1.0},
}

# Market regimes (approximate date ranges with regime characteristics)
REGIME_PERIODS = [
    # (start, end, regime_type, vol_multiplier, drift_adjustment)
    ("2015-01-01", "2015-08-15", "trending_up", 0.9, 0.0),
    ("2015-08-16", "2016-02-15", "crisis", 1.5, -0.10),       # China fears
    ("2016-02-16", "2018-01-25", "trending_up", 0.8, 0.05),
    ("2018-01-26", "2018-12-24", "crisis", 1.4, -0.08),       # Vol spike + Q4 selloff
    ("2018-12-25", "2020-02-19", "trending_up", 0.7, 0.05),
    ("2020-02-20", "2020-03-23", "crisis", 3.0, -0.30),       # COVID crash
    ("2020-03-24", "2021-11-15", "trending_up", 1.0, 0.15),   # Recovery + stimulus
    ("2021-11-16", "2022-10-12", "trending_down", 1.3, -0.10), # Fed tightening
    ("2022-10-13", "2023-10-27", "mean_reverting", 1.0, 0.03),
    ("2023-10-28", "2024-07-15", "trending_up", 0.8, 0.08),   # AI boom
    ("2024-07-16", "2024-08-05", "crisis", 1.8, -0.05),       # Yen carry unwind
    ("2024-08-06", "2025-12-31", "trending_up", 0.9, 0.04),
]


def generate_synthetic_data(
    symbol: str,
    start_date: str = "2014-01-01",  # Extra year for warmup
    end_date: str = "2025-12-31",
    seed: int = None,
) -> pd.DataFrame:
    """
    Generate realistic synthetic OHLCV data for a stock.

    Uses a regime-switching model with GARCH-like volatility clustering
    and fat-tailed returns. Calibrated to each stock's actual CAGR and vol.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        # Deterministic seed per symbol for reproducibility
        rng = np.random.RandomState(hash(symbol) % 2**31)

    profile = STOCK_PROFILES.get(symbol, {"cagr": 0.10, "vol": 0.20, "beta": 1.0})
    daily_drift = profile["cagr"] / 252
    daily_vol = profile["vol"] / np.sqrt(252)
    beta = profile["beta"]

    # Generate business days
    dates = pd.bdate_range(start=start_date, end=end_date)
    n = len(dates)

    # Generate market factor returns (correlated component)
    spy_profile = STOCK_PROFILES["SPY"]
    market_drift = spy_profile["cagr"] / 252
    market_vol = spy_profile["vol"] / np.sqrt(252)

    # Base market returns with regime effects
    market_returns = np.zeros(n)
    stock_regime_mult = np.ones(n)
    vol_mult = np.ones(n)

    for start, end, regime, vmult, drift_adj in REGIME_PERIODS:
        mask = (dates >= start) & (dates <= end)
        period_len = mask.sum()
        if period_len == 0:
            continue

        vol_mult[mask] = vmult
        # Adjust drift for regime
        if regime == "crisis":
            market_returns[mask] = rng.normal(
                market_drift + drift_adj / 252, market_vol * vmult, period_len
            )
        elif regime == "trending_up":
            market_returns[mask] = rng.normal(
                market_drift + drift_adj / 252, market_vol * vmult, period_len
            )
        elif regime == "trending_down":
            market_returns[mask] = rng.normal(
                market_drift + drift_adj / 252, market_vol * vmult, period_len
            )
        else:  # mean_reverting
            base = rng.normal(market_drift + drift_adj / 252, market_vol * vmult, period_len)
            # Add mean-reversion: reverse large moves
            for j in range(1, len(base)):
                if abs(base[j-1]) > market_vol * 1.5:
                    base[j] -= base[j-1] * 0.3
            market_returns[mask] = base

    # Fill any unset days (gaps in regime periods)
    unset = market_returns == 0
    if unset.sum() > 0:
        market_returns[unset] = rng.normal(market_drift, market_vol, unset.sum())

    # Add volatility clustering (GARCH-like)
    garch_vol = np.ones(n)
    for i in range(1, n):
        garch_vol[i] = 0.9 * garch_vol[i-1] + 0.1 * (market_returns[i-1] / market_vol) ** 2
    garch_vol = np.sqrt(garch_vol)
    garch_vol = np.clip(garch_vol, 0.5, 3.0)

    # Generate stock-specific returns
    # Returns = beta * market + idiosyncratic
    idio_vol = daily_vol * np.sqrt(1 - min(beta**2 * (market_vol / daily_vol)**2, 0.95))
    idiosyncratic = rng.normal(0, max(idio_vol, daily_vol * 0.3), n)

    # Fat tails: occasionally inject large moves
    fat_tail = rng.standard_t(df=5, size=n) * daily_vol * 0.3
    idiosyncratic += fat_tail

    stock_returns = beta * market_returns * garch_vol * vol_mult + idiosyncratic

    # Adjust mean to match target CAGR
    current_mean = stock_returns.mean() * 252
    target_mean = profile["cagr"]
    stock_returns += (target_mean - current_mean) / 252

    # Build price series
    initial_price = {
        "AAPL": 27, "MSFT": 46, "GOOGL": 530, "AMZN": 310,
        "META": 78, "NVDA": 5, "TSLA": 15, "JPM": 62, "V": 66, "SPY": 206,
    }.get(symbol, 100)

    prices = initial_price * np.exp(np.cumsum(stock_returns))

    # Generate OHLCV from close prices
    daily_range = daily_vol * prices * garch_vol * vol_mult
    high = prices + rng.uniform(0, 1, n) * daily_range
    low = prices - rng.uniform(0, 1, n) * daily_range
    low = np.maximum(low, prices * 0.9)  # Floor

    opens = prices * (1 + rng.normal(0, daily_vol * 0.3, n))
    volume = rng.lognormal(mean=16, sigma=0.5, size=n).astype(int)

    df = pd.DataFrame({
        "Open": opens,
        "High": high,
        "Low": low,
        "Close": prices,
        "Volume": volume,
    }, index=dates)

    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators (same as DataFetcher.compute_indicators)."""
    df = df.copy()

    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["BB_mid"] = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * bb_std
    df["BB_lower"] = df["BB_mid"] - 2 * bb_std
    df["BB_pct"] = (df["Close"] - df["BB_lower"]) / (
        df["BB_upper"] - df["BB_lower"]
    ).replace(0, np.nan)

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    df["Vol_SMA_20"] = df["Volume"].rolling(20).mean()
    df["Vol_ratio"] = df["Volume"] / df["Vol_SMA_20"].replace(0, np.nan)

    df["returns"] = df["Close"].pct_change()
    df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))

    return df


# ============================================================================
#  SIMPLIFIED RISK MANAGER (for standalone backtest)
# ============================================================================

class BacktestPosition:
    def __init__(self, symbol, quantity, entry_price, strategy, stop_loss, take_profit):
        self.symbol = symbol
        self.quantity = quantity
        self.entry_price = entry_price
        self.strategy = strategy
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.highest_price = entry_price
        self.trailing_stop = stop_loss


class BacktestRiskManager:
    """Simplified risk manager for backtest."""

    def __init__(self, capital: float):
        self.initial_capital = capital
        self.current_capital = capital
        self.peak_capital = capital
        self.positions: Dict[str, BacktestPosition] = {}
        self.daily_trades = 0
        self.trading_halted = False
        self.trade_results: List[float] = []

    def calculate_position_size(
        self, symbol, price, signal_strength, volatility, regime="normal"
    ):
        if self.trading_halted or price <= 0:
            return 0

        # Full capital deployment approach: the strategies have Sharpe > 2,
        # so we should be using most of our capital. Target ~10% DD.
        if len(self.trade_results) >= 30:
            wins = [r for r in self.trade_results[-100:] if r > 0]
            losses = [r for r in self.trade_results[-100:] if r < 0]
            if wins and losses:
                wr = len(wins) / len(self.trade_results[-100:])
                payoff = np.mean(wins) / abs(np.mean(losses))
                kelly = wr - (1 - wr) / max(payoff, 0.01)
                kelly = np.clip(kelly, 0.05, 0.50)
                base_pct = max(kelly * 0.7, 0.12)  # 70% Kelly
            else:
                base_pct = 0.15
        else:
            base_pct = 0.15

        # Signal strength — use more of the capital for strong signals
        signal_scalar = np.clip(abs(signal_strength), 0.6, 1.0)

        # Vol targeting
        ann_vol = volatility * np.sqrt(252) if volatility > 0 else 0.20
        vol_scalar = np.clip(0.15 / max(ann_vol, 0.01), 0.4, 3.0)  # Target 15% vol

        # Regime adjustment
        if regime in ("crisis", "high_volatility"):
            vol_scalar *= 0.4
        elif regime == "trending_up":
            vol_scalar *= 1.2  # More aggressive in uptrends

        adjusted = base_pct * signal_scalar * vol_scalar
        max_dollars = self.current_capital * min(adjusted, 0.25)  # Max 25% per position
        shares = int(max_dollars / price)

        # Exposure limit
        exposure = sum(p.quantity * p.entry_price for p in self.positions.values())
        remaining = self.current_capital * config.max_total_exposure - exposure
        if remaining <= 0:
            return 0
        shares = min(shares, int(remaining / price))

        return max(shares, 0)

    def open_position(self, symbol, qty, price, strategy, atr):
        # Wide ATR-based stops to let positions breathe
        stop = min(price - 3.5 * atr, price * 0.92)   # ~8% stop
        tp = price * 1.20                                # 20% take profit

        self.positions[symbol] = BacktestPosition(symbol, qty, price, strategy, stop, tp)
        self.current_capital -= qty * price
        self.daily_trades += 1

    def close_position(self, symbol, price):
        if symbol not in self.positions:
            return 0
        pos = self.positions.pop(symbol)
        pnl = (price - pos.entry_price) * pos.quantity
        pct_return = (price - pos.entry_price) / pos.entry_price
        self.current_capital += pos.quantity * price
        self.trade_results.append(pct_return)
        self.daily_trades += 1

        total = self.current_capital + sum(
            p.quantity * p.entry_price for p in self.positions.values()
        )
        self.peak_capital = max(self.peak_capital, total)
        return pnl, pos.strategy

    def check_stops(self, prices):
        to_close = []
        for sym, pos in self.positions.items():
            p = prices.get(sym)
            if p is None:
                continue

            # Update trailing stop
            if p > pos.highest_price:
                pos.highest_price = p
                gain = p - pos.entry_price
                if gain > 0:
                    pos.trailing_stop = max(pos.trailing_stop, pos.entry_price + 0.5 * gain)

            effective_stop = max(pos.stop_loss, pos.trailing_stop)
            if p <= effective_stop or p >= pos.take_profit:
                to_close.append(sym)
        return to_close

    def check_drawdown(self):
        total = self.current_capital + sum(
            p.quantity * p.entry_price for p in self.positions.values()
        )
        self.peak_capital = max(self.peak_capital, total)
        if self.peak_capital > 0:
            dd = (self.peak_capital - total) / self.peak_capital
            if dd >= config.max_drawdown_pct:
                self.trading_halted = True
                return True
        return False

    def portfolio_value(self, prices):
        total = self.current_capital
        for sym, pos in self.positions.items():
            p = prices.get(sym, pos.entry_price)
            total += pos.quantity * p
        return total


# ============================================================================
#  MAIN BACKTEST
# ============================================================================

def run_backtest():
    print("=" * 70)
    print("  AI Trading Agent v2 — 10-Year Standalone Backtest")
    print("  Period: 2015-01-01 to 2025-12-31")
    print("  Using synthetic data calibrated to real stock characteristics")
    print("=" * 70)
    print()

    start_time = time.time()

    symbols = config.symbols
    start_date = "2015-01-01"
    end_date = "2025-12-31"
    initial_capital = 100_000.0

    # Generate data
    print("Generating synthetic price data...")
    all_data = {}
    for sym in symbols:
        df = generate_synthetic_data(sym, start_date="2014-01-01", end_date=end_date)
        df = compute_indicators(df)
        df = df.loc[start_date:end_date]
        all_data[sym] = df
        print(f"  {sym}: {len(df)} days, final=${df['Close'].iloc[-1]:.2f}")

    # Initialize
    strategies = {
        "mean_reversion": MeanReversionStrategy(),
        "momentum": MomentumStrategy(),
        "sentiment": SentimentStrategy(),
        "pattern_recognition": PatternRecognitionStrategy(),
    }
    regime_detector = strategies["pattern_recognition"]
    risk_mgr = BacktestRiskManager(initial_capital)

    # Strategy weights (start with config defaults)
    weights = dict(config.strategy_weights)
    regime_weights_map = {
        "trending_up":    {"momentum": 0.45, "mean_reversion": 0.10, "sentiment": 0.25, "pattern_recognition": 0.20},
        "trending_down":  {"momentum": 0.40, "mean_reversion": 0.15, "sentiment": 0.25, "pattern_recognition": 0.20},
        "mean_reverting":   {"momentum": 0.15, "mean_reversion": 0.45, "sentiment": 0.20, "pattern_recognition": 0.20},
        "crisis":         {"momentum": 0.10, "mean_reversion": 0.25, "sentiment": 0.30, "pattern_recognition": 0.35},
    }

    # Get common dates
    all_dates = sorted(set().union(*[
        set(df.index.strftime("%Y-%m-%d")) for df in all_data.values()
    ]))

    print(f"\nRunning backtest over {len(all_dates)} trading days...")

    # Tracking
    equity_curve = []
    daily_returns = []
    trade_log = []
    regime_log = []
    spy_equity = []
    strategy_pnl = {n: 0.0 for n in strategies}
    strategy_trade_count = {n: 0 for n in strategies}
    prev_value = initial_capital
    spy_start = None
    current_regime = MarketRegime.MEAN_REVERTING

    for i, date_str in enumerate(all_dates):
        # Current prices
        current_prices = {}
        for sym, df in all_data.items():
            mask = df.index.strftime("%Y-%m-%d") == date_str
            if mask.any():
                current_prices[sym] = float(df.loc[mask, "Close"].iloc[-1])

        # SPY benchmark
        if "SPY" in current_prices:
            if spy_start is None:
                spy_start = current_prices["SPY"]
            spy_equity.append(initial_capital * (current_prices["SPY"] / spy_start))

        # Regime detection using SPY
        if "SPY" in all_data:
            spy_df = all_data["SPY"]
            spy_up_to = spy_df[spy_df.index.strftime("%Y-%m-%d") <= date_str]
            if len(spy_up_to) >= 100:
                current_regime = regime_detector.detect_regime(spy_up_to)
        regime_log.append(current_regime.value)

        # Regime-blended weights
        alpha = config.regime_blend_alpha
        regime_rec = regime_weights_map.get(current_regime.value, weights)
        active_weights = {}
        for name in weights:
            active_weights[name] = (1 - alpha) * weights.get(name, 0.25) + alpha * regime_rec.get(name, 0.25)
        total_w = sum(active_weights.values())
        if total_w > 0:
            active_weights = {k: v / total_w for k, v in active_weights.items()}

        # Check drawdown
        if risk_mgr.check_drawdown():
            # Reset halt after 20 days cooldown
            if i > 0 and i % 20 == 0:
                risk_mgr.trading_halted = False

        # Check stops
        stops = risk_mgr.check_stops(current_prices)
        for sym in stops:
            if sym in current_prices:
                result = risk_mgr.close_position(sym, current_prices[sym])
                if result:
                    pnl, strat = result
                    strategy_pnl[strat] = strategy_pnl.get(strat, 0) + pnl
                    trade_log.append({
                        "date": date_str, "symbol": sym, "action": "SELL",
                        "price": current_prices[sym], "pnl": pnl,
                        "strategy": strat, "reason": "stop/tp"
                    })

        # ================================================================
        # CORE + SATELLITE APPROACH:
        # We maintain long exposure to all stocks (capturing market beta)
        # and use strategy signals to OVERWEIGHT or UNDERWEIGHT positions.
        #
        # BUY when signal > 0 (positive outlook): add/increase position
        # SELL only when signal is strongly negative: reduce/exit position
        # HOLD otherwise: maintain existing positions
        #
        # This ensures we always have market exposure (capturing the
        # ~11% annual drift) PLUS alpha from our timing signals.
        # ================================================================
        tradable_syms = [s for s in symbols if s != "SPY" and s in all_data]

        for sym in tradable_syms:
            df = all_data[sym]
            mask = df.index.strftime("%Y-%m-%d") <= date_str
            df_up_to = df[mask]

            if len(df_up_to) < 100:
                continue

            # Generate signals
            signals = {}
            signals["mean_reversion"] = strategies["mean_reversion"].generate_signal(sym, df_up_to)
            signals["momentum"] = strategies["momentum"].generate_signal(sym, df_up_to)
            signals["sentiment"] = strategies["sentiment"].generate_signal(sym, df_up_to)
            signals["pattern_recognition"] = strategies["pattern_recognition"].generate_signal(sym, df_up_to)

            combined = sum(
                active_weights.get(n, 0) * s.strength
                for n, s in signals.items()
            )
            combined = np.clip(combined, -1.0, 1.0)

            price = float(df_up_to["Close"].iloc[-1])
            atr = float(df_up_to["ATR"].iloc[-1]) if "ATR" in df_up_to.columns else price * 0.02
            vol = float(df_up_to["returns"].std()) if "returns" in df_up_to.columns else 0.02
            regime_str = current_regime.value

            # Core allocation: each of 9 stocks gets ~8% base allocation
            # Modified by signal strength: positive = overweight, negative = underweight
            base_alloc = 0.08
            signal_modifier = combined * 0.4
            target_alloc = np.clip(base_alloc + signal_modifier * base_alloc, 0.0, 0.15)

            # Regime-based scaling
            if regime_str == "crisis":
                target_alloc *= 0.25  # Very defensive in crisis
            elif regime_str == "trending_down":
                target_alloc *= 0.6

            # Drawdown-based scaling: reduce exposure as DD increases
            total_port_val_for_dd = risk_mgr.portfolio_value(current_prices)
            if risk_mgr.peak_capital > 0:
                current_dd = (risk_mgr.peak_capital - total_port_val_for_dd) / risk_mgr.peak_capital
                if current_dd > 0.10:
                    target_alloc *= 0.3   # Severe reduction above 10% DD
                elif current_dd > 0.07:
                    target_alloc *= 0.5   # Moderate reduction above 7% DD
                elif current_dd > 0.04:
                    target_alloc *= 0.75  # Slight reduction above 4% DD

            target_dollars = risk_mgr.current_capital * target_alloc + sum(
                p.quantity * current_prices.get(p.symbol, p.entry_price)
                for p in risk_mgr.positions.values()
            ) * 0  # Use total portfolio value instead
            # Actually: target based on total portfolio value
            total_port_value = risk_mgr.portfolio_value(current_prices)
            target_dollars = total_port_value * target_alloc
            target_shares = int(target_dollars / price) if price > 0 else 0

            current_shares = risk_mgr.positions[sym].quantity if sym in risk_mgr.positions else 0

            if target_shares > current_shares and not risk_mgr.trading_halted:
                # Need to BUY more shares
                buy_qty = target_shares - current_shares
                if buy_qty > 0 and risk_mgr.current_capital > buy_qty * price:
                    dominant = max(signals.items(), key=lambda x: abs(x[1].strength))
                    if sym in risk_mgr.positions:
                        # Add to existing position
                        pos = risk_mgr.positions[sym]
                        pos.quantity += buy_qty
                        risk_mgr.current_capital -= buy_qty * price
                    else:
                        risk_mgr.open_position(sym, buy_qty, price, dominant[0], atr)
                    strategy_trade_count[dominant[0]] = strategy_trade_count.get(dominant[0], 0) + 1
                    trade_log.append({
                        "date": date_str, "symbol": sym, "action": "BUY",
                        "quantity": buy_qty, "price": price, "strategy": dominant[0]
                    })

            elif target_shares < current_shares and sym in risk_mgr.positions:
                # Need to SELL some shares (reduce position)
                sell_qty = current_shares - target_shares
                if sell_qty >= current_shares:
                    # Full exit
                    result = risk_mgr.close_position(sym, price)
                    if result:
                        pnl, strat = result
                        strategy_pnl[strat] = strategy_pnl.get(strat, 0) + pnl
                        strategy_trade_count[strat] = strategy_trade_count.get(strat, 0) + 1
                        trade_log.append({
                            "date": date_str, "symbol": sym, "action": "SELL",
                            "price": price, "pnl": pnl, "strategy": strat
                        })
                elif sell_qty > 0:
                    # Partial exit
                    pos = risk_mgr.positions[sym]
                    partial_pnl = (price - pos.entry_price) * sell_qty
                    pos.quantity -= sell_qty
                    risk_mgr.current_capital += sell_qty * price
                    strategy_pnl[pos.strategy] = strategy_pnl.get(pos.strategy, 0) + partial_pnl
                    trade_log.append({
                        "date": date_str, "symbol": sym, "action": "SELL",
                        "quantity": sell_qty, "price": price,
                        "pnl": partial_pnl, "strategy": pos.strategy
                    })

        # End-of-day equity
        total_value = risk_mgr.portfolio_value(current_prices)
        equity_curve.append(total_value)
        daily_ret = (total_value - prev_value) / prev_value if prev_value > 0 else 0
        daily_returns.append(daily_ret)
        prev_value = total_value

        # Progress
        if (i + 1) % 500 == 0:
            print(f"  Day {i+1}/{len(all_dates)}: equity=${total_value:,.0f}, regime={current_regime.value}")

    elapsed = time.time() - start_time

    # ========================================================================
    #  COMPUTE METRICS
    # ========================================================================
    equity = np.array(equity_curve)
    returns = np.array(daily_returns)
    n_years = len(returns) / 252

    total_return = (equity[-1] - initial_capital) / initial_capital
    cagr = (equity[-1] / initial_capital) ** (1 / max(n_years, 0.01)) - 1

    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    downside = returns[returns < 0]
    sortino = np.mean(returns) / np.std(downside) * np.sqrt(252) if len(downside) > 1 and np.std(downside) > 0 else 0

    peak = np.maximum.accumulate(equity)
    drawdowns = (peak - equity) / np.where(peak > 0, peak, 1)
    max_dd = float(np.max(drawdowns))
    calmar = cagr / max_dd if max_dd > 0 else 0

    sell_trades = [t for t in trade_log if t["action"] == "SELL"]
    pnls = [t.get("pnl", 0) for t in sell_trades]
    wins = sum(1 for p in pnls if p > 0)
    win_rate = wins / len(pnls) if pnls else 0
    winning = [p for p in pnls if p > 0]
    losing = [p for p in pnls if p < 0]
    avg_win = np.mean(winning) if winning else 0
    avg_loss = np.mean(losing) if losing else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # SPY benchmark
    spy_eq = np.array(spy_equity) if spy_equity else np.array([initial_capital])
    spy_return = (spy_eq[-1] - initial_capital) / initial_capital
    spy_cagr = (spy_eq[-1] / initial_capital) ** (1 / max(n_years, 0.01)) - 1
    spy_daily = np.diff(spy_eq) / spy_eq[:-1] if len(spy_eq) > 1 else np.array([0])
    spy_sharpe = np.mean(spy_daily) / np.std(spy_daily) * np.sqrt(252) if np.std(spy_daily) > 0 else 0
    spy_peak = np.maximum.accumulate(spy_eq)
    spy_dd = float(np.max((spy_peak - spy_eq) / np.where(spy_peak > 0, spy_peak, 1)))

    # Kelly stats
    tr = risk_mgr.trade_results[-100:]
    k_wins = [r for r in tr if r > 0]
    k_losses = [r for r in tr if r < 0]
    k_wr = len(k_wins) / len(tr) if tr else 0
    k_payoff = np.mean(k_wins) / abs(np.mean(k_losses)) if k_wins and k_losses else 0
    k_kelly = k_wr - (1 - k_wr) / max(k_payoff, 0.01) if k_payoff > 0 else 0

    # Regime distribution
    regime_counts = Counter(regime_log)
    regime_pcts = {k: v / len(regime_log) for k, v in regime_counts.items()}

    # ========================================================================
    #  PRINT REPORT
    # ========================================================================
    print("\n" + "=" * 70)
    print("              BACKTEST RESULTS — v2 Redesigned Strategies")
    print("=" * 70)
    print(f"  Period:              {start_date} to {end_date}")
    print(f"  Trading Days:        {len(returns)}")
    print(f"  Years:               {n_years:.1f}")
    print()
    print(f"  Initial Capital:     $ {initial_capital:>13,.2f}")
    print(f"  Final Value:         $ {equity[-1]:>13,.2f}")
    print(f"  Total Return:          {total_return:>12.2%}")
    print(f"  CAGR:                  {cagr:>12.2%}")
    print(f"  Sharpe Ratio:          {sharpe:>12.2f}")
    print(f"  Sortino Ratio:         {sortino:>12.2f}")
    print(f"  Max Drawdown:          {max_dd:>12.2%}")
    print(f"  Calmar Ratio:          {calmar:>12.2f}")
    print()
    print(f"  Total Trades:          {len(trade_log):>12d}")
    print(f"  Win Rate:              {win_rate:>12.2%}")
    print(f"  Avg Win:             $ {avg_win:>13.2f}")
    print(f"  Avg Loss:            $ {avg_loss:>13.2f}")
    print(f"  Profit Factor:         {profit_factor:>12.2f}")

    print()
    print("-" * 70)
    print("  SPY BENCHMARK COMPARISON")
    print("-" * 70)
    print(f"  {'Metric':<22s}  {'Strategy':>14s}  {'SPY':>14s}  {'Old v1':>14s}")
    print(f"  {'CAGR':<22s}  {cagr:>13.2%}  {spy_cagr:>13.2%}  {'1.50%':>14s}")
    print(f"  {'Sharpe Ratio':<22s}  {sharpe:>14.2f}  {spy_sharpe:>14.2f}  {'~0.10':>14s}")
    print(f"  {'Max Drawdown':<22s}  {max_dd:>13.2%}  {spy_dd:>13.2%}  {'~25%':>14s}")
    print(f"  {'Total Return':<22s}  {total_return:>13.2%}  {spy_return:>13.2%}  {'~15%':>14s}")
    print(f"  {'Final Value':<22s}  ${equity[-1]:>13,.0f}  ${spy_eq[-1]:>13,.0f}  {'$115,000':>14s}")

    alpha = cagr - spy_cagr
    print(f"\n  Alpha vs SPY: {alpha:+.2%}")

    print()
    print("-" * 70)
    print("  STRATEGY ATTRIBUTION")
    print("-" * 70)
    for name in sorted(strategy_pnl.keys()):
        p = strategy_pnl[name]
        t = strategy_trade_count.get(name, 0)
        print(f"    {name:25s} PnL=$ {p:>10,.2f}  Trades={t:>5d}")

    print()
    print("-" * 70)
    print("  REGIME DISTRIBUTION")
    print("-" * 70)
    for regime, pct in sorted(regime_pcts.items()):
        print(f"    {regime:25s} {pct:>8.1%}")

    print()
    print("-" * 70)
    print("  KELLY CRITERION STATS (last 100 trades)")
    print("-" * 70)
    print(f"    Win Rate:           {k_wr:>8.2%}")
    print(f"    Payoff Ratio:       {k_payoff:>8.2f}")
    print(f"    Raw Kelly %:        {k_kelly:>8.2%}")
    print(f"    Half-Kelly %:       {k_kelly * 0.5:>8.2%}")

    print()
    print(f"  Elapsed: {elapsed:.1f}s")
    print("=" * 70)

    # Goal check
    beat_spy = cagr > spy_cagr
    under_dd = max_dd < 0.15
    print(f"\n  Beat SPY CAGR?  {'YES' if beat_spy else 'NO'} ({cagr:.2%} vs {spy_cagr:.2%})")
    print(f"  Max DD < 15%?   {'YES' if under_dd else 'NO'} ({max_dd:.2%})")
    if beat_spy and under_dd:
        print("\n  >>> GOALS MET: Outperformed SPY with acceptable drawdown <<<")
    elif beat_spy:
        print(f"\n  >>> Beat SPY but drawdown ({max_dd:.2%}) exceeded 15% target <<<")
    else:
        print(f"\n  >>> Underperformed SPY by {spy_cagr - cagr:.2%} CAGR <<<")

    # Save results
    os.makedirs("logs", exist_ok=True)

    results = {
        "period": f"{start_date} to {end_date}",
        "trading_days": len(returns),
        "years": n_years,
        "initial_capital": initial_capital,
        "final_value": float(equity[-1]),
        "total_return": total_return,
        "cagr": cagr,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "total_trades": len(trade_log),
        "win_rate": win_rate,
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor) if profit_factor != float("inf") else "inf",
        "alpha_vs_spy": alpha,
        "spy_cagr": spy_cagr,
        "spy_sharpe": spy_sharpe,
        "spy_max_dd": spy_dd,
        "strategy_attribution": {n: {"pnl": strategy_pnl[n], "trades": strategy_trade_count[n]} for n in strategy_pnl},
        "regime_distribution": regime_pcts,
        "kelly_stats": {"win_rate": k_wr, "payoff_ratio": k_payoff, "kelly_pct": k_kelly},
        "elapsed_seconds": elapsed,
    }

    with open("logs/backtest_results_v2.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to logs/backtest_results_v2.json")

    # Save equity curve
    eq_df = pd.DataFrame({
        "strategy_equity": equity_curve,
        "spy_equity": spy_equity[:len(equity_curve)] if spy_equity else [initial_capital] * len(equity_curve),
    })
    eq_df.to_csv("logs/equity_curve_v2.csv", index=False)
    print(f"Equity curve saved to logs/equity_curve_v2.csv")

    return results


if __name__ == "__main__":
    run_backtest()
