#!/usr/bin/env python3
"""
30-Year Backtest (1996-2026) — Synthetic Data Calibrated to Historical SPY.

Compares 7 strategy configurations:
  1. Buy & Hold (SPY benchmark)
  2. Original Momentum (TSMOM)
  3. Original Mean Reversion (Stat Arb)
  4. Original Factor Momentum (Sentiment)
  5. NEW: Dual Momentum (Antonacci GEM + Keller VAA)
  6. NEW: Cross-Asset Signals (vol regime + trend)
  7. NEW: Enhanced Ensemble (all strategies with adaptive allocation)

Uses synthetic daily returns calibrated to known SPY statistics per era:
  - 1996-2000: +20% annual, 18% vol (dot-com boom)
  - 2000-2002: -15% annual, 25% vol (crash)
  - 2003-2007: +12% annual, 14% vol (recovery)
  - 2007-2009: -20% annual, 35% vol (financial crisis)
  - 2009-2019: +14% annual, 15% vol (bull market)
  - 2020:      -5% annual, 35% vol (COVID)
  - 2021:      +27% annual, 17% vol (recovery)
  - 2022:      -19% annual, 25% vol (bear)
  - 2023-2026: +18% annual, 16% vol (AI rally)
"""

import json
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent dir to path so we can import strategies
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from strategies.mean_reversion import MeanReversionStrategy, Signal
from strategies.momentum import MomentumStrategy
from strategies.sentiment import SentimentStrategy
from strategies.pattern_recognition import PatternRecognitionStrategy, MarketRegime
from strategies.dual_momentum import DualMomentumStrategy
from strategies.cross_asset_signals import CrossAssetSignalStrategy
from strategies.adaptive_allocation import AdaptiveAllocationStrategy


# =============================================================================
# Synthetic Market Data Generator
# =============================================================================

ERA_CONFIGS = [
    {"start": "1996-01-02", "end": "1999-12-31", "annual_return": 0.20, "annual_vol": 0.18, "label": "Dot-Com Boom"},
    {"start": "2000-01-03", "end": "2002-12-31", "annual_return": -0.15, "annual_vol": 0.25, "label": "Dot-Com Crash"},
    {"start": "2003-01-02", "end": "2007-09-30", "annual_return": 0.12, "annual_vol": 0.14, "label": "Recovery"},
    {"start": "2007-10-01", "end": "2009-03-09", "annual_return": -0.20, "annual_vol": 0.35, "label": "Financial Crisis"},
    {"start": "2009-03-10", "end": "2019-12-31", "annual_return": 0.14, "annual_vol": 0.15, "label": "Bull Market"},
    {"start": "2020-01-02", "end": "2020-12-31", "annual_return": -0.05, "annual_vol": 0.35, "label": "COVID"},
    {"start": "2021-01-04", "end": "2021-12-31", "annual_return": 0.27, "annual_vol": 0.17, "label": "Recovery"},
    {"start": "2022-01-03", "end": "2022-12-30", "annual_return": -0.19, "annual_vol": 0.25, "label": "Bear Market"},
    {"start": "2023-01-03", "end": "2026-04-07", "annual_return": 0.18, "annual_vol": 0.16, "label": "AI Rally"},
]


def generate_synthetic_spy(seed: int = 42) -> pd.DataFrame:
    """
    Generate 30 years of synthetic SPY-like daily data.

    Calibrated to known era-by-era statistics. Uses geometric Brownian motion
    with mild vol clustering for realistic dynamics while ensuring the overall
    price path matches expected era returns.
    """
    np.random.seed(seed)

    all_dates = []
    all_prices = []

    # Start price: SPY was ~$62 in Jan 1996
    current_price = 62.0

    for era in ERA_CONFIGS:
        start = pd.Timestamp(era["start"])
        end = pd.Timestamp(era["end"])

        # Generate business days for this era
        dates = pd.bdate_range(start, end)
        n_days = len(dates)

        if n_days == 0:
            continue

        # Daily parameters — adjust mu for geometric return
        daily_sigma = era["annual_vol"] / np.sqrt(252)
        # Geometric return adjustment: E[log(1+r)] = mu - sigma^2/2
        daily_mu = era["annual_return"] / 252 + 0.5 * daily_sigma**2

        # Generate returns with mild vol clustering (normal distribution)
        current_vol = daily_sigma

        era_prices = []
        for i in range(n_days):
            # Mild GARCH-style vol update
            if i > 0 and len(era_prices) > 0:
                prev_ret = (current_price - era_prices[-1]) / era_prices[-1] if era_prices[-1] > 0 else 0
                current_vol = np.sqrt(
                    0.05 * daily_sigma**2 +
                    0.05 * prev_ret**2 +
                    0.90 * current_vol**2
                )
                current_vol = np.clip(current_vol, daily_sigma * 0.5, daily_sigma * 2.0)

            # Normal distribution (not t — avoids unrealistic tail events)
            daily_ret = daily_mu + current_vol * np.random.normal()
            # Cap single-day moves at ±8%
            daily_ret = np.clip(daily_ret, -0.08, 0.08)

            current_price = current_price * (1 + daily_ret)
            current_price = max(current_price, 5.0)  # Floor at $5

            era_prices.append(current_price)

        all_dates.extend(dates)
        all_prices.extend(era_prices)

    prices = np.array(all_prices)

    # Compute returns from prices
    all_returns = np.diff(prices) / prices[:-1]
    all_returns = np.insert(all_returns, 0, 0.0)

    # Build DataFrame with OHLCV
    df = pd.DataFrame(index=pd.DatetimeIndex(all_dates))
    df.index.name = "Date"

    df["Close"] = prices
    # Synthetic OHLV
    daily_range = np.abs(all_returns) * prices + prices * 0.005
    df["High"] = prices + daily_range * np.random.uniform(0.3, 0.7, len(prices))
    df["Low"] = prices - daily_range * np.random.uniform(0.3, 0.7, len(prices))
    df["Low"] = np.maximum(df["Low"], 0.5)
    df["Open"] = df["Close"].shift(1).fillna(prices[0]) + np.random.normal(0, 0.5, len(prices))
    df["Open"] = np.maximum(df["Open"], 0.5)
    df["Volume"] = np.random.lognormal(mean=18, sigma=0.5, size=len(prices)).astype(int)

    # Compute indicators needed by strategies
    df = compute_indicators(df)

    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators needed by the strategy suite."""
    close = df["Close"]

    # Returns
    df["returns"] = close.pct_change()

    # RSI (14-period)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].fillna(50)

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    band_range = upper - lower
    df["BB_pct"] = np.where(band_range > 0, (close - lower) / band_range, 0.5)

    # ATR (14-period)
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - close.shift()).abs()
    low_close = (df["Low"] - close.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    # Volume ratio
    vol_sma = df["Volume"].rolling(20).mean()
    df["Vol_ratio"] = df["Volume"] / vol_sma.replace(0, 1)

    return df


# =============================================================================
# Strategy Runner (Simulates Trading Day-by-Day)
# =============================================================================

@dataclass
class Position:
    symbol: str
    quantity: int
    entry_price: float
    strategy: str
    entry_date: str
    stop_loss: float = 0.0
    take_profit: float = 0.0


@dataclass
class StrategyConfig:
    name: str
    description: str
    strategies: Dict  # name -> strategy instance
    weights: Dict[str, float]  # name -> weight
    use_dual_momentum_overlay: bool = False
    use_cross_asset_overlay: bool = False
    use_adaptive_allocation: bool = False


def run_strategy_backtest(
    df: pd.DataFrame,
    strategy_config: StrategyConfig,
    initial_capital: float = 10000.0,
    stop_loss_pct: float = 0.08,
    take_profit_pct: float = 0.20,
    max_position_pct: float = 0.30,
    warmup_days: int = 260,
) -> Dict:
    """
    Run a single strategy configuration through the synthetic data.

    Simulates day-by-day trading with:
      - Signal generation from all active strategies
      - Weighted signal combination
      - Position sizing with risk management
      - Stop-loss and take-profit
    """
    dates = df.index[warmup_days:]
    capital = initial_capital
    position: Optional[Position] = None
    equity_curve = []
    trades = []
    daily_returns_list = []
    prev_equity = initial_capital

    # Overlay instances
    dual_mom = strategy_config.strategies.get("dual_momentum")
    cross_asset = strategy_config.strategies.get("cross_asset")
    adaptive = strategy_config.strategies.get("adaptive_allocation")

    for i, date in enumerate(dates):
        idx = df.index.get_loc(date)
        df_up_to = df.iloc[:idx+1]
        price = float(df_up_to["Close"].iloc[-1])

        # Check stop-loss / take-profit
        if position is not None:
            pnl_pct = (price - position.entry_price) / position.entry_price
            if pnl_pct <= -stop_loss_pct or pnl_pct >= take_profit_pct:
                pnl = position.quantity * (price - position.entry_price)
                capital += position.quantity * price
                trades.append({
                    "date": str(date.date()),
                    "action": "SELL",
                    "price": price,
                    "pnl": pnl,
                    "reason": "stop_loss" if pnl_pct <= -stop_loss_pct else "take_profit",
                })
                position = None

        # Generate signals
        signals = {}
        for name, strat in strategy_config.strategies.items():
            if name in ("dual_momentum", "cross_asset", "adaptive_allocation"):
                continue  # These are overlays, not direct signal generators here
            try:
                sig = strat.generate_signal("SPY", df_up_to)
                signals[name] = sig.strength
            except Exception:
                signals[name] = 0.0

        # Weighted combination
        weights = dict(strategy_config.weights)
        combined = sum(weights.get(n, 0) * s for n, s in signals.items())

        # Apply overlays
        risk_multiplier = 1.0

        if strategy_config.use_dual_momentum_overlay and dual_mom:
            try:
                dm_signal = dual_mom.generate_signal("SPY", df_up_to)
                # If dual momentum is strongly negative, reduce or go to cash
                if dm_signal.strength < -0.3:
                    risk_multiplier *= 0.3
                elif dm_signal.strength < 0:
                    risk_multiplier *= 0.6
                else:
                    risk_multiplier *= 1.0 + dm_signal.strength * 0.2
            except Exception:
                pass

        if strategy_config.use_cross_asset_overlay and cross_asset:
            try:
                ca_mult = cross_asset.get_risk_multiplier(df_up_to)
                risk_multiplier *= ca_mult
            except Exception:
                pass

        if strategy_config.use_adaptive_allocation and adaptive:
            try:
                aa_signal = adaptive.generate_signal("SPY", df_up_to)
                # Blend adaptive signal into risk multiplier
                if aa_signal.strength < -0.3:
                    risk_multiplier *= 0.5
                elif aa_signal.strength > 0.3:
                    risk_multiplier *= 1.1
            except Exception:
                pass

        combined *= risk_multiplier
        combined = np.clip(combined, -1.0, 1.0)

        # Position management
        total_equity = capital + (
            position.quantity * price if position else 0
        )

        if combined > 0.20 and position is None:
            # Buy
            invest_amount = total_equity * max_position_pct * min(abs(combined), 1.0)
            qty = max(1, int(invest_amount / price))
            cost = qty * price
            if cost <= capital:
                capital -= cost
                position = Position(
                    symbol="SPY",
                    quantity=qty,
                    entry_price=price,
                    strategy=strategy_config.name,
                    entry_date=str(date.date()),
                    stop_loss=price * (1 - stop_loss_pct),
                    take_profit=price * (1 + take_profit_pct),
                )
                trades.append({
                    "date": str(date.date()),
                    "action": "BUY",
                    "quantity": qty,
                    "price": price,
                })

        elif combined < -0.20 and position is not None:
            # Sell
            pnl = position.quantity * (price - position.entry_price)
            capital += position.quantity * price
            trades.append({
                "date": str(date.date()),
                "action": "SELL",
                "price": price,
                "pnl": pnl,
                "reason": "signal",
            })
            position = None

        # End-of-day equity
        total_equity = capital + (position.quantity * price if position else 0)
        equity_curve.append(total_equity)

        daily_ret = (total_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
        daily_returns_list.append(daily_ret)
        prev_equity = total_equity

    # Close any remaining position at end
    if position is not None:
        final_price = float(df["Close"].iloc[-1])
        pnl = position.quantity * (final_price - position.entry_price)
        capital += position.quantity * final_price
        trades.append({
            "date": str(df.index[-1].date()),
            "action": "SELL",
            "price": final_price,
            "pnl": pnl,
            "reason": "end_of_backtest",
        })
        position = None
        equity_curve[-1] = capital

    return compute_metrics(
        strategy_config.name,
        strategy_config.description,
        equity_curve,
        daily_returns_list,
        trades,
        initial_capital,
        df,
        warmup_days,
    )


def run_buy_and_hold(
    df: pd.DataFrame,
    initial_capital: float = 10000.0,
    warmup_days: int = 260,
) -> Dict:
    """Simple buy-and-hold benchmark."""
    dates = df.index[warmup_days:]
    start_price = float(df["Close"].iloc[warmup_days])
    qty = int(initial_capital / start_price)
    cash_remaining = initial_capital - qty * start_price

    equity_curve = []
    daily_returns_list = []
    prev_equity = initial_capital

    for date in dates:
        idx = df.index.get_loc(date)
        price = float(df["Close"].iloc[idx])
        total = cash_remaining + qty * price
        equity_curve.append(total)
        daily_ret = (total - prev_equity) / prev_equity if prev_equity > 0 else 0
        daily_returns_list.append(daily_ret)
        prev_equity = total

    return compute_metrics(
        "Buy & Hold",
        "SPY Buy & Hold Benchmark",
        equity_curve,
        daily_returns_list,
        [],
        initial_capital,
        df,
        warmup_days,
    )


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_metrics(
    name: str,
    description: str,
    equity_curve: List[float],
    daily_returns: List[float],
    trades: List[Dict],
    initial_capital: float,
    df: pd.DataFrame,
    warmup_days: int,
) -> Dict:
    """Compute comprehensive performance metrics."""
    equity = np.array(equity_curve)
    returns = np.array(daily_returns)

    if len(equity) == 0:
        return {"name": name, "error": "No data"}

    final_value = equity[-1]
    total_return = (final_value - initial_capital) / initial_capital
    n_years = len(returns) / 252

    # CAGR
    if n_years > 0 and final_value > 0:
        cagr = (final_value / initial_capital) ** (1 / n_years) - 1
    else:
        cagr = 0.0

    # Sharpe
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 1 and np.std(downside) > 0:
        sortino = np.mean(returns) / np.std(downside) * np.sqrt(252)
    else:
        sortino = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    drawdowns = (peak - equity) / np.where(peak > 0, peak, 1)
    max_dd = float(np.max(drawdowns))

    # Calmar
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    # Trade stats
    sell_trades = [t for t in trades if t.get("action") == "SELL" and "pnl" in t]
    wins = [t for t in sell_trades if t["pnl"] > 0]
    losses = [t for t in sell_trades if t["pnl"] <= 0]
    win_rate = len(wins) / len(sell_trades) if sell_trades else 0
    avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0

    # Era-by-era performance
    era_performance = compute_era_performance(equity_curve, df, warmup_days)

    return {
        "name": name,
        "description": description,
        "initial_capital": initial_capital,
        "final_value": float(final_value),
        "total_return": total_return,
        "total_return_pct": f"{total_return:.1%}",
        "cagr": cagr,
        "cagr_pct": f"{cagr:.2%}",
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "max_drawdown_pct": f"{max_dd:.2%}",
        "calmar_ratio": calmar,
        "n_years": n_years,
        "total_trades": len(trades),
        "win_rate": win_rate,
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else float("inf"),
        "ten_k_becomes": f"${final_value:,.0f}",
        "era_performance": era_performance,
        "equity_curve": equity_curve,
    }


def compute_era_performance(
    equity_curve: List[float],
    df: pd.DataFrame,
    warmup_days: int,
) -> Dict[str, Dict]:
    """Compute performance for each market era."""
    dates = df.index[warmup_days:]
    equity = np.array(equity_curve)

    era_results = {}
    for era in ERA_CONFIGS:
        era_start = pd.Timestamp(era["start"])
        era_end = pd.Timestamp(era["end"])

        # Find indices within the backtest dates
        mask = (dates >= era_start) & (dates <= era_end)
        indices = np.where(mask)[0]

        if len(indices) < 2:
            continue

        era_equity = equity[indices]
        start_val = era_equity[0]
        end_val = era_equity[-1]
        era_return = (end_val - start_val) / start_val if start_val > 0 else 0

        # Era drawdown
        era_peak = np.maximum.accumulate(era_equity)
        era_dd = np.max((era_peak - era_equity) / np.where(era_peak > 0, era_peak, 1))

        era_results[era["label"]] = {
            "return": era_return,
            "return_pct": f"{era_return:.1%}",
            "max_drawdown": era_dd,
            "max_drawdown_pct": f"{era_dd:.1%}",
            "start_value": f"${start_val:,.0f}",
            "end_value": f"${end_val:,.0f}",
        }

    return era_results


# =============================================================================
# Main: Define Strategy Configs and Run
# =============================================================================

def main():
    print("=" * 80)
    print("  30-YEAR BACKTEST: 1996-2026")
    print("  Synthetic SPY data calibrated to historical era statistics")
    print("=" * 80)
    print()

    # Generate synthetic data
    print("Generating 30 years of synthetic market data...")
    t0 = time.time()
    df = generate_synthetic_spy(seed=42)
    print(f"  Generated {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")
    print(f"  Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
    print(f"  Time: {time.time()-t0:.1f}s")
    print()

    initial_capital = 10000.0
    warmup = 260  # ~1 year warmup for indicators

    # Define strategy configurations
    configs = []

    # 1. Original Momentum
    configs.append(StrategyConfig(
        name="Original Momentum",
        description="TSMOM with vol scaling, acceleration, volume confirmation",
        strategies={"momentum": MomentumStrategy()},
        weights={"momentum": 1.0},
    ))

    # 2. Original Mean Reversion
    configs.append(StrategyConfig(
        name="Original Mean Reversion",
        description="Z-score stat-arb with dynamic thresholds and RSI",
        strategies={"mean_reversion": MeanReversionStrategy()},
        weights={"mean_reversion": 1.0},
    ))

    # 3. Original Factor Momentum
    configs.append(StrategyConfig(
        name="Original Factor Momentum",
        description="Multi-factor (momentum, value, quality, reversal)",
        strategies={"sentiment": SentimentStrategy()},
        weights={"sentiment": 1.0},
    ))

    # 4. NEW: Dual Momentum (Antonacci GEM + VAA)
    configs.append(StrategyConfig(
        name="Dual Momentum (GEM+VAA)",
        description="Antonacci absolute+relative momentum with Keller VAA breadth",
        strategies={
            "momentum": MomentumStrategy(),
            "dual_momentum": DualMomentumStrategy(),
        },
        weights={"momentum": 1.0},
        use_dual_momentum_overlay=True,
    ))

    # 5. NEW: Cross-Asset Enhanced
    configs.append(StrategyConfig(
        name="Cross-Asset Enhanced",
        description="Momentum + cross-asset vol/trend risk overlay",
        strategies={
            "momentum": MomentumStrategy(),
            "cross_asset": CrossAssetSignalStrategy(),
        },
        weights={"momentum": 1.0},
        use_cross_asset_overlay=True,
    ))

    # 6. NEW: Full Ensemble with Adaptive Allocation
    configs.append(StrategyConfig(
        name="Enhanced Ensemble",
        description="All strategies + dual momentum overlay + cross-asset + adaptive allocation",
        strategies={
            "momentum": MomentumStrategy(),
            "mean_reversion": MeanReversionStrategy(),
            "sentiment": SentimentStrategy(),
            "pattern_recognition": PatternRecognitionStrategy(),
            "dual_momentum": DualMomentumStrategy(),
            "cross_asset": CrossAssetSignalStrategy(),
            "adaptive_allocation": AdaptiveAllocationStrategy(),
        },
        weights={
            "momentum": 0.30,
            "mean_reversion": 0.20,
            "sentiment": 0.20,
            "pattern_recognition": 0.30,
        },
        use_dual_momentum_overlay=True,
        use_cross_asset_overlay=True,
        use_adaptive_allocation=True,
    ))

    # Run all backtests
    print("Running backtests...")
    print("-" * 80)

    all_results = []

    # Buy & Hold benchmark
    print(f"\n  [1/7] Buy & Hold (SPY)...")
    t0 = time.time()
    bh_result = run_buy_and_hold(df, initial_capital, warmup)
    print(f"    Done in {time.time()-t0:.1f}s — $10K → {bh_result['ten_k_becomes']}")
    all_results.append(bh_result)

    # Run each strategy
    for i, cfg in enumerate(configs):
        print(f"\n  [{i+2}/7] {cfg.name}...")
        t0 = time.time()
        result = run_strategy_backtest(
            df, cfg, initial_capital, warmup_days=warmup,
        )
        print(f"    Done in {time.time()-t0:.1f}s — $10K → {result['ten_k_becomes']}")
        all_results.append(result)

    # ==========================================================================
    # Print Comparison Report
    # ==========================================================================
    print("\n\n")
    print("=" * 120)
    print("                          30-YEAR BACKTEST COMPARISON REPORT (1996-2026)")
    print("=" * 120)

    # Summary table
    print(f"\n{'Strategy':<30} {'$10K Becomes':>14} {'Total Return':>13} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'Max DD':>8} {'Calmar':>8} {'Trades':>8} {'Win%':>7}")
    print("-" * 120)

    for r in all_results:
        name = r["name"][:29]
        print(
            f"{name:<30} "
            f"{r['ten_k_becomes']:>14} "
            f"{r['total_return_pct']:>13} "
            f"{r['cagr_pct']:>8} "
            f"{r['sharpe_ratio']:>8.2f} "
            f"{r['sortino_ratio']:>8.2f} "
            f"{r['max_drawdown_pct']:>8} "
            f"{r['calmar_ratio']:>8.2f} "
            f"{r['total_trades']:>8} "
            f"{r['win_rate']:>6.1%}"
        )

    # Era-by-era comparison
    print(f"\n\n{'='*120}")
    print("  ERA-BY-ERA PERFORMANCE (Return / Max Drawdown)")
    print(f"{'='*120}")

    era_labels = [e["label"] for e in ERA_CONFIGS]
    header = f"{'Strategy':<26}"
    for label in era_labels:
        header += f" {label[:14]:>14}"
    print(header)
    print("-" * 120)

    for r in all_results:
        row = f"{r['name'][:25]:<26}"
        for label in era_labels:
            era_data = r.get("era_performance", {}).get(label, {})
            ret_pct = era_data.get("return_pct", "N/A")
            dd_pct = era_data.get("max_drawdown_pct", "N/A")
            row += f" {ret_pct:>6}/{dd_pct:<6}"
        print(row)

    # Crisis performance detail
    print(f"\n\n{'='*120}")
    print("  CRISIS PERIOD DETAIL")
    print(f"{'='*120}")
    crisis_eras = ["Dot-Com Crash", "Financial Crisis", "COVID", "Bear Market"]
    for era_name in crisis_eras:
        print(f"\n  {era_name}:")
        for r in all_results:
            era_data = r.get("era_performance", {}).get(era_name, {})
            if era_data:
                print(
                    f"    {r['name']:<28} "
                    f"Return: {era_data.get('return_pct', 'N/A'):>8} "
                    f"Max DD: {era_data.get('max_drawdown_pct', 'N/A'):>8} "
                    f"({era_data.get('start_value', '?')} → {era_data.get('end_value', '?')})"
                )

    # Recommendations
    print(f"\n\n{'='*120}")
    print("  ANALYSIS & RECOMMENDATIONS")
    print(f"{'='*120}")

    # Find best by each metric
    best_cagr = max(all_results, key=lambda r: r["cagr"])
    best_sharpe = max(all_results, key=lambda r: r["sharpe_ratio"])
    lowest_dd = min(all_results, key=lambda r: r["max_drawdown"])
    best_calmar = max(all_results, key=lambda r: r["calmar_ratio"])

    print(f"\n  Best CAGR:           {best_cagr['name']} ({best_cagr['cagr_pct']})")
    print(f"  Best Sharpe:         {best_sharpe['name']} ({best_sharpe['sharpe_ratio']:.2f})")
    print(f"  Lowest Max Drawdown: {lowest_dd['name']} ({lowest_dd['max_drawdown_pct']})")
    print(f"  Best Risk-Adjusted:  {best_calmar['name']} (Calmar={best_calmar['calmar_ratio']:.2f})")

    # Save results
    results_path = os.path.join(SCRIPT_DIR, "backtest_30yr_results.json")
    save_results = []
    for r in all_results:
        save_r = {k: v for k, v in r.items() if k != "equity_curve"}
        save_results.append(save_r)

    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_path}")

    # Save equity curves as CSV
    eq_df = pd.DataFrame()
    dates = df.index[warmup:]
    eq_df["Date"] = [str(d.date()) for d in dates]
    for r in all_results:
        col_name = r["name"].replace(" ", "_").replace("(", "").replace(")", "").replace("+", "")
        curve = r.get("equity_curve", [])
        if len(curve) == len(dates):
            eq_df[col_name] = curve
        elif len(curve) > 0:
            # Pad or trim to match
            if len(curve) < len(dates):
                curve = curve + [curve[-1]] * (len(dates) - len(curve))
            eq_df[col_name] = curve[:len(dates)]

    eq_path = os.path.join(SCRIPT_DIR, "equity_curves_30yr.csv")
    eq_df.to_csv(eq_path, index=False)
    print(f"  Equity curves saved to: {eq_path}")

    print(f"\n{'='*120}")
    print("  BACKTEST COMPLETE")
    print(f"{'='*120}")

    return all_results


if __name__ == "__main__":
    main()
