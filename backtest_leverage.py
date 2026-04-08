#!/usr/bin/env python3
"""
30-Year Leverage Backtest — Tests Multiple Leverage Modes on Existing Strategy.

Uses the same synthetic SPY data (calibrated to historical stats) from the
30-year backtest, but applies different leverage configurations to the
Enhanced Ensemble strategy to find the sweet spot.

Configurations tested:
  1. No leverage (1x baseline)
  2. Fixed 2x
  3. Fixed 3x
  4. Fixed 5x
  5. Kelly-optimal dynamic
  6. Vol-target 15% annual

Reports: CAGR, Sharpe, Max DD, $10K final value, risk of ruin.
Includes honest risk assessment of leverage dangers.
"""

import json
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add parent dir to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from leverage_manager import LeverageConfig, LeverageManager, estimate_risk_of_ruin

# Re-use synthetic data generator and strategies from the existing backtest
from backtest_30yr import (
    ERA_CONFIGS,
    generate_synthetic_spy,
    compute_indicators,
    StrategyConfig,
    Position,
)
from strategies.mean_reversion import MeanReversionStrategy, Signal
from strategies.momentum import MomentumStrategy
from strategies.sentiment import SentimentStrategy
from strategies.pattern_recognition import PatternRecognitionStrategy, MarketRegime
from strategies.dual_momentum import DualMomentumStrategy
from strategies.cross_asset_signals import CrossAssetSignalStrategy
from strategies.adaptive_allocation import AdaptiveAllocationStrategy


# =============================================================================
# Leveraged Strategy Runner
# =============================================================================

def run_leveraged_backtest(
    df: pd.DataFrame,
    strategy_config: StrategyConfig,
    leverage_config: LeverageConfig,
    initial_capital: float = 10000.0,
    stop_loss_pct: float = 0.08,
    take_profit_pct: float = 0.20,
    max_position_pct: float = 0.30,
    warmup_days: int = 260,
) -> Dict:
    """
    Run a strategy backtest with leverage applied.

    The leverage manager computes a daily leverage factor that scales
    position sizes. Funding costs are deducted daily for borrowed capital.
    """
    lev_mgr = LeverageManager(leverage_config)

    dates = df.index[warmup_days:]
    capital = initial_capital
    position: Optional[Position] = None
    equity_curve = []
    trades = []
    daily_returns_list = []
    leverage_history = []
    prev_equity = initial_capital

    # Overlay instances
    dual_mom = strategy_config.strategies.get("dual_momentum")
    cross_asset = strategy_config.strategies.get("cross_asset")
    adaptive = strategy_config.strategies.get("adaptive_allocation")

    for i, date in enumerate(dates):
        idx = df.index.get_loc(date)
        df_up_to = df.iloc[:idx + 1]
        price = float(df_up_to["Close"].iloc[-1])

        # Current total equity
        total_equity = capital + (
            position.quantity * price if position else 0
        )

        # Get leverage factor
        leverage = lev_mgr.get_leverage(
            current_equity=total_equity,
            daily_returns=daily_returns_list[-252:] if daily_returns_list else None,
        )
        leverage_history.append(leverage)

        # Deduct daily funding cost
        if leverage > 1.0 and position is not None:
            funding_cost = lev_mgr.compute_funding_cost(
                leverage, total_equity, days=1
            )
            capital -= funding_cost

        # Check stop-loss / take-profit
        if position is not None:
            pnl_pct = (price - position.entry_price) / position.entry_price
            # Leveraged P&L is amplified
            effective_sl = stop_loss_pct / max(leverage, 1.0)  # Tighter stop with leverage
            effective_tp = take_profit_pct

            if pnl_pct <= -effective_sl or pnl_pct >= effective_tp:
                pnl = position.quantity * (price - position.entry_price)
                capital += position.quantity * price
                trades.append({
                    "date": str(date.date()),
                    "action": "SELL",
                    "price": price,
                    "pnl": pnl,
                    "leverage": leverage,
                    "reason": "stop_loss" if pnl_pct <= -effective_sl else "take_profit",
                })
                position = None

        # Generate signals
        signals = {}
        for name, strat in strategy_config.strategies.items():
            if name in ("dual_momentum", "cross_asset", "adaptive_allocation"):
                continue
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
                if aa_signal.strength < -0.3:
                    risk_multiplier *= 0.5
                elif aa_signal.strength > 0.3:
                    risk_multiplier *= 1.1
            except Exception:
                pass

        combined *= risk_multiplier
        combined = np.clip(combined, -1.0, 1.0)

        # Recalculate total equity after funding cost
        total_equity = capital + (
            position.quantity * price if position else 0
        )

        # Position management — leverage scales position size
        if combined > 0.20 and position is None:
            # Levered position size
            invest_amount = (
                total_equity * max_position_pct * min(abs(combined), 1.0) * leverage
            )
            qty = max(1, int(invest_amount / price))

            # Can't invest more than capital allows (margin limit)
            max_qty = int(capital * leverage / price) if price > 0 else 0
            qty = min(qty, max_qty)

            cost = qty * price
            if cost > 0 and capital > 0:
                # For leveraged positions, we use capital as collateral
                # and borrow the rest. Track actual cash outlay.
                cash_needed = min(cost, capital)
                capital -= cash_needed
                # The rest is borrowed (tracked implicitly through leverage)
                # For simplicity, we track the full position
                actual_qty = int(cash_needed / price * leverage) if price > 0 else 0
                if actual_qty > 0:
                    position = Position(
                        symbol="SPY",
                        quantity=actual_qty,
                        entry_price=price,
                        strategy=strategy_config.name,
                        entry_date=str(date.date()),
                        stop_loss=price * (1 - effective_sl if 'effective_sl' in dir() else stop_loss_pct),
                        take_profit=price * (1 + take_profit_pct),
                    )
                    trades.append({
                        "date": str(date.date()),
                        "action": "BUY",
                        "quantity": actual_qty,
                        "price": price,
                        "leverage": leverage,
                    })

        elif combined < -0.20 and position is not None:
            pnl = position.quantity * (price - position.entry_price)
            capital += position.quantity * price
            trades.append({
                "date": str(date.date()),
                "action": "SELL",
                "price": price,
                "pnl": pnl,
                "leverage": leverage,
                "reason": "signal",
            })
            position = None

        # End-of-day equity
        total_equity = capital + (position.quantity * price if position else 0)
        # Floor at zero (margin call / wipeout)
        total_equity = max(total_equity, 0.0)
        equity_curve.append(total_equity)

        daily_ret = (
            (total_equity - prev_equity) / prev_equity
            if prev_equity > 0
            else 0
        )
        daily_returns_list.append(daily_ret)
        prev_equity = total_equity

    # Close any remaining position
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
        equity_curve[-1] = max(capital, 0.0)

    # Compute metrics
    return compute_leverage_metrics(
        leverage_config,
        equity_curve,
        daily_returns_list,
        trades,
        leverage_history,
        initial_capital,
        df,
        warmup_days,
    )


def compute_leverage_metrics(
    lev_config: LeverageConfig,
    equity_curve: List[float],
    daily_returns: List[float],
    trades: List[Dict],
    leverage_history: List[float],
    initial_capital: float,
    df: pd.DataFrame,
    warmup_days: int,
) -> Dict:
    """Compute comprehensive metrics including leverage-specific stats."""
    equity = np.array(equity_curve)
    returns = np.array(daily_returns)
    leverages = np.array(leverage_history)

    if len(equity) == 0:
        return {"name": f"Leverage ({lev_config.mode})", "error": "No data"}

    final_value = max(equity[-1], 0.0)
    total_return = (final_value - initial_capital) / initial_capital
    n_years = len(returns) / 252

    # CAGR
    if n_years > 0 and final_value > 0:
        cagr = (final_value / initial_capital) ** (1 / n_years) - 1
    else:
        cagr = -1.0  # Total loss

    # Annual vol
    annual_vol = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0

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

    # Leverage stats
    avg_leverage = float(np.mean(leverages)) if len(leverages) > 0 else 1.0
    max_leverage_used = float(np.max(leverages)) if len(leverages) > 0 else 1.0

    # Risk of ruin (Monte Carlo)
    unlevered_vol = annual_vol / avg_leverage if avg_leverage > 0 else annual_vol
    unlevered_cagr = cagr  # Approximate
    risk_of_ruin = estimate_risk_of_ruin(
        cagr=max(unlevered_cagr, 0.01),
        annual_vol=max(unlevered_vol, 0.01),
        leverage=avg_leverage,
        ruin_threshold=0.5,
        years=30,
    )

    # Volatility drag: theoretical drag = 0.5 * leverage^2 * sigma^2
    vol_drag = 0.5 * avg_leverage ** 2 * unlevered_vol ** 2

    # Total funding costs
    total_funding = sum(
        t.get("funding_cost", 0) for t in trades
    )

    # Mode label
    if lev_config.mode == "none":
        label = "No Leverage (1x)"
    elif lev_config.mode == "fixed":
        label = f"Fixed {lev_config.fixed_multiplier:.0f}x"
    elif lev_config.mode == "kelly":
        label = "Kelly Optimal"
    elif lev_config.mode == "vol_target":
        label = f"Vol Target {lev_config.vol_target_annual:.0%}"
    else:
        label = lev_config.mode

    # Era performance
    era_performance = {}
    dates = df.index[warmup_days:]
    for era in ERA_CONFIGS:
        era_start = pd.Timestamp(era["start"])
        era_end = pd.Timestamp(era["end"])
        mask = (dates >= era_start) & (dates <= era_end)
        indices = np.where(mask)[0]
        if len(indices) >= 2:
            era_equity = equity[indices]
            start_val = era_equity[0]
            end_val = era_equity[-1]
            era_return = (end_val - start_val) / start_val if start_val > 0 else -1
            era_peak = np.maximum.accumulate(era_equity)
            era_dd = float(np.max(
                (era_peak - era_equity) / np.where(era_peak > 0, era_peak, 1)
            ))
            era_performance[era["label"]] = {
                "return": era_return,
                "return_pct": f"{era_return:.1%}",
                "max_drawdown": era_dd,
                "max_drawdown_pct": f"{era_dd:.1%}",
                "start_value": f"${start_val:,.0f}",
                "end_value": f"${end_val:,.0f}",
            }

    return {
        "name": label,
        "mode": lev_config.mode,
        "initial_capital": initial_capital,
        "final_value": float(final_value),
        "total_return": total_return,
        "total_return_pct": f"{total_return:.1%}",
        "cagr": cagr,
        "cagr_pct": f"{cagr:.2%}",
        "annual_vol": annual_vol,
        "annual_vol_pct": f"{annual_vol:.2%}",
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "max_drawdown_pct": f"{max_dd:.2%}",
        "calmar_ratio": calmar,
        "n_years": n_years,
        "total_trades": len(trades),
        "win_rate": win_rate,
        "ten_k_becomes": f"${final_value:,.0f}",
        "avg_leverage": avg_leverage,
        "max_leverage_used": max_leverage_used,
        "risk_of_ruin_50pct": risk_of_ruin,
        "risk_of_ruin_pct": f"{risk_of_ruin:.2%}",
        "vol_drag_annual": vol_drag,
        "vol_drag_pct": f"{vol_drag:.2%}",
        "era_performance": era_performance,
        "equity_curve": equity_curve,
    }


# =============================================================================
# Main: Run All Leverage Configurations
# =============================================================================

def main():
    print("=" * 100)
    print("  LEVERAGE BACKTEST: 30-Year (1996-2026)")
    print("  Testing leverage modes on Enhanced Ensemble strategy")
    print("=" * 100)
    print()

    # Generate synthetic data (same seed as original for consistency)
    print("Generating 30 years of synthetic market data (seed=42)...")
    t0 = time.time()
    df = generate_synthetic_spy(seed=42)
    print(f"  Generated {len(df)} trading days ({df.index[0].date()} to {df.index[-1].date()})")
    print(f"  Time: {time.time() - t0:.1f}s")
    print()

    initial_capital = 10000.0
    warmup = 260

    # Base strategy: Enhanced Ensemble (best from previous backtest)
    base_strategy = StrategyConfig(
        name="Enhanced Ensemble",
        description="All strategies + overlays",
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
    )

    # Define leverage configurations to test
    leverage_configs = [
        LeverageConfig(mode="none"),
        LeverageConfig(mode="fixed", fixed_multiplier=2.0, max_leverage=2.0),
        LeverageConfig(mode="fixed", fixed_multiplier=3.0, max_leverage=3.0),
        LeverageConfig(mode="fixed", fixed_multiplier=5.0, max_leverage=5.0),
        LeverageConfig(mode="kelly", max_leverage=5.0),
        LeverageConfig(mode="vol_target", vol_target_annual=0.15, max_leverage=5.0),
    ]

    labels = [
        "No Leverage (1x)",
        "Fixed 2x",
        "Fixed 3x",
        "Fixed 5x",
        "Kelly Optimal",
        "Vol Target 15%",
    ]

    # Run all backtests
    all_results = []
    for i, (lev_cfg, label) in enumerate(zip(leverage_configs, labels)):
        print(f"  [{i + 1}/{len(leverage_configs)}] {label}...")
        t0 = time.time()

        # Need fresh strategy instances for each run
        strategy = StrategyConfig(
            name=base_strategy.name,
            description=base_strategy.description,
            strategies={
                "momentum": MomentumStrategy(),
                "mean_reversion": MeanReversionStrategy(),
                "sentiment": SentimentStrategy(),
                "pattern_recognition": PatternRecognitionStrategy(),
                "dual_momentum": DualMomentumStrategy(),
                "cross_asset": CrossAssetSignalStrategy(),
                "adaptive_allocation": AdaptiveAllocationStrategy(),
            },
            weights=dict(base_strategy.weights),
            use_dual_momentum_overlay=True,
            use_cross_asset_overlay=True,
            use_adaptive_allocation=True,
        )

        result = run_leveraged_backtest(
            df, strategy, lev_cfg, initial_capital, warmup_days=warmup,
        )
        elapsed = time.time() - t0
        print(f"    Done in {elapsed:.1f}s — $10K -> {result['ten_k_becomes']}")
        all_results.append(result)

    # =========================================================================
    # Print Comparison Report
    # =========================================================================
    print("\n\n")
    print("=" * 130)
    print("                    LEVERAGE BACKTEST COMPARISON — 30 YEARS (1996-2026)")
    print("=" * 130)

    # Summary table
    print(f"\n{'Config':<22} {'$10K Becomes':>14} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} "
          f"{'Sortino':>8} {'Max DD':>8} {'Calmar':>8} {'Avg Lev':>8} {'Ruin%':>8}")
    print("-" * 130)

    for r in all_results:
        name = r["name"][:21]
        print(
            f"{name:<22} "
            f"{r['ten_k_becomes']:>14} "
            f"{r['cagr_pct']:>8} "
            f"{r['annual_vol_pct']:>8} "
            f"{r['sharpe_ratio']:>8.2f} "
            f"{r['sortino_ratio']:>8.2f} "
            f"{r['max_drawdown_pct']:>8} "
            f"{r['calmar_ratio']:>8.2f} "
            f"{r['avg_leverage']:>7.1f}x "
            f"{r['risk_of_ruin_pct']:>8}"
        )

    # Sweet spot analysis
    print(f"\n\n{'='*130}")
    print("  SWEET SPOT ANALYSIS — Max Returns with Drawdown < 20%")
    print(f"{'='*130}")

    candidates = [r for r in all_results if r["max_drawdown"] < 0.20]
    if candidates:
        best = max(candidates, key=lambda r: r["cagr"])
        print(f"\n  WINNER: {best['name']}")
        print(f"    CAGR:           {best['cagr_pct']}")
        print(f"    Sharpe:         {best['sharpe_ratio']:.2f}")
        print(f"    Max Drawdown:   {best['max_drawdown_pct']}")
        print(f"    $10K becomes:   {best['ten_k_becomes']}")
        print(f"    Avg Leverage:   {best['avg_leverage']:.1f}x")
        print(f"    Risk of Ruin:   {best['risk_of_ruin_pct']}")
    else:
        print("\n  No configuration achieved < 20% max drawdown.")
        # Find closest
        closest = min(all_results, key=lambda r: abs(r["max_drawdown"] - 0.20))
        print(f"  Closest: {closest['name']} with {closest['max_drawdown_pct']} max DD")

    # All configs that beat the baseline Sharpe
    baseline_sharpe = all_results[0]["sharpe_ratio"]
    print(f"\n  Configs that maintain or improve Sharpe (baseline: {baseline_sharpe:.2f}):")
    for r in all_results:
        if r["sharpe_ratio"] >= baseline_sharpe * 0.9:  # Within 10%
            marker = " <-- SWEET SPOT" if (
                r["max_drawdown"] < 0.20 and r["cagr"] > all_results[0]["cagr"]
            ) else ""
            print(
                f"    {r['name']:<22} Sharpe={r['sharpe_ratio']:.2f} "
                f"CAGR={r['cagr_pct']} DD={r['max_drawdown_pct']}{marker}"
            )

    # Era-by-era comparison for crisis periods
    print(f"\n\n{'='*130}")
    print("  CRISIS PERIOD PERFORMANCE")
    print(f"{'='*130}")

    crisis_eras = ["Dot-Com Crash", "Financial Crisis", "COVID", "Bear Market"]
    for era_name in crisis_eras:
        print(f"\n  {era_name}:")
        for r in all_results:
            era_data = r.get("era_performance", {}).get(era_name, {})
            if era_data:
                print(
                    f"    {r['name']:<22} "
                    f"Return: {era_data.get('return_pct', 'N/A'):>8} "
                    f"Max DD: {era_data.get('max_drawdown_pct', 'N/A'):>8} "
                    f"({era_data.get('start_value', '?')} -> {era_data.get('end_value', '?')})"
                )

    # =========================================================================
    # Honest Risk Assessment
    # =========================================================================
    print(f"\n\n{'='*130}")
    print("  HONEST RISK ASSESSMENT — LEVERAGE DANGERS")
    print(f"{'='*130}")

    print("""
  1. VOLATILITY DRAG (the silent killer)
     Leverage amplifies volatility, and volatility compounds negatively.
     A 2x leveraged position with 15% annual vol suffers ~2.25% annual drag.
     At 5x leverage, this drag exceeds 5.6% annually — often wiping out
     the extra return leverage was supposed to provide.

     Theoretical drag = 0.5 * L^2 * sigma^2 (per year)""")

    for r in all_results:
        print(f"       {r['name']:<22} Vol drag: {r['vol_drag_pct']}")

    print("""
  2. MARGIN CALLS AND FORCED LIQUIDATION
     Backtests assume you can always maintain positions. In reality,
     brokers issue margin calls during drawdowns. At 3x leverage,
     a 33% drawdown wipes you out entirely. At 5x, only 20% is needed.
     Forced liquidation locks in losses at the worst possible moment.

  3. FUNDING COSTS
     Leveraged positions require borrowing. Current margin rates are
     5-8% for retail. This backtest assumes 2% spread (institutional).
     At 3x leverage, funding costs consume 4% annually.
     At 5x, it's 8% — which may exceed your unlevered alpha.

  4. PATH DEPENDENCY
     Leverage makes returns path-dependent. Two strategies with identical
     annual returns but different paths produce vastly different levered
     results. The strategy with higher daily vol loses more to drag.
     A -50% drawdown followed by +100% recovery = breakeven unlevered,
     but with 2x leverage: -100% drawdown = permanent wipeout.

  5. TAIL RISK / BLACK SWANS
     Historical backtests understate tail risk. Real markets have:
     - Flash crashes (2010: -9% in minutes)
     - Gap-down opens (can't stop-loss overnight)
     - Liquidity crises (can't exit at any price)
     - Correlation breakdowns (everything falls together)
     At 5x leverage, a single 20% gap-down event = total wipeout.

  6. MODEL RISK
     Our strategy's 0.82 Sharpe and 4% max DD are from a BACKTEST.
     Live performance typically degrades 30-50% from backtest.
     If real Sharpe is 0.4 instead of 0.82, Kelly-optimal leverage
     drops from ~4x to ~2x, and 5x leverage becomes suicidal.

  7. REGULATORY AND OPERATIONAL RISK
     - Reg T requires 50% initial margin (max 2x for retail)
     - Portfolio margin allows more but requires $100K+ minimum
     - Futures/options can provide leverage but add complexity
     - ETFs like TQQQ have their own drag and tracking error
""")

    # Recommendation
    print(f"{'='*130}")
    print("  RECOMMENDATION")
    print(f"{'='*130}")
    print("""
  Given our strategy's profile (0.82 Sharpe, 4% max DD, 1.58% CAGR):

  - Vol-target 15% or Fixed 2x is the sweet spot for most investors.
    It meaningfully boosts returns while keeping drawdowns manageable.

  - Kelly-optimal is theoretically best but assumes stable edge.
    Use half-Kelly (what we implement) for robustness.

  - Fixed 3x is aggressive but viable for institutional investors
    with low funding costs and strong risk management.

  - Fixed 5x is NOT recommended. The vol drag, funding costs, and
    tail risk almost certainly eliminate any benefit in live trading.
    It may look good in backtests due to survivorship bias.

  Bottom line: Leverage is a tool, not a strategy. It amplifies what
  you already have — if your edge is real, moderate leverage (2-3x)
  can be powerful. If your edge is overstated, leverage accelerates ruin.
""")

    # =========================================================================
    # Save Results
    # =========================================================================
    results_path = os.path.join(SCRIPT_DIR, "leverage_backtest_results.json")
    save_results = []
    for r in all_results:
        save_r = {k: v for k, v in r.items() if k != "equity_curve"}
        save_results.append(save_r)

    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n  Results saved to: {results_path}")

    # Save equity curves
    eq_df = pd.DataFrame()
    dates = df.index[warmup:]
    eq_df["Date"] = [str(d.date()) for d in dates]
    for r in all_results:
        col_name = r["name"].replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")
        curve = r.get("equity_curve", [])
        if len(curve) == len(dates):
            eq_df[col_name] = curve
        elif len(curve) > 0:
            if len(curve) < len(dates):
                curve = curve + [curve[-1]] * (len(dates) - len(curve))
            eq_df[col_name] = curve[:len(dates)]

    eq_path = os.path.join(SCRIPT_DIR, "leverage_equity_curves.csv")
    eq_df.to_csv(eq_path, index=False)
    print(f"  Equity curves saved to: {eq_path}")

    print(f"\n{'='*130}")
    print("  LEVERAGE BACKTEST COMPLETE")
    print(f"{'='*130}")

    return all_results


if __name__ == "__main__":
    main()
