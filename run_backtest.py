#!/usr/bin/env python3
"""
Run 10-year backtest comparing redesigned strategies vs SPY.

Outputs:
  - Console report with all metrics
  - JSON results file
  - CSV equity curve
"""

import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtester import Backtester
from config import config


def main():
    print("=" * 70)
    print("  AI Trading Agent v2 — 10-Year Backtest")
    print("  Period: 2015-01-01 to 2025-12-31")
    print("=" * 70)
    print()

    # Run backtest with default config (10 years)
    bt = Backtester(
        symbols=config.symbols,
        start_date="2015-01-01",
        end_date="2025-12-31",
        initial_capital=100_000.0,
    )

    results = bt.run()

    if "error" in results:
        print(f"\nBacktest failed: {results['error']}")
        return

    # Save results to JSON
    results_path = os.path.join("logs", "backtest_results_v2.json")
    os.makedirs("logs", exist_ok=True)

    # Convert non-serializable types
    clean_results = {}
    for k, v in results.items():
        if isinstance(v, float) and (v == float("inf") or v == float("-inf")):
            clean_results[k] = str(v)
        else:
            clean_results[k] = v

    with open(results_path, "w") as f:
        json.dump(clean_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Save equity curve to CSV
    eq_curve = bt.get_equity_curve()
    eq_path = os.path.join("logs", "equity_curve_v2.csv")
    eq_curve.to_csv(eq_path, index=False)
    print(f"Equity curve saved to {eq_path}")

    # Print comparison summary
    print("\n" + "=" * 70)
    print("  QUICK COMPARISON: OLD vs NEW vs SPY")
    print("=" * 70)
    print(f"  {'Metric':<25s} {'Old (v1)':>12s} {'New (v2)':>12s} {'SPY':>12s}")
    print("-" * 70)

    spy = results.get("spy_benchmark", {})
    old_cagr = 0.015  # 1.5% as reported
    new_cagr = results["cagr"]
    spy_cagr = spy.get("cagr", 0.117)

    print(f"  {'CAGR':<25s} {old_cagr:>11.2%} {new_cagr:>11.2%} {spy_cagr:>11.2%}")
    print(f"  {'Sharpe Ratio':<25s} {'~0.1':>12s} {results['sharpe_ratio']:>12.2f} {spy.get('sharpe', 0):>12.2f}")
    print(f"  {'Max Drawdown':<25s} {'~25%':>12s} {results['max_drawdown']:>12.2%} {spy.get('max_drawdown', 0):>12.2%}")
    print(f"  {'Total Return':<25s} {'~15%':>12s} {results['total_return']:>12.2%} {spy.get('total_return', 0):>12.2%}")
    print(f"  {'Final Value ($100K)':<25s} {'$115,000':>12s} ${results['final_value']:>11,.0f} ${spy.get('final_value', 0):>11,.0f}")
    print("=" * 70)

    beat_spy = new_cagr > spy_cagr
    under_dd = results["max_drawdown"] < 0.15
    print(f"\n  Beat SPY CAGR? {'YES' if beat_spy else 'NO'} ({new_cagr:.2%} vs {spy_cagr:.2%})")
    print(f"  Max DD < 15%?  {'YES' if under_dd else 'NO'} ({results['max_drawdown']:.2%})")

    if beat_spy and under_dd:
        print("\n  >>> GOALS MET: Outperformed SPY with acceptable drawdown <<<")
    elif beat_spy:
        print("\n  >>> Outperformed SPY but drawdown exceeded 15% target <<<")
    else:
        print(f"\n  >>> Underperformed SPY by {spy_cagr - new_cagr:.2%} CAGR <<<")


if __name__ == "__main__":
    main()
