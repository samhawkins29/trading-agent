"""
Paper Trading Launcher — Single entry point for the entire system.

Usage:
    python start_paper_trading.py              # Start live paper trading
    python start_paper_trading.py --dry-run    # Simulate without Alpaca API
    python start_paper_trading.py --status     # Print system status and exit
    python start_paper_trading.py --once       # Run a single cycle and exit

Features:
    - Pre-flight checks: API keys, connectivity, strategy loading
    - Loads all strategies from config
    - Starts the live trading loop on configured interval
    - --dry-run mode: full pipeline without API calls
    - Ctrl+C for graceful shutdown (saves state, logs final positions)
"""

import argparse
import os
import sys
import time
from datetime import datetime

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ALPACA_API_KEY, ALPACA_BASE_URL, ALPACA_SECRET_KEY, config


def print_banner():
    """Print startup banner."""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║           AI TRADING AGENT — PAPER TRADING              ║
    ║                                                         ║
    ║   Regime-Aware | Kelly Sizing | Vol Targeting           ║
    ║   Dynamic Leverage | Self-Improving Weights             ║
    ╚══════════════════════════════════════════════════════════╝
    """)


def check_dependencies() -> bool:
    """Verify all required packages are installed."""
    missing = []
    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "yfinance": "yfinance",
        "requests": "requests",
    }
    for name, pip_name in packages.items():
        try:
            __import__(name)
        except ImportError:
            missing.append(pip_name)

    # Optional but recommended
    try:
        import alpaca_trade_api
        print("  [OK] alpaca-trade-api installed")
    except ImportError:
        print("  [!!] alpaca-trade-api NOT installed")
        print("       Install with: pip install alpaca-trade-api")
        print("       (Required for live paper trading; dry-run still works)")

    if missing:
        print(f"\n  [FAIL] Missing required packages: {', '.join(missing)}")
        print(f"         Install with: pip install {' '.join(missing)}")
        return False

    print("  [OK] All required dependencies installed")
    return True


def check_api_keys() -> bool:
    """Check if Alpaca API keys are configured."""
    if ALPACA_API_KEY == "YOUR_ALPACA_API_KEY":
        print("\n  [!!] Alpaca API keys not configured")
        print("       Set environment variables:")
        print("         export ALPACA_API_KEY='your-key-here'")
        print("         export ALPACA_SECRET_KEY='your-secret-here'")
        print("       Or update config.py directly.")
        return False

    print(f"  [OK] API Key: {ALPACA_API_KEY[:8]}...")
    print(f"  [OK] Base URL: {ALPACA_BASE_URL}")
    return True


def check_alpaca_connection() -> bool:
    """Test Alpaca API connectivity."""
    import requests as req

    try:
        resp = req.get(
            f"{ALPACA_BASE_URL}/v2/account",
            headers={
                "APCA-API-KEY-ID": ALPACA_API_KEY,
                "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            account = resp.json()
            print(f"  [OK] Alpaca connected")
            print(f"       Account: {account.get('account_number', 'N/A')}")
            print(f"       Status:  {account.get('status', 'N/A')}")
            print(f"       Cash:    ${float(account.get('cash', 0)):,.2f}")
            print(f"       Equity:  ${float(account.get('portfolio_value', 0)):,.2f}")
            return True
        else:
            print(f"  [FAIL] Alpaca returned status {resp.status_code}")
            print(f"         {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"  [FAIL] Alpaca connection failed: {e}")
        return False


def check_strategies() -> bool:
    """Verify all strategy modules load correctly."""
    try:
        from strategies.mean_reversion import MeanReversionStrategy
        from strategies.momentum import MomentumStrategy
        from strategies.pattern_recognition import PatternRecognitionStrategy
        from strategies.sentiment import SentimentStrategy

        strategies = {
            "mean_reversion": MeanReversionStrategy(),
            "momentum": MomentumStrategy(),
            "sentiment": SentimentStrategy(),
            "pattern_recognition": PatternRecognitionStrategy(),
        }
        print(f"  [OK] {len(strategies)} strategies loaded:")
        for name, strat in strategies.items():
            weight = config.strategy_weights.get(name, 0)
            print(f"       - {name} (weight: {weight:.2f})")
        return True
    except Exception as e:
        print(f"  [FAIL] Strategy loading failed: {e}")
        return False


def check_config():
    """Display key configuration settings."""
    print("\n  CONFIGURATION")
    print(f"  {'Symbols:':<25} {config.symbols}")
    print(f"  {'Initial Capital:':<25} ${config.initial_capital:,.2f}")
    print(f"  {'Stop Loss:':<25} {config.stop_loss_pct:.0%}")
    print(f"  {'Take Profit:':<25} {config.take_profit_pct:.0%}")
    print(f"  {'Max Drawdown:':<25} {config.max_drawdown_pct:.0%}")
    print(f"  {'Max Positions:':<25} {config.max_open_positions}")
    print(f"  {'Kelly Enabled:':<25} {config.use_kelly}")
    print(f"  {'Vol Target:':<25} {config.vol_target:.0%}")
    print(f"  {'Regime Weighting:':<25} {config.use_regime_weighting}")

    lev = config.leverage
    print(f"  {'Leverage Mode:':<25} {lev.get('mode', 'none')}")
    print(f"  {'Max Leverage:':<25} {lev.get('max_leverage', 1.0)}x")

    pt = getattr(config, "paper_trading", {})
    if pt:
        print(f"\n  PAPER TRADING")
        print(f"  {'Interval:':<25} {pt.get('interval_minutes', 15)} min")
        print(f"  {'Market Open:':<25} "
              f"{pt.get('market_open_hour', 9)}:{pt.get('market_open_minute', 30):02d} ET")
        print(f"  {'Market Close:':<25} "
              f"{pt.get('market_close_hour', 16)}:{pt.get('market_close_minute', 0):02d} ET")
        pt_symbols = pt.get("symbols")
        if pt_symbols:
            print(f"  {'Paper Symbols:':<25} {pt_symbols}")


def run_preflight(dry_run: bool = False) -> bool:
    """
    Run all pre-flight checks.

    Returns True if system is ready to trade.
    """
    print("\n  PRE-FLIGHT CHECKS")
    print("  " + "-" * 40)

    # Dependencies
    deps_ok = check_dependencies()
    if not deps_ok:
        return False

    # Strategies
    strats_ok = check_strategies()
    if not strats_ok:
        return False

    # API (not required for dry run)
    if dry_run:
        print("\n  [DRY RUN] Skipping API checks")
        api_ok = True
    else:
        api_ok = check_api_keys()
        if api_ok:
            api_ok = check_alpaca_connection()

    check_config()

    # Final verdict
    print("\n  " + "-" * 40)
    if deps_ok and strats_ok:
        if api_ok or dry_run:
            print("  [READY] All checks passed. System ready.")
            return True
        else:
            print("  [WARN] API not connected. Use --dry-run to simulate.")
            print("         Or configure Alpaca keys (see PAPER_TRADING_SETUP.md)")
            return False
    else:
        print("  [FAIL] Pre-flight checks failed. Fix issues above.")
        return False


def print_status():
    """Print current system status without starting the trader."""
    from live_trader import LiveTrader

    print("\n  SYSTEM STATUS")
    print("  " + "-" * 40)

    trader = LiveTrader(dry_run=True)
    status = trader.get_status()

    print(f"  {'Market Open:':<25} {status['market_open']}")
    print(f"  {'Alpaca Connected:':<25} {status['alpaca_connected']}")
    print(f"  {'Current Regime:':<25} {status['current_regime']}")
    print(f"  {'Total PnL:':<25} ${status['total_pnl']:+,.2f}")

    risk = status.get("risk", {})
    print(f"  {'Capital:':<25} ${risk.get('capital', 0):,.2f}")
    print(f"  {'Total Value:':<25} ${risk.get('total_value', 0):,.2f}")
    print(f"  {'Open Positions:':<25} {risk.get('open_positions', 0)}")
    print(f"  {'Exposure:':<25} {risk.get('exposure_pct', 0):.1%}")
    print(f"  {'Drawdown:':<25} {risk.get('drawdown_pct', 0):.1%}")
    print(f"  {'Trading Halted:':<25} {risk.get('trading_halted', False)}")

    weights = status.get("strategy_weights", {})
    print(f"\n  STRATEGY WEIGHTS")
    for name, w in weights.items():
        print(f"  {'  ' + name:<25} {w:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Trading Agent — Paper Trading Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_paper_trading.py              Start live paper trading
  python start_paper_trading.py --dry-run    Simulate without API calls
  python start_paper_trading.py --once       Run single cycle and exit
  python start_paper_trading.py --status     Show system status
        """,
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run full pipeline without executing real API orders",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single trading cycle and exit",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print system status and exit",
    )
    parser.add_argument(
        "--skip-checks", action="store_true",
        help="Skip pre-flight checks (not recommended)",
    )
    args = parser.parse_args()

    print_banner()

    # Status mode
    if args.status:
        print_status()
        return

    # Pre-flight
    if not args.skip_checks:
        ready = run_preflight(dry_run=args.dry_run)
        if not ready:
            print("\n  Aborting. Fix issues above or use --dry-run.\n")
            sys.exit(1)

    # Import trader (after pre-flight so errors are caught)
    from live_trader import LiveTrader

    mode = "DRY RUN" if args.dry_run else "LIVE PAPER"
    print(f"\n  Starting {mode} trading...")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Press Ctrl+C for graceful shutdown.\n")

    trader = LiveTrader(dry_run=args.dry_run)

    if args.once:
        # Single cycle mode
        result = trader.run_cycle()
        print(f"\n  Cycle complete. Buys: {len(result['buys'])}, "
              f"Sells: {len(result['sells'])}, Holds: {len(result['holds'])}")
    else:
        # Continuous trading loop
        trader.run()


if __name__ == "__main__":
    main()
