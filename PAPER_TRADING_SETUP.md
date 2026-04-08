# Paper Trading Setup Guide

Step-by-step instructions for setting up and running the AI Trading Agent in paper trading mode.

## 1. Create an Alpaca Account

1. Go to [https://alpaca.markets](https://alpaca.markets) and click **Sign Up**
2. Complete registration (email, password, basic info)
3. You do NOT need to fund the account — paper trading uses simulated money
4. Once logged in, switch to **Paper Trading** mode using the toggle in the dashboard sidebar

## 2. Get Your API Keys

1. In the Alpaca dashboard, navigate to **Paper Trading** > **API Keys**
2. Click **Generate New Key**
3. Copy both the **API Key ID** and **Secret Key** (the secret is only shown once)
4. Store them securely — you'll need them in the next step

## 3. Configure Environment Variables

Set your API keys as environment variables. Choose one method:

**Option A — Shell export (temporary, per session):**

```bash
export ALPACA_API_KEY="your-api-key-here"
export ALPACA_SECRET_KEY="your-secret-key-here"
export ALPACA_BASE_URL="https://paper-api.alpaca.markets"
```

**Option B — `.env` file (recommended for persistence):**

Create a file called `.env` in the project root:

```
ALPACA_API_KEY=your-api-key-here
ALPACA_SECRET_KEY=your-secret-key-here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

Then load it before running: `source .env` (or use `python-dotenv`).

**Option C — Edit `config.py` directly** (not recommended for version control):

Replace the placeholder values at the top of `config.py`.

**Important:** The `ALPACA_BASE_URL` must point to `https://paper-api.alpaca.markets` for paper trading. Never point it to the live URL until you are ready for real money.

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs `alpaca-trade-api`, `yfinance`, `pandas`, `numpy`, and other required packages.

## 5. Verify Setup

Run the launcher with pre-flight checks:

```bash
python start_paper_trading.py --status
```

This will verify all dependencies are installed, API keys are configured, Alpaca is reachable, all strategies load correctly, and display current configuration.

## 6. Start Paper Trading

**Full live mode (runs continuously during market hours):**

```bash
python start_paper_trading.py
```

**Dry-run mode (simulates everything without API calls):**

```bash
python start_paper_trading.py --dry-run
```

**Single cycle (run once and exit):**

```bash
python start_paper_trading.py --once
```

Press `Ctrl+C` at any time for graceful shutdown. The system saves all state and logs open positions.

## 7. Monitor Performance

**Console dashboard (one-time snapshot):**

```bash
python paper_trading_dashboard.py
```

**Live dashboard (refreshes every 60 seconds):**

```bash
python paper_trading_dashboard.py --live
```

**Export snapshot to JSON:**

```bash
python paper_trading_dashboard.py --export
```

## 8. End-of-Day Results Collection

After market close, run the results collector to feed performance data back into the self-improvement system:

```bash
python results_collector.py
```

**View multi-day performance report:**

```bash
python results_collector.py --report --days 30
```

**Collect for a specific date:**

```bash
python results_collector.py --date 2026-04-08
```

## 9. Configuration Options

Key settings in `config.py` under `paper_trading`:

| Setting | Default | Description |
|---|---|---|
| `symbols` | 10 tickers | Which stocks to trade |
| `interval_minutes` | 15 | Minutes between trading cycles |
| `market_open_hour/minute` | 9:30 ET | When to start trading |
| `market_close_hour/minute` | 16:00 ET | When to stop trading |
| `min_paper_days` | 180 | Minimum paper trading days before live |
| `min_sharpe_for_live` | 0.5 | Sharpe ratio required before live |
| `max_drawdown_for_live` | 0.15 | Max drawdown allowed before live |
| `min_trades_for_live` | 200 | Minimum trades needed before live |

## 10. How Long to Paper Trade Before Real Money

**Minimum: 6 months.** This is not negotiable. Here's why:

- **Statistical significance:** You need at least 200+ trades across different market conditions to have confidence in your strategy's edge.
- **Regime coverage:** Markets cycle through trending, range-bound, and volatile regimes. Six months gives you exposure to multiple regimes.
- **Emotional calibration:** Even with paper money, watching drawdowns teaches you how the system behaves under stress.
- **Bug discovery:** Six months of continuous operation surfaces edge cases in data handling, order execution, and risk management.

**Readiness checklist before going live:**

1. Paper traded for 6+ months continuously
2. Sharpe ratio above 0.5 (after transaction costs)
3. Maximum drawdown under 15%
4. At least 200 completed round-trip trades
5. Positive performance in at least 2 different market regimes
6. No system crashes or unhandled errors for 30+ days
7. Reviewed all losing trades to confirm they were managed correctly
8. Compared performance to SPY buy-and-hold benchmark

**Recommended progression:**

| Phase | Duration | Capital |
|---|---|---|
| Paper trading | 6-12 months | $100K simulated |
| Micro-live | 3 months | $1,000-5,000 real |
| Small-live | 3 months | $10,000-25,000 real |
| Full allocation | Ongoing | Your target amount |

Scale up only if each phase meets the performance criteria above.

## File Overview

| File | Purpose |
|---|---|
| `start_paper_trading.py` | Single launcher with pre-flight checks |
| `live_trader.py` | Core execution engine, connects to Alpaca |
| `paper_trading_dashboard.py` | Portfolio monitoring and performance tracking |
| `results_collector.py` | EOD data collection, feeds self-improver |
| `config.py` | All configuration (paper_trading section) |

## Logs and Data

All logs are saved to the `logs/` directory:

- `paper_trades.csv` — Every trade with timestamps, prices, strategy, regime
- `agent_YYYYMMDD.log` — Detailed execution log per day
- `performance.json` — Performance snapshots over time
- `dashboard_snapshots/` — Daily portfolio snapshots in JSON
- `daily_results/` — Per-day results from the results collector
- `strategy_performance_history.json` — Strategy weight evolution over time
- `self_improver_state.json` — Current weights and experience replay buffer

## Troubleshooting

**"Alpaca API keys not configured"** — Set the `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` environment variables. See Step 3.

**"alpaca-trade-api not installed"** — Run `pip install alpaca-trade-api`.

**"Market closed. Sleeping..."** — Normal. The trader only runs during US market hours (9:30 AM - 4:00 PM ET, weekdays).

**"Trading halted — max drawdown breached"** — The risk manager stopped trading because portfolio drawdown exceeded the configured limit (default 15%). Review positions and consider resetting with the results collector.

**Dashboard shows no data** — Run at least one trading cycle first. Use `--dry-run` if market is closed.
