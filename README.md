# AI Trading Agent

A self-improving, multi-strategy automated trading agent that runs on **Alpaca paper trading**. It blends four classic quantitative strategies (mean reversion, momentum, sentiment, pattern/regime), passes their combined signal through a Claude decision layer, sizes and risk-checks each order, executes against Alpaca's paper API, and feeds the results back into a learning loop that adjusts strategy weights and parameters over time. It is an educational research system — a sandbox for studying systematic trading end-to-end with real market data and a real (paper) broker.

> ⚠️ **THIS IS PAPER TRADING ONLY. DO NOT RUN IT WITH REAL MONEY.** See the [Safety & Current State](#safety--current-state) section below — the live paper account is currently **down**, the edge is unproven, and several mandatory guardrails are still incomplete.

---

## Why it exists

The goal is to build, in the open, a complete systematic-trading pipeline — data → signals → decision → risk → execution → logging → self-improvement — and to learn what actually survives contact with a live (paper) broker, as opposed to what merely looks good in a backtest. The project deliberately keeps an honest paper-trading record and a candid internal review ([`REVIEW_AND_IMPROVEMENTS.md`](REVIEW_AND_IMPROVEMENTS.md)) precisely because the gap between polished backtests and real behavior is the most interesting and important thing to study. It is a place to experiment with regime-aware weighting, Kelly/vol position sizing, an LLM decision overlay, and reinforcement-style weight learning — not a product, and not financial advice.

---

## Getting it running

### Prerequisites

- **Python 3.10+** (uses dataclasses, `from __future__`-style typing, f-strings throughout).
- An **Alpaca paper trading** account: <https://alpaca.markets> (free).
- Optional API keys for richer behavior (see below).

### Install

```bash
cd trading-agent
pip install -r requirements.txt
```

Core dependencies: `yfinance`, `pandas`, `numpy`, `requests`, `python-dotenv`, `alpaca-trade-api`, `pytest`. `alpaca-trade-api` is only required for live paper trading — `--dry-run` works without it.

### Configure API keys

Copy the example env file and fill it in. **`.env` is gitignored and must never be committed.**

```bash
cp .env.example .env
```

```ini
# Alpaca paper trading (required for live paper trading)
ALPACA_API_KEY=your-key
ALPACA_SECRET_KEY=your-secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Claude decision layer (optional) — Sonnet brain + Opus weekly review
ANTHROPIC_API_KEY=your-key

# News (optional) — Finnhub preferred (free tier), NewsAPI as fallback
FINNHUB_API_KEY=your-key
NEWS_API_KEY=your-key

# OHLCV data fallback (optional)
ALPHA_VANTAGE_KEY=your-key
```

Everything is read from environment variables via `python-dotenv` in [`config.py`](config.py); missing optional keys degrade gracefully (no Claude brain → falls back to a threshold rule; no news key → no headlines; no Alpha Vantage → yfinance only).

### Run

**Recommended entry point — the paper trading launcher** ([`start_paper_trading.py`](start_paper_trading.py)), which runs pre-flight checks (dependencies, strategy loading, Alpaca connectivity, config summary) before starting:

```bash
python start_paper_trading.py --status     # Pre-flight: print system status and exit (no trading)
python start_paper_trading.py --dry-run    # Full pipeline, no real API orders (no keys needed)
python start_paper_trading.py --once       # Run a single live cycle and exit
python start_paper_trading.py              # Start the continuous live paper-trading loop
```

Always start with `--status` and `--dry-run` to confirm the system is wired correctly before doing anything live.

**Alternative entry point** ([`main.py`](main.py)) — note this drives the simpler `agent.py` engine (see [How it works](#how-it-works)):

```bash
python main.py status                      # Show agent status
python main.py single                      # Run one cycle (useful for cron)
python main.py live                        # Continuous live paper loop
python main.py backtest                    # Historical backtest
python main.py backtest --symbols AAPL,MSFT --start 2024-01-01 --end 2025-12-31
```

On Windows, `start-trading-bot.bat` and `start-trading-bot-status.bat` wrap the launcher commands.

**Market-hours behavior:** the live loop checks U.S. market hours (Mon–Fri, 9:30–16:00 ET by default, configurable in `config.paper_trading`). When the market is open it runs a cycle every `interval_minutes` (default 5). When closed, it sleeps and waits for the next session rather than trading. Broker-native resting stops (below) are submitted GTC so they protect positions even while the loop is idle overnight/weekends.

---

## What it does

### Strategies

Four strategies run every cycle on each symbol; each returns a `Signal` (action + strength in `[-1, 1]` + reason):

| Strategy | File | Approach |
|----------|------|----------|
| **Mean Reversion** | `strategies/mean_reversion.py` | Z-score / Bollinger Bands / RSI extremes (statistical-arbitrage style) |
| **Momentum** | `strategies/momentum.py` | Multi-timeframe trend following, MA crossovers, MACD, rate-of-change |
| **Sentiment** | `strategies/sentiment.py` | Price-based factor momentum (the older keyword text-sentiment path was removed) |
| **Pattern Recognition** | `strategies/pattern_recognition.py` | Candlesticks, support/resistance, and **market-regime detection** (drives dynamic weighting) |

Three additional strategy files exist (`dual_momentum.py`, `cross_asset_signals.py`, `adaptive_allocation.py`) but are **not wired into the live engine** — they are experimental/orphan modules.

### Self-improvement loop

The system learns at two cadences:

- **Per-trade RL** ([`self_improver.py`](self_improver.py)) — every closed trade is recorded in an experience-replay buffer. Strategy weights are updated with **Thompson Sampling** (Beta posteriors per strategy) plus **James–Stein shrinkage** toward equal weights, with per-regime weight profiles. Wins/losses are computed **net of an estimated round-trip cost and corrected for short direction**. Learning is **frozen until at least `min_trades_for_learning` (30) closed round-trips** exist, to avoid curve-fitting to a handful of trades.
- **Weekly meta-review** ([`weekly_review.py`](weekly_review.py)) — Claude **Opus** reads the trade journal and executed trades, then writes `learned_params.json` (strategy weights, buy/sell thresholds, stop-loss %, take-profit %) plus a human-readable memo in `weekly_reviews/`. These learned parameters are loaded on the next start and applied (bounds-checked) to the live config.
- **Daily review** ([`daily_review.py`](daily_review.py)) and an **EOD results collector** ([`results_collector.py`](results_collector.py)) add end-of-day analysis and feedback.

### Execution

Paper trades are executed against Alpaca via passive limit orders (priced ±0.1%, polled up to ~2 minutes, falling back to market after repeated cancels), and every fill is logged to `logs/paper_trades.csv` with the real filled price. A dashboard ([`paper_trading_dashboard.py`](paper_trading_dashboard.py)) tracks portfolio value, drawdown, win rate, and a running Sharpe vs. SPY.

---

## How it works

### Single merged signal engine

After the engine-merge work on this branch, the signal pipeline lives in **one shared library**, [`signal_engine.py`](signal_engine.py) (`SignalEngine`):

```
data → indicators → regime detection → regime-weighted aggregation → combined Signal
```

`SignalEngine` is pure and stateless — it touches no broker and holds no positions. The execution engine, [`live_trader.py`](live_trader.py) (`LiveTrader`), is what actually paper-trades and owns all stateful concerns: Alpaca connection, position reconciliation, the Claude brain, risk/leverage, native stops, the kill switch, and logging. It delegates regime detection, weight blending, and combined-signal generation to `SignalEngine`.

`agent.py` remains as the simpler, heavily-tested "library" engine used by `main.py` and most of the test suite. It shares the **same `RiskManager` and `config` singleton** as `live_trader.py`, so risk-core behavior (position sizing, sector caps, stop levels, Kelly) is identical across both. Honest caveat: `live_trader.py` is the path that trades, and collapsing the two engines into a single execution path with shared logic is still a tracked follow-up (`REVIEW_AND_IMPROVEMENTS.md` §1.3).

### Claude decision layer

When `ANTHROPIC_API_KEY` is set, [`agent_brain.py`](agent_brain.py) (`AgentBrain`, **claude-sonnet-4-6**) is called each cycle: it packages the per-strategy signals, portfolio state, regime, and optional news headlines into a prompt and returns structured BUY/SELL/SHORT decisions with confidence and rationale. Every decision is journaled for the weekly Opus review. If the key is missing or the call fails, the system **falls back to a deterministic signal-threshold rule** — the brain is an overlay, not a hard dependency.

### Risk, stops, and circuit breakers

Managed by [`risk_manager.py`](risk_manager.py) and `live_trader.py`, driven by `config.py`:

- **Position sizing:** Kelly fraction (`kelly_fraction=0.6`, min 30 trades) × volatility targeting (15% annualized) × regime adjustment, capped at **10% of portfolio per trade**.
- **Concentration:** **per-sector exposure cap** (`max_sector_exposure=0.40`) via `config.sector_map`, which deliberately buckets leveraged/tech ETFs with single-name tech so a "diversified" book can't secretly be one tech-beta bet.
- **Exposure / cash reserve:** total levered exposure capped at **0.80**, enforcing a ≥20% cash buffer.
- **Broker-native resting stops:** with `use_native_stops=True`, every opened position gets a **GTC stop order resting at Alpaca** immediately after fill — so it's protected overnight, on weekends, and during agent downtime, not just when a cycle polls. Stops are cancelled/resynced on close or scale-out.
- **Soft drawdown halt:** trading halts at 15% drawdown, computed on **real Alpaca equity**.
- **Real-time kill switch:** an independent circuit breaker (`config.kill_switch`) checked at the top of every cycle against real broker equity. On a trip (peak-to-trough ≥20%, intraday loss ≥8%, or equity floor) it **cancels all orders, liquidates all positions, and writes a persistent `logs/KILL_SWITCH.flag`** that survives restarts — the agent stays halted until the flag is manually removed. (Note: it is real-time *per cycle*; a fully independent always-on watchdog thread is still TODO.)
- **Position reconciliation:** on startup, existing Alpaca positions are reconciled into local state so the duplicate-buy guard and exposure accounting reflect reality.
- **Leverage** ([`leverage_manager.py`](leverage_manager.py)): defaults to **1.0× (unlevered)**, hard cap 2.0×, with fixed/Kelly/vol-target modes and a drawdown circuit breaker.

### Learning loop wiring

`learned_params.json` is written by the weekly Opus review and read at startup by `live_trader._apply_learned_params`, which applies the strategy weights, buy/sell thresholds, **and** `stop_loss_pct`/`take_profit_pct` onto the shared `config` singleton (bounds-checked and logged). Earlier, the stop/take-profit values were written but silently dropped; that disconnect has been fixed so the system no longer reports risk changes that never took effect.

### Data

[`data_fetcher.py`](data_fetcher.py) uses **yfinance** as the primary OHLCV source with an **Alpha Vantage** fallback, and Alpaca for live prices. Indicators (ATR, RSI, MACD, Bollinger, etc.) are computed centrally. Data-quality guards (staleness/spike checks, unified adjustment semantics) are a known gap.

### Tests

A `pytest` suite under `tests/` covers strategies, risk manager, self-improver, backtester (including integrity checks), the signal engine, and live-path modules:

```bash
python -m pytest -q
```

Some assertions are known to be stale relative to current config values (catalogued in the review); treat the suite as broad structural coverage, and note that the full real-money execution path remains less covered than the library engine.

---

## Safety & Current State

**This system is for PAPER TRADING and is NOT ready for real money.** This is the most important section — read it before doing anything.

### Honest current state

- The internal review ([`REVIEW_AND_IMPROVEMENTS.md`](REVIEW_AND_IMPROVEMENTS.md), 2026-06-10) found the **real paper account has drifted negative — roughly −2.75% over ~2 months** (from $100,000). The latest logged portfolio value (`logs/paper_trades.csv`, 2026-06-12) is about **$96,228 (≈ −3.8%)**. **There is no demonstrated edge.**
- **Do not trust the headline backtests.** Two of the three backtests run on *synthetic* price data, and the real-data backtester has structural optimism (same-bar look-ahead and survivorship bias were the original issues; next-bar fills + commission/slippage were since added, and synthetic runs are now tagged with a trust warning). The live paper record is the only honest performance signal, and it is negative.
- The **self-improvement loop has very little data to learn from** — recent weekly reviews covered only a handful of closed trades. Learning is now frozen below a minimum sample, but the parameters should still be treated as essentially un-validated.

### Guardrails that MUST exist (and be verified live) before ANY real stakes

1. **Broker-native stops verified working live** on every position — resting at the exchange, not just polled per cycle. (Implemented via GTC stops; must be confirmed end-to-end against a live account.)
2. **A real-time max-loss kill switch** (per-day and peak-to-trough) that liquidates and halts independently of the cycle loop. (Per-cycle + persistent-flag version exists; a fully independent watchdog is still TODO.)
3. **A tested live execution path** — order submission, partial fills, reconciliation-after-restart, short cover, and drawdown halt all covered by tests against a mocked broker, with the live and tested paths unified.
4. **No overfit / a real out-of-sample, cost-and-slippage-inclusive positive track record** over a statistically meaningful number of *closed* trades — before a single real dollar.
5. If ever taken live: start at a tiny fraction of capital, leverage = 1.0×, and scale only after the live edge is demonstrated **net of costs**.

> Until all of the above are met and the paper record turns convincingly positive net of costs, **run this with paper credentials only.** Never trade money you cannot afford to lose. This is an educational project and not financial advice; backtest results do not guarantee future performance.
