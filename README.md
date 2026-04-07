# AI Trading Agent

A multi-strategy, self-improving algorithmic trading agent for paper trading via Alpaca. Inspired by approaches used at Renaissance Technologies, Two Sigma, DE Shaw, and Citadel.

## Architecture

```
main.py                    Entry point (live, backtest, single, status)
  |
  +-- agent.py             Central decision engine
  |     |
  |     +-- data_fetcher.py      Market data (yfinance, Alpha Vantage, Alpaca)
  |     +-- strategies/
  |     |     +-- mean_reversion.py      Stat-arb / Bollinger / Z-score
  |     |     +-- momentum.py            Trend following / MACD / ROC
  |     |     +-- sentiment.py           News sentiment scoring
  |     |     +-- pattern_recognition.py Candlestick / regime detection
  |     +-- risk_manager.py   Position sizing, stop-loss, drawdown
  |     +-- self_improver.py  RL weight adjustment, experience replay
  |     +-- logger.py         CSV trade log, JSON performance log
  |
  +-- backtester.py        Historical simulation
  +-- config.py            All settings and API keys
```

## Quick Start

### 1. Install dependencies

```bash
cd trading-agent
pip install -r requirements.txt
```

### 2. Configure API keys

Edit `config.py` or set environment variables:

```bash
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"
export NEWS_API_KEY="your_key"           # optional, for sentiment
export ALPHA_VANTAGE_KEY="your_key"      # optional, backup data
```

**Free API accounts:**
- Alpaca (paper trading): https://alpaca.markets
- NewsAPI: https://newsapi.org
- Alpha Vantage: https://www.alphavantage.co

### 3. Run backtest first

```bash
python main.py backtest
python main.py backtest --symbols AAPL,MSFT,GOOGL --start 2024-01-01 --end 2025-12-31
```

### 4. Run live paper trading

```bash
python main.py live
```

### 5. Other commands

```bash
python main.py single    # one cycle only (for cron jobs)
python main.py status    # current agent state
```

## Strategies

| Strategy | Approach | Inspiration |
|----------|----------|-------------|
| Mean Reversion | Z-score, Bollinger Bands, RSI extremes | Renaissance / Medallion stat-arb |
| Momentum | MA crossovers, MACD, Rate of Change | Two Sigma systematic trend-following |
| Sentiment | News keyword scoring, weighted recency | Citadel real-time news analysis |
| Pattern Recognition | Candlesticks, support/resistance, regime | RenTech pattern detection, HMM |

## Self-Improvement Loop

The agent continuously learns from its trades:

1. Every trade result is stored in an experience replay buffer
2. Every 5 cycles, strategy weights are re-evaluated
3. Weights shift toward strategies with better risk-adjusted returns (Sharpe-like metric)
4. Min/max constraints prevent any single strategy from dominating or disappearing

## Risk Management

- Per-position sizing: volatility-adjusted, max 5% of portfolio
- Stop-loss: ATR-based or 3% fixed (whichever is tighter)
- Take-profit: ATR-based or 8% fixed
- Max drawdown circuit breaker: 10% (halts all trading)
- Max 10 open positions, max 20 trades/day
- Max 95% portfolio exposure

## Output Files

```
logs/
  agent_YYYYMMDD.log   Console-style log
  trades.csv           Every trade with timestamps
  performance.json     Periodic snapshots
  self_improver_state.json   Learned weights + replay buffer
logs/backtest/
  results.json         Backtest metrics
```

## Disclaimer

This is an educational project for paper trading only. It is not financial advice. Past performance in backtests does not guarantee future results. Never trade with money you cannot afford to lose.
