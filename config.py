"""
Configuration for the AI Trading Agent.
Fill in your API keys and adjust parameters before running.
"""

import os
from dataclasses import dataclass, field
from typing import List

# ─── API Keys ────────────────────────────────────────────────────────────────
# Alpaca (free paper trading): https://alpaca.markets/
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "YOUR_ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "YOUR_ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv(
    "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"  # paper trading
)

# Alpha Vantage (free tier: 25 req/day): https://www.alphavantage.co/
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "YOUR_ALPHA_VANTAGE_KEY")

# News API (free tier): https://newsapi.org/
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY")


# ─── Trading Universe ────────────────────────────────────────────────────────
@dataclass
class TradingConfig:
    """Core trading parameters."""

    # Symbols to trade (start small, expand as confidence grows)
    symbols: List[str] = field(
        default_factory=lambda: [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "NVDA", "TSLA", "JPM", "V", "SPY",
        ]
    )

    # Timeframes
    lookback_days: int = 252          # 1 year of trading days
    rebalance_interval_minutes: int = 60  # how often the agent re-evaluates
    data_granularity: str = "1d"       # '1m', '5m', '15m', '1h', '1d'

    # ── Risk Parameters ──────────────────────────────────────────────────
    max_portfolio_pct_per_trade: float = 0.05   # max 5% of portfolio per position
    max_total_exposure: float = 0.95            # max 95% invested at once
    max_drawdown_pct: float = 0.10              # halt trading if drawdown > 10%
    stop_loss_pct: float = 0.03                 # per-position stop loss 3%
    take_profit_pct: float = 0.08               # per-position take profit 8%
    max_open_positions: int = 10
    max_daily_trades: int = 20

    # ── Strategy Weights (sum to 1.0) ────────────────────────────────────
    # These are the INITIAL weights; the self-improver adjusts them over time
    strategy_weights: dict = field(
        default_factory=lambda: {
            "mean_reversion": 0.30,
            "momentum": 0.30,
            "sentiment": 0.15,
            "pattern_recognition": 0.25,
        }
    )

    # ── Self-Improvement ─────────────────────────────────────────────────
    learning_rate: float = 0.01          # how fast weights shift
    min_strategy_weight: float = 0.05    # never let a strategy drop below 5%
    max_strategy_weight: float = 0.60    # never let a strategy exceed 60%
    evaluation_window: int = 20          # trades to look back for performance
    experience_replay_size: int = 500    # max stored experiences

    # ── Backtesting ──────────────────────────────────────────────────────
    backtest_start: str = "2023-01-01"
    backtest_end: str = "2025-12-31"
    initial_capital: float = 100_000.0
    commission_per_trade: float = 0.0    # Alpaca is commission-free

    # ── Logging ──────────────────────────────────────────────────────────
    log_dir: str = "logs"
    log_level: str = "INFO"
    save_trades_csv: bool = True


# Singleton instance
config = TradingConfig()
