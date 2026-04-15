"""
Configuration for the AI Trading Agent.
Redesigned with evidence-based parameters from quantitative research.

Key changes from v1:
  - Stop-loss widened from 3% to 6% (ATR-based, research shows 3% too tight)
  - Take-profit widened from 8% to 15%
  - Position sizing via half-Kelly criterion with vol targeting
  - Backtest period extended to 10 years (2015-2025)
  - Regime-based dynamic strategy weighting enabled
"""

import os
from dataclasses import dataclass, field
from typing import List

# --- API Keys ---------------------------------------------------------------
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# Claude AI decision layer (agent_brain.py) and weekly review (weekly_review.py)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Finnhub news integration (news_fetcher.py) — free tier, optional
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")


# --- Trading Universe -------------------------------------------------------
@dataclass
class TradingConfig:
    """Core trading parameters — evidence-based defaults."""

    # Symbols: diversified across sectors for pairs trading and factor exposure
    symbols: List[str] = field(
        default_factory=lambda: [
            # Large Cap Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'CRM',
            # Financials & Blue Chips
            'JPM', 'V', 'MA', 'GS', 'BAC', 'BRK-B',
            # Growth & High Beta
            'COIN', 'PLTR', 'SOFI', 'ARKK', 'MSTR',
            # Consumer & Industrial
            'COST', 'HD', 'DIS', 'BA', 'UNH',
            # Energy
            'XOM', 'CVX',
            # Major ETFs
            'SPY', 'QQQ', 'IWM', 'DIA',
            # Sector ETFs
            'XLK', 'XLF', 'XLE', 'XLV', 'XLI',
            # Cross-Asset (bonds, gold, commodities)
            'TLT', 'GLD', 'SLV', 'USO',
            # Uncorrelated / Cross-Asset Diversifiers
            'VXX', 'EEM', 'EFA', 'GBTC', 'IBIT', 'FXI', 'DBA', 'COPX', 'URA', 'SH', 'TQQQ',
        ]
    )

    # Timeframes
    lookback_days: int = 504           # 2 years of trading days (for 12m momentum)
    rebalance_interval_minutes: int = 60
    data_granularity: str = "1d"

    # -- Risk Parameters (research-based, tuned via backtest) --
    max_portfolio_pct_per_trade: float = 0.15   # Max 15% per position (core+satellite)
    max_total_exposure: float = 0.95
    max_drawdown_pct: float = 0.15              # Halt at 15% drawdown
    stop_loss_pct: float = 0.08                 # 8% stop loss (was 3% — too tight)
    take_profit_pct: float = 0.20               # 20% take profit
    drawdown_scale_threshold: float = 0.04      # Start reducing at 4% DD
    drawdown_severe_threshold: float = 0.10     # Severe reduction at 10% DD
    max_open_positions: int = 15
    max_daily_trades: int = 20

    # -- Kelly Criterion Position Sizing --
    use_kelly: bool = True
    kelly_fraction: float = 0.6          # 60% Kelly (slightly above half)
    kelly_min_trades: int = 30           # Min trades before Kelly kicks in
    kelly_lookback: int = 100            # Trades to look back for W and R

    # -- Volatility Targeting --
    vol_target: float = 0.15             # 15% annualized target volatility
    vol_lookback_days: int = 20          # Rolling window for realized vol
    vol_scale_min: float = 0.3           # Min position scale
    vol_scale_max: float = 2.0           # Max position scale (allows slight leverage)

    # -- Strategy Weights (initial — regime detector overrides dynamically) --
    strategy_weights: dict = field(
        default_factory=lambda: {
            "mean_reversion": 0.25,      # Statistical Arbitrage / Pairs
            "momentum": 0.30,            # Time Series Momentum
            "sentiment": 0.20,           # Factor Momentum (price-based factors)
            "pattern_recognition": 0.25, # Volatility-Regime Detection
        }
    )

    # -- Shorting --
    shorting_enabled: bool = True          # Allow brain to open short positions

    # -- Regime-Based Dynamic Weighting --
    use_regime_weighting: bool = True
    regime_blend_alpha: float = 0.3      # How much to blend regime weights vs base

    # -- Self-Improvement --
    learning_rate: float = 0.01
    min_strategy_weight: float = 0.05
    max_strategy_weight: float = 0.60
    evaluation_window: int = 20
    experience_replay_size: int = 1000   # Doubled from 500

    # -- Backtesting --
    backtest_start: str = "2015-01-01"   # 10 years of data
    backtest_end: str = "2025-12-31"
    initial_capital: float = 100_000.0
    commission_per_trade: float = 0.0

    # -- Leverage --
    leverage: dict = field(
        default_factory=lambda: {
            "mode": "vol_target",           # none, fixed, kelly, vol_target
            "fixed_multiplier": 3.0,        # For fixed mode
            "max_leverage": 5.0,            # Hard cap on leverage
            "vol_target_annual": 0.15,      # Annual vol target for vol_target mode
            "max_drawdown_trigger": 0.10,   # Drawdown that triggers circuit breaker
            "ramp_days": 5,                 # Days to ramp back after circuit breaker
            "funding_cost_annual": 0.02,    # Annual cost of leverage (spread over risk-free)
            "min_leverage": 0.5,            # Minimum leverage floor
        }
    )

    # -- Logging --
    log_dir: str = "logs"
    log_level: str = "INFO"
    save_trades_csv: bool = True

    # -- Paper Trading --
    paper_trading: dict = field(
        default_factory=lambda: {
            # Symbols to paper trade (defaults to main symbols if empty)
            "symbols": [
                # Large Cap Tech
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'CRM',
                # Financials & Blue Chips
                'JPM', 'V', 'MA', 'GS', 'BAC', 'BRK-B',
                # Growth & High Beta
                'COIN', 'PLTR', 'SOFI', 'ARKK', 'MSTR',
                # Consumer & Industrial
                'COST', 'HD', 'DIS', 'BA', 'UNH',
                # Energy
                'XOM', 'CVX',
                # Major ETFs
                'SPY', 'QQQ', 'IWM', 'DIA',
                # Sector ETFs
                'XLK', 'XLF', 'XLE', 'XLV', 'XLI',
                # Cross-Asset (bonds, gold, commodities)
                'TLT', 'GLD', 'SLV', 'USO',
                # Uncorrelated / Cross-Asset Diversifiers
                'VXX', 'EEM', 'EFA', 'GBTC', 'IBIT', 'FXI', 'DBA', 'COPX', 'URA', 'SH', 'TQQQ',
            ],
            # How often to run the trading cycle (minutes)
            "interval_minutes": 5,
            # Market hours (Eastern Time)
            "market_open_hour": 9,
            "market_open_minute": 30,
            "market_close_hour": 16,
            "market_close_minute": 0,
            # Minimum days of paper trading before considering real money
            "min_paper_days": 180,       # 6 months minimum
            # Performance thresholds to pass before going live
            "min_sharpe_for_live": 0.5,
            "max_drawdown_for_live": 0.15,
            "min_trades_for_live": 200,
            # End-of-day results collection time (ET, 24h format)
            "eod_collect_hour": 16,
            "eod_collect_minute": 15,
            # Dashboard refresh interval (seconds)
            "dashboard_refresh_seconds": 60,
        }
    )


# Single shared instance — importable as `from config import config`
config = TradingConfig()