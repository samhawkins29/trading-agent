"""
Shared fixtures for the AI Trading Agent test suite.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on sys.path so imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── Price Data Fixtures ─────────────────────────────────────────────────


@pytest.fixture
def sample_ohlcv():
    """Return a 100-row OHLCV DataFrame with realistic-looking price data."""
    np.random.seed(42)
    n = 100
    dates = pd.bdate_range("2024-01-02", periods=n)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    opn = close + np.random.randn(n) * 0.2
    volume = np.random.randint(1_000_000, 10_000_000, size=n).astype(float)
    df = pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    return df


@pytest.fixture
def sample_ohlcv_with_indicators(sample_ohlcv):
    """OHLCV DataFrame with all technical indicators already computed."""
    from data_fetcher import DataFetcher

    return DataFetcher.compute_indicators(sample_ohlcv)


@pytest.fixture
def flat_ohlcv():
    """100-row OHLCV where price is completely flat (edge case)."""
    n = 100
    dates = pd.bdate_range("2024-01-02", periods=n)
    price = 100.0
    df = pd.DataFrame(
        {
            "Open": [price] * n,
            "High": [price] * n,
            "Low": [price] * n,
            "Close": [price] * n,
            "Volume": [5_000_000.0] * n,
        },
        index=dates,
    )
    return df


@pytest.fixture
def trending_up_ohlcv():
    """100-row strongly upward-trending OHLCV."""
    np.random.seed(7)
    n = 100
    dates = pd.bdate_range("2024-01-02", periods=n)
    close = 100.0 + np.arange(n) * 0.5 + np.random.randn(n) * 0.1
    high = close + 0.3
    low = close - 0.3
    opn = close - 0.1
    volume = np.full(n, 5_000_000.0)
    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def trending_down_ohlcv():
    """100-row strongly downward-trending OHLCV."""
    np.random.seed(8)
    n = 100
    dates = pd.bdate_range("2024-01-02", periods=n)
    close = 150.0 - np.arange(n) * 0.5 + np.random.randn(n) * 0.1
    high = close + 0.3
    low = close - 0.3
    opn = close + 0.1
    volume = np.full(n, 5_000_000.0)
    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


@pytest.fixture
def short_ohlcv():
    """Only 5 rows — below the minimum for most strategies."""
    dates = pd.bdate_range("2024-01-02", periods=5)
    return pd.DataFrame(
        {
            "Open": [100, 101, 102, 101, 103],
            "High": [101, 102, 103, 102, 104],
            "Low": [99, 100, 101, 100, 102],
            "Close": [100.5, 101.5, 102.5, 101.5, 103.5],
            "Volume": [1e6, 1e6, 1e6, 1e6, 1e6],
        },
        index=dates,
    )


@pytest.fixture
def extreme_move_ohlcv():
    """Data with a huge gap / extreme move at the end."""
    np.random.seed(99)
    n = 100
    dates = pd.bdate_range("2024-01-02", periods=n)
    close = np.full(n, 100.0)
    close[-1] = 150.0  # +50% single-day move
    high = close + 1.0
    low = close - 1.0
    opn = close - 0.5
    volume = np.full(n, 5_000_000.0)
    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


# ── News / Sentiment Fixtures ───────────────────────────────────────────


@pytest.fixture
def positive_articles():
    """List of clearly positive news articles."""
    return [
        {"title": "Company beats earnings expectations", "description": "Revenue surged to record high with strong growth"},
        {"title": "Analyst upgrade: outperform rating", "description": "Bullish outlook with profit gains and innovation"},
    ]


@pytest.fixture
def negative_articles():
    """List of clearly negative news articles."""
    return [
        {"title": "Company misses earnings, shares crash", "description": "Revenue declined amid recession fears and layoffs"},
        {"title": "Analyst downgrade amid fraud investigation", "description": "Bearish outlook with loss and bankruptcy risk"},
    ]


@pytest.fixture
def neutral_articles():
    """List of neutral articles."""
    return [
        {"title": "Company holds annual meeting", "description": "The board discussed various topics today."},
    ]


# ── Logger / Temp Dir Fixtures ───────────────────────────────────────────


@pytest.fixture
def tmp_log_dir(tmp_path):
    """Temporary directory for log files."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return str(log_dir)


@pytest.fixture
def trade_logger(tmp_log_dir):
    """A TradeLogger instance writing to a temp directory."""
    from logger import TradeLogger

    return TradeLogger(log_dir=tmp_log_dir)


@pytest.fixture
def risk_manager(trade_logger):
    """A RiskManager with 100k capital and a temp logger."""
    from risk_manager import RiskManager

    return RiskManager(initial_capital=100_000.0, logger=trade_logger)


@pytest.fixture
def self_improver(trade_logger, tmp_log_dir):
    """A SelfImprover writing state to a temp directory."""
    from self_improver import SelfImprover

    return SelfImprover(logger=trade_logger, save_path=tmp_log_dir)
