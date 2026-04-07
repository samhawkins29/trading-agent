"""Tests for data_fetcher.py — data fetching, indicators, cache, error handling."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestComputeIndicators:
    """Test DataFetcher.compute_indicators (static method)."""

    def test_sma_columns_present(self, sample_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        assert "SMA_20" in df.columns
        assert "SMA_50" in df.columns

    def test_ema_columns_present(self, sample_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        assert "EMA_12" in df.columns
        assert "EMA_26" in df.columns

    def test_macd_columns_present(self, sample_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        assert "MACD" in df.columns
        assert "MACD_signal" in df.columns
        assert "MACD_hist" in df.columns

    def test_rsi_column_present(self, sample_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        assert "RSI" in df.columns

    def test_bollinger_bands_present(self, sample_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        for col in ["BB_mid", "BB_upper", "BB_lower", "BB_pct"]:
            assert col in df.columns

    def test_atr_present(self, sample_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        assert "ATR" in df.columns

    def test_volume_indicators(self, sample_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        assert "Vol_SMA_20" in df.columns
        assert "Vol_ratio" in df.columns

    def test_returns_columns(self, sample_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        assert "returns" in df.columns
        assert "log_returns" in df.columns

    def test_does_not_modify_original(self, sample_ohlcv):
        from data_fetcher import DataFetcher

        original_cols = set(sample_ohlcv.columns)
        DataFetcher.compute_indicators(sample_ohlcv)
        assert set(sample_ohlcv.columns) == original_cols

    def test_sma20_manual_check(self, sample_ohlcv):
        """Verify SMA_20 against manual rolling mean calculation."""
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        idx = 50
        expected = sample_ohlcv["Close"].iloc[idx - 19 : idx + 1].mean()
        assert abs(df["SMA_20"].iloc[idx] - expected) < 1e-10

    def test_ema12_first_value_reasonable(self, sample_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        ema12 = df["EMA_12"].dropna()
        # EMA should be close to the price
        assert abs(ema12.iloc[-1] - df["Close"].iloc[-1]) < 10

    def test_rsi_range(self, sample_ohlcv):
        """RSI should be between 0 and 100."""
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        rsi = df["RSI"].dropna()
        assert (rsi >= 0).all()
        assert (rsi <= 100).all()

    def test_macd_is_ema12_minus_ema26(self, sample_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        # MACD = EMA_12 - EMA_26
        diff = (df["MACD"] - (df["EMA_12"] - df["EMA_26"])).dropna()
        assert (diff.abs() < 1e-10).all()

    def test_macd_hist_is_macd_minus_signal(self, sample_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        diff = (df["MACD_hist"] - (df["MACD"] - df["MACD_signal"])).dropna()
        assert (diff.abs() < 1e-10).all()

    def test_bb_upper_above_lower(self, sample_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        valid = df.dropna(subset=["BB_upper", "BB_lower"])
        assert (valid["BB_upper"] >= valid["BB_lower"]).all()

    def test_bb_mid_is_sma20(self, sample_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        diff = (df["BB_mid"] - df["SMA_20"]).dropna()
        assert (diff.abs() < 1e-10).all()

    def test_atr_positive(self, sample_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        atr = df["ATR"].dropna()
        assert (atr >= 0).all()

    def test_returns_first_is_nan(self, sample_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        assert pd.isna(df["returns"].iloc[0])

    def test_vol_ratio_around_one_for_stable_volume(self):
        """Constant volume should give vol_ratio ≈ 1.0 after warmup."""
        from data_fetcher import DataFetcher

        n = 50
        dates = pd.bdate_range("2024-01-02", periods=n)
        df = pd.DataFrame(
            {
                "Open": [100.0] * n,
                "High": [101.0] * n,
                "Low": [99.0] * n,
                "Close": [100.0] * n,
                "Volume": [5_000_000.0] * n,
            },
            index=dates,
        )
        result = DataFetcher.compute_indicators(df)
        vr = result["Vol_ratio"].dropna()
        assert abs(vr.iloc[-1] - 1.0) < 1e-10


class TestCacheBehavior:
    """Test the DataFetcher cache logic."""

    def test_cache_hit(self):
        from data_fetcher import DataFetcher

        fetcher = DataFetcher()
        df = pd.DataFrame({"Close": [1, 2, 3]})
        fetcher._set_cache("test_key", df)
        assert fetcher._is_cached("test_key")

    def test_cache_miss_unknown_key(self):
        from data_fetcher import DataFetcher

        fetcher = DataFetcher()
        assert not fetcher._is_cached("nonexistent")

    def test_cache_expiry(self):
        from data_fetcher import DataFetcher

        fetcher = DataFetcher()
        df = pd.DataFrame({"Close": [1, 2, 3]})
        fetcher._set_cache("old_key", df)
        # Manually backdate the timestamp
        fetcher._cache_ts["old_key"] = datetime.now() - timedelta(minutes=10)
        assert not fetcher._is_cached("old_key")

    def test_cache_returns_same_dataframe(self):
        from data_fetcher import DataFetcher

        fetcher = DataFetcher()
        df = pd.DataFrame({"Close": [10, 20, 30]})
        fetcher._set_cache("k", df)
        assert fetcher._cache["k"] is df


class TestGetHistoricalMocked:
    """Test get_historical with mocked yfinance."""

    @patch("data_fetcher.yf.Ticker")
    def test_get_historical_success(self, mock_ticker_class, sample_ohlcv):
        from data_fetcher import DataFetcher

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_ohlcv.copy()
        mock_ticker_class.return_value = mock_ticker

        fetcher = DataFetcher()
        df = fetcher.get_historical("AAPL", period="1y", interval="1d")
        assert not df.empty
        assert set(df.columns) == {"Open", "High", "Low", "Close", "Volume"}

    @patch("data_fetcher.yf.Ticker")
    def test_get_historical_empty_returns_fallback(self, mock_ticker_class):
        from data_fetcher import DataFetcher

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker

        fetcher = DataFetcher()
        # With default Alpha Vantage key, fallback returns empty
        df = fetcher.get_historical("AAPL")
        assert df.empty

    @patch("data_fetcher.yf.Ticker")
    def test_get_historical_exception_returns_fallback(self, mock_ticker_class):
        from data_fetcher import DataFetcher

        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = Exception("Network error")
        mock_ticker_class.return_value = mock_ticker

        fetcher = DataFetcher()
        df = fetcher.get_historical("AAPL")
        assert isinstance(df, pd.DataFrame)

    @patch("data_fetcher.yf.Ticker")
    def test_get_historical_uses_cache(self, mock_ticker_class, sample_ohlcv):
        from data_fetcher import DataFetcher

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = sample_ohlcv.copy()
        mock_ticker_class.return_value = mock_ticker

        fetcher = DataFetcher()
        df1 = fetcher.get_historical("AAPL", period="1y", interval="1d")
        df2 = fetcher.get_historical("AAPL", period="1y", interval="1d")
        # yfinance should only be called once
        assert mock_ticker.history.call_count == 1
        assert df1 is df2


class TestGetLatestPriceMocked:
    """Test get_latest_price with mocked yfinance."""

    @patch("data_fetcher.yf.Ticker")
    def test_latest_price_success(self, mock_ticker_class):
        from data_fetcher import DataFetcher

        mock_ticker = MagicMock()
        mock_data = pd.DataFrame({"Close": [150.0]})
        mock_ticker.history.return_value = mock_data
        mock_ticker_class.return_value = mock_ticker

        fetcher = DataFetcher()
        price = fetcher.get_latest_price("AAPL")
        assert price == 150.0

    @patch("data_fetcher.yf.Ticker")
    def test_latest_price_empty_returns_none(self, mock_ticker_class):
        from data_fetcher import DataFetcher

        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()
        mock_ticker_class.return_value = mock_ticker

        fetcher = DataFetcher()
        price = fetcher.get_latest_price("AAPL")
        # With default Alpaca key, fallback returns None
        assert price is None


class TestGetNewsMocked:
    """Test get_news with mocked requests."""

    def test_get_news_no_api_key(self):
        from data_fetcher import DataFetcher

        fetcher = DataFetcher()
        # Default key is placeholder, should return empty
        articles = fetcher.get_news("AAPL")
        assert articles == []

    @patch("data_fetcher.NEWS_API_KEY", "real_key")
    @patch("data_fetcher.requests.get")
    def test_get_news_success(self, mock_get):
        from data_fetcher import DataFetcher

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {
            "articles": [
                {
                    "title": "Test article",
                    "description": "desc",
                    "publishedAt": "2024-01-01",
                    "source": {"name": "TestSource"},
                }
            ]
        }
        mock_get.return_value = mock_resp

        fetcher = DataFetcher()
        articles = fetcher.get_news("AAPL")
        assert len(articles) == 1
        assert articles[0]["title"] == "Test article"

    @patch("data_fetcher.NEWS_API_KEY", "real_key")
    @patch("data_fetcher.requests.get")
    def test_get_news_network_error(self, mock_get):
        from data_fetcher import DataFetcher

        mock_get.side_effect = Exception("Timeout")

        fetcher = DataFetcher()
        articles = fetcher.get_news("AAPL")
        assert articles == []


class TestDataValidation:
    """Ensure computed indicators produce valid data."""

    def test_no_nan_in_close_column(self, sample_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        assert not df["Close"].isna().any()

    def test_no_nan_in_volume_column(self, sample_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        assert not df["Volume"].isna().any()

    def test_indicator_row_count_matches(self, sample_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        assert len(df) == len(sample_ohlcv)

    def test_all_expected_columns(self, sample_ohlcv):
        from data_fetcher import DataFetcher

        df = DataFetcher.compute_indicators(sample_ohlcv)
        expected = {
            "Open", "High", "Low", "Close", "Volume",
            "SMA_20", "SMA_50", "EMA_12", "EMA_26",
            "MACD", "MACD_signal", "MACD_hist",
            "RSI", "BB_mid", "BB_upper", "BB_lower", "BB_pct",
            "ATR", "Vol_SMA_20", "Vol_ratio",
            "returns", "log_returns",
        }
        assert expected.issubset(set(df.columns))
