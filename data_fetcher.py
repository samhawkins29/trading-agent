"""
Market data retrieval using free APIs.
Supports yfinance (primary), Alpha Vantage (backup), and Alpaca for live data.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from config import (
    ALPHA_VANTAGE_KEY,
    ALPACA_API_KEY,
    ALPACA_BASE_URL,
    ALPACA_SECRET_KEY,
    NEWS_API_KEY,
    config,
)


class DataFetcher:
    """Fetches and caches market data from multiple sources."""

    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_ts: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=5)

    # ── Historical OHLCV ─────────────────────────────────────────────────
    def get_historical(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data via yfinance.
        Returns DataFrame with columns: Open, High, Low, Close, Volume.
        """
        cache_key = f"{symbol}_{period}_{interval}"
        if self._is_cached(cache_key):
            return self._cache[cache_key]

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                raise ValueError(f"No data returned for {symbol}")
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.dropna(inplace=True)
            self._set_cache(cache_key, df)
            return df
        except Exception as e:
            print(f"[DataFetcher] yfinance failed for {symbol}: {e}")
            return self._fallback_alpha_vantage(symbol)

    def get_multiple_historical(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for multiple symbols."""
        results = {}
        for sym in symbols:
            df = self.get_historical(sym, period, interval)
            if not df.empty:
                results[sym] = df
            time.sleep(0.1)  # rate-limit courtesy
        return results

    # ── Live / Latest Price ──────────────────────────────────────────────
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the most recent price for a symbol."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data["Close"].iloc[-1])
        except Exception:
            pass

        # Fallback: Alpaca
        return self._alpaca_latest_price(symbol)

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get latest prices for multiple symbols."""
        prices = {}
        for sym in symbols:
            p = self.get_latest_price(sym)
            if p is not None:
                prices[sym] = p
        return prices

    # ── News Sentiment Data ──────────────────────────────────────────────
    def get_news(self, query: str, days_back: int = 3) -> List[Dict]:
        """
        Fetch recent news articles via NewsAPI.
        Returns list of dicts with 'title', 'description', 'publishedAt'.
        """
        if NEWS_API_KEY == "YOUR_NEWS_API_KEY":
            return []

        from_date = (datetime.now() - timedelta(days=days_back)).strftime(
            "%Y-%m-%d"
        )
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": from_date,
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": 20,
            "apiKey": NEWS_API_KEY,
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            return [
                {
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "publishedAt": a.get("publishedAt", ""),
                    "source": a.get("source", {}).get("name", ""),
                }
                for a in articles
            ]
        except Exception as e:
            print(f"[DataFetcher] News API error: {e}")
            return []

    # ── Technical Indicators (computed from OHLCV) ───────────────────────
    @staticmethod
    def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add common technical indicators to a DataFrame."""
        df = df.copy()

        # Moving averages
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["SMA_50"] = df["Close"].rolling(50).mean()
        df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()

        # MACD
        df["MACD"] = df["EMA_12"] - df["EMA_26"]
        df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

        # RSI (14-period)
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df["BB_mid"] = df["Close"].rolling(20).mean()
        bb_std = df["Close"].rolling(20).std()
        df["BB_upper"] = df["BB_mid"] + 2 * bb_std
        df["BB_lower"] = df["BB_mid"] - 2 * bb_std
        df["BB_pct"] = (df["Close"] - df["BB_lower"]) / (
            df["BB_upper"] - df["BB_lower"]
        ).replace(0, np.nan)

        # ATR (Average True Range)
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(
            axis=1
        )
        df["ATR"] = true_range.rolling(14).mean()

        # Volume moving average
        df["Vol_SMA_20"] = df["Volume"].rolling(20).mean()
        df["Vol_ratio"] = df["Volume"] / df["Vol_SMA_20"].replace(0, np.nan)

        # Returns
        df["returns"] = df["Close"].pct_change()
        df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))

        return df

    # ── Private helpers ──────────────────────────────────────────────────
    def _is_cached(self, key: str) -> bool:
        if key in self._cache and key in self._cache_ts:
            return datetime.now() - self._cache_ts[key] < self._cache_ttl
        return False

    def _set_cache(self, key: str, df: pd.DataFrame):
        self._cache[key] = df
        self._cache_ts[key] = datetime.now()

    def _fallback_alpha_vantage(self, symbol: str) -> pd.DataFrame:
        """Fallback data source via Alpha Vantage."""
        if ALPHA_VANTAGE_KEY == "YOUR_ALPHA_VANTAGE_KEY":
            return pd.DataFrame()
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol,
                "outputsize": "full",
                "apikey": ALPHA_VANTAGE_KEY,
            }
            resp = requests.get(url, params=params, timeout=15)
            data = resp.json().get("Time Series (Daily)", {})
            if not data:
                return pd.DataFrame()

            rows = []
            for date_str, vals in data.items():
                rows.append({
                    "Date": pd.Timestamp(date_str),
                    "Open": float(vals["1. open"]),
                    "High": float(vals["2. high"]),
                    "Low": float(vals["3. low"]),
                    "Close": float(vals["4. close"]),
                    "Volume": int(vals["5. volume"]),
                })
            df = pd.DataFrame(rows).set_index("Date").sort_index()
            return df
        except Exception as e:
            print(f"[DataFetcher] Alpha Vantage fallback failed: {e}")
            return pd.DataFrame()

    def _alpaca_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price from Alpaca API."""
        if ALPACA_API_KEY == "YOUR_ALPACA_API_KEY":
            return None
        try:
            url = f"{ALPACA_BASE_URL}/v2/stocks/{symbol}/quotes/latest"
            headers = {
                "APCA-API-KEY-ID": ALPACA_API_KEY,
                "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
            }
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            quote = resp.json().get("quote", {})
            return float(quote.get("ap", 0))  # ask price
        except Exception:
            return None
