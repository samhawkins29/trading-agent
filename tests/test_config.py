"""Tests for config.py — default values, parameter ranges, env var overrides."""

import os
import importlib

import pytest


class TestTradingConfigDefaults:
    """Verify all default values in TradingConfig."""

    def test_default_symbols(self):
        from config import TradingConfig

        cfg = TradingConfig()
        assert isinstance(cfg.symbols, list)
        assert len(cfg.symbols) == 10
        assert "AAPL" in cfg.symbols
        assert "SPY" in cfg.symbols

    def test_default_lookback_days(self):
        from config import TradingConfig

        cfg = TradingConfig()
        assert cfg.lookback_days == 252

    def test_default_rebalance_interval(self):
        from config import TradingConfig

        cfg = TradingConfig()
        assert cfg.rebalance_interval_minutes == 60

    def test_default_data_granularity(self):
        from config import TradingConfig

        cfg = TradingConfig()
        assert cfg.data_granularity == "1d"

    def test_default_risk_parameters(self):
        from config import TradingConfig

        cfg = TradingConfig()
        assert cfg.max_portfolio_pct_per_trade == 0.05
        assert cfg.max_total_exposure == 0.95
        assert cfg.max_drawdown_pct == 0.10
        assert cfg.stop_loss_pct == 0.03
        assert cfg.take_profit_pct == 0.08
        assert cfg.max_open_positions == 10
        assert cfg.max_daily_trades == 20

    def test_default_strategy_weights_sum_to_one(self):
        from config import TradingConfig

        cfg = TradingConfig()
        total = sum(cfg.strategy_weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_default_strategy_weights_keys(self):
        from config import TradingConfig

        cfg = TradingConfig()
        expected = {"mean_reversion", "momentum", "sentiment", "pattern_recognition"}
        assert set(cfg.strategy_weights.keys()) == expected

    def test_default_self_improvement_params(self):
        from config import TradingConfig

        cfg = TradingConfig()
        assert cfg.learning_rate == 0.01
        assert cfg.min_strategy_weight == 0.05
        assert cfg.max_strategy_weight == 0.60
        assert cfg.evaluation_window == 20
        assert cfg.experience_replay_size == 500

    def test_default_backtest_params(self):
        from config import TradingConfig

        cfg = TradingConfig()
        assert cfg.backtest_start == "2023-01-01"
        assert cfg.backtest_end == "2025-12-31"
        assert cfg.initial_capital == 100_000.0
        assert cfg.commission_per_trade == 0.0

    def test_default_logging_params(self):
        from config import TradingConfig

        cfg = TradingConfig()
        assert cfg.log_dir == "logs"
        assert cfg.log_level == "INFO"
        assert cfg.save_trades_csv is True


class TestParameterRanges:
    """Verify that parameters make logical sense."""

    def test_risk_pct_between_0_and_1(self):
        from config import TradingConfig

        cfg = TradingConfig()
        assert 0 < cfg.max_portfolio_pct_per_trade <= 1.0
        assert 0 < cfg.max_total_exposure <= 1.0
        assert 0 < cfg.max_drawdown_pct <= 1.0
        assert 0 < cfg.stop_loss_pct <= 1.0
        assert 0 < cfg.take_profit_pct <= 1.0

    def test_take_profit_greater_than_stop_loss(self):
        from config import TradingConfig

        cfg = TradingConfig()
        assert cfg.take_profit_pct > cfg.stop_loss_pct

    def test_strategy_weights_within_bounds(self):
        from config import TradingConfig

        cfg = TradingConfig()
        for w in cfg.strategy_weights.values():
            assert cfg.min_strategy_weight <= w <= cfg.max_strategy_weight

    def test_positive_learning_rate(self):
        from config import TradingConfig

        cfg = TradingConfig()
        assert 0 < cfg.learning_rate < 1.0

    def test_positive_capital(self):
        from config import TradingConfig

        cfg = TradingConfig()
        assert cfg.initial_capital > 0


class TestEnvVarOverrides:
    """Test that API keys can be overridden via environment variables."""

    def test_alpaca_api_key_default(self):
        import config

        assert config.ALPACA_API_KEY == os.getenv("ALPACA_API_KEY", "YOUR_ALPACA_API_KEY")

    def test_alpaca_secret_key_default(self):
        import config

        assert config.ALPACA_SECRET_KEY == os.getenv("ALPACA_SECRET_KEY", "YOUR_ALPACA_SECRET_KEY")

    def test_alpaca_base_url_default(self):
        import config

        assert "alpaca" in config.ALPACA_BASE_URL.lower()

    def test_alpha_vantage_key_default(self):
        import config

        assert config.ALPHA_VANTAGE_KEY == os.getenv("ALPHA_VANTAGE_KEY", "YOUR_ALPHA_VANTAGE_KEY")

    def test_news_api_key_default(self):
        import config

        assert config.NEWS_API_KEY == os.getenv("NEWS_API_KEY", "YOUR_NEWS_API_KEY")

    def test_env_override_alpaca_key(self, monkeypatch):
        monkeypatch.setenv("ALPACA_API_KEY", "test_key_123")
        importlib.reload(__import__("config"))
        import config

        assert config.ALPACA_API_KEY == "test_key_123"
        # Reset
        monkeypatch.delenv("ALPACA_API_KEY", raising=False)
        importlib.reload(__import__("config"))

    def test_singleton_config_exists(self):
        from config import config

        assert config is not None
        assert hasattr(config, "symbols")


class TestCustomConfig:
    """Test creating config with non-default values."""

    def test_custom_symbols(self):
        from config import TradingConfig

        cfg = TradingConfig(symbols=["TSLA", "NVDA"])
        assert cfg.symbols == ["TSLA", "NVDA"]

    def test_custom_capital(self):
        from config import TradingConfig

        cfg = TradingConfig(initial_capital=50_000.0)
        assert cfg.initial_capital == 50_000.0

    def test_custom_risk_params(self):
        from config import TradingConfig

        cfg = TradingConfig(max_drawdown_pct=0.05, stop_loss_pct=0.02)
        assert cfg.max_drawdown_pct == 0.05
        assert cfg.stop_loss_pct == 0.02
