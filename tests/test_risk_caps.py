"""
P5 risk-fix tests (REVIEW_AND_IMPROVEMENTS.md §2.1):
  - per-sector concentration cap
  - sane leverage defaults

Uses the shared `risk_manager` fixture (100k capital, temp logger).
"""

from config import config


class TestSectorExposureCap:
    def test_sector_of_known_and_unknown(self, risk_manager):
        assert risk_manager._sector_of("NVDA") == "tech"
        assert risk_manager._sector_of("QQQ") == "tech"      # tech-beta ETF
        assert risk_manager._sector_of("ZZZZ") == "other"    # fallback

    def test_tech_buy_blocked_when_sector_full_but_other_sector_ok(self, risk_manager):
        # Fill the tech bucket: 350 sh NVDA @ $100 = $35k > 0.40 * (remaining cash).
        risk_manager.open_position("NVDA", quantity=350, price=100.0,
                                   strategy="momentum", atr=0.0)

        tech_shares = risk_manager.calculate_position_size(
            "AAPL", price=100.0, signal_strength=1.0, volatility=0.15
        )
        energy_shares = risk_manager.calculate_position_size(
            "XOM", price=100.0, signal_strength=1.0, volatility=0.15
        )

        # Tech is concentration-capped; an uncorrelated sector is not.
        assert tech_shares == 0
        assert energy_shares > 0

    def test_empty_book_single_trade_not_constrained_by_sector(self, risk_manager):
        # With an empty book the 40% sector budget is far above the 10%
        # per-trade cap, so the sector cap must NOT bind a first position.
        shares = risk_manager.calculate_position_size(
            "AAPL", price=100.0, signal_strength=1.0, volatility=0.15
        )
        assert shares > 0

    def test_sector_exposure_excludes_shorts(self, risk_manager):
        risk_manager.open_short_position("NVDA", quantity=100, price=100.0,
                                         strategy="s", atr=0.0)
        # A short is a hedge, not added long concentration.
        assert risk_manager._sector_exposure("tech") == 0.0


class TestLeverageDefaults:
    def test_default_leverage_is_unlevered(self):
        # Edge is unproven — default must not lever the book.
        assert config.leverage["fixed_multiplier"] == 1.0
        assert config.leverage["max_leverage"] <= 2.0
