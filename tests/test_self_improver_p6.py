"""
P6 tests (REVIEW_AND_IMPROVEMENTS.md §1.4, §2.3):
  - RL weight-learning is frozen until a meaningful trade sample exists
  - win/loss labelling is net-of-cost and direction-correct for shorts

Uses the shared `self_improver` fixture (temp save dir).
"""

from config import config


class TestLearningFreeze:
    def test_weights_frozen_with_no_data(self, self_improver):
        before = dict(self_improver.weights)
        after = self_improver.update_weights()
        assert after == before  # no movement on an empty sample

    def test_weights_move_once_sample_threshold_met(self, self_improver):
        before = dict(self_improver.weights)
        # Record enough closed round-trips to clear the freeze, all winning for
        # momentum so the posterior actually pulls weights off the prior.
        n = config.min_trades_for_learning
        for _ in range(n):
            self_improver.record_experience(
                symbol="AAPL", strategy="momentum", action="SELL",
                signal_strength=0.5, entry_price=100.0, exit_price=110.0,
            )
        after = self_improver.update_weights()
        # Learning is no longer frozen: weights moved, and the all-winning
        # strategy is now the highest-weighted one. (Absolute weight is still
        # pulled toward equal by James-Stein shrinkage at only 30 trades, which
        # is the intended variance control, so we check ranking, not magnitude.)
        assert after != before
        assert max(after, key=after.get) == "momentum"


class TestWinLabelling:
    def test_short_with_price_rise_is_a_loss(self, self_improver):
        b0 = self_improver.ts_beta["momentum"]
        self_improver.record_experience(
            symbol="X", strategy="momentum", action="SELL",
            signal_strength=0.0, entry_price=100.0, exit_price=110.0,
            is_short=True,                       # price ROSE on a short => loss
        )
        assert self_improver.ts_beta["momentum"] == b0 + 1

    def test_short_with_price_fall_is_a_win(self, self_improver):
        a0 = self_improver.ts_alpha["momentum"]
        self_improver.record_experience(
            symbol="X", strategy="momentum", action="SELL",
            signal_strength=0.0, entry_price=100.0, exit_price=90.0,
            is_short=True,                       # price FELL on a short => win
        )
        assert self_improver.ts_alpha["momentum"] == a0 + 1

    def test_cover_action_infers_short_direction(self, self_improver):
        # No is_short flag, but action=COVER must still be treated as a short.
        a0 = self_improver.ts_alpha["mean_reversion"]
        self_improver.record_experience(
            symbol="X", strategy="mean_reversion", action="COVER",
            signal_strength=0.0, entry_price=100.0, exit_price=90.0,
        )
        assert self_improver.ts_alpha["mean_reversion"] == a0 + 1

    def test_tiny_gain_below_cost_is_a_loss(self, self_improver):
        b0 = self_improver.ts_beta["sentiment"]
        # +0.05% gross < 0.1% round-trip cost => net loss.
        self_improver.record_experience(
            symbol="X", strategy="sentiment", action="SELL",
            signal_strength=0.0, entry_price=100.0, exit_price=100.05,
        )
        assert self_improver.ts_beta["sentiment"] == b0 + 1

    def test_real_gain_above_cost_is_a_win(self, self_improver):
        a0 = self_improver.ts_alpha["sentiment"]
        self_improver.record_experience(
            symbol="X", strategy="sentiment", action="SELL",
            signal_strength=0.0, entry_price=100.0, exit_price=101.0,  # +1%
        )
        assert self_improver.ts_alpha["sentiment"] == a0 + 1
