"""Tests for self_improver.py."""

import json
import os

import numpy as np
import pytest

from self_improver import SelfImprover, Experience


class TestExperienceReplayBuffer:
    """Test experience buffer add, sample, capacity."""

    def test_add_experience(self, self_improver):
        self_improver.record_experience(
            symbol="AAPL", strategy="momentum", action="BUY",
            signal_strength=0.7, entry_price=100.0, exit_price=105.0,
        )
        assert len(self_improver.replay_buffer) == 1

    def test_buffer_capacity(self, self_improver):
        from config import config

        for i in range(config.experience_replay_size + 50):
            self_improver.record_experience(
                symbol="AAPL", strategy="momentum", action="BUY",
                signal_strength=0.5, entry_price=100.0, exit_price=101.0,
            )
        assert len(self_improver.replay_buffer) == config.experience_replay_size

    def test_pnl_calculated_correctly(self, self_improver):
        self_improver.record_experience(
            symbol="AAPL", strategy="momentum", action="BUY",
            signal_strength=0.5, entry_price=100.0, exit_price=110.0,
        )
        exp = self_improver.replay_buffer[-1]
        expected_pnl = (110.0 - 100.0) / 100.0  # 0.10
        assert abs(exp.pnl - expected_pnl) < 1e-10

    def test_strategy_pnl_tracked(self, self_improver):
        self_improver.record_experience(
            symbol="AAPL", strategy="momentum", action="BUY",
            signal_strength=0.5, entry_price=100.0, exit_price=105.0,
        )
        assert len(self_improver.strategy_pnls["momentum"]) == 1

    def test_strategy_trade_count(self, self_improver):
        self_improver.record_experience(
            symbol="AAPL", strategy="momentum", action="BUY",
            signal_strength=0.5, entry_price=100.0, exit_price=105.0,
        )
        assert self_improver.strategy_trades["momentum"] == 1


class TestSampleExperiences:
    """Test experience replay sampling."""

    def test_sample_from_empty_buffer(self, self_improver):
        samples = self_improver.sample_experiences(n=10)
        assert samples == []

    def test_sample_single_experience(self, self_improver):
        self_improver.record_experience(
            symbol="AAPL", strategy="momentum", action="BUY",
            signal_strength=0.5, entry_price=100.0, exit_price=101.0,
        )
        samples = self_improver.sample_experiences(n=1)
        assert len(samples) == 1

    def test_sample_n_larger_than_buffer(self, self_improver):
        for _ in range(3):
            self_improver.record_experience(
                symbol="AAPL", strategy="momentum", action="BUY",
                signal_strength=0.5, entry_price=100.0, exit_price=101.0,
            )
        samples = self_improver.sample_experiences(n=100)
        assert len(samples) == 3

    def test_sample_by_strategy(self, self_improver):
        self_improver.record_experience(
            symbol="AAPL", strategy="momentum", action="BUY",
            signal_strength=0.5, entry_price=100.0, exit_price=101.0,
        )
        self_improver.record_experience(
            symbol="AAPL", strategy="mean_reversion", action="SELL",
            signal_strength=-0.5, entry_price=100.0, exit_price=95.0,
        )
        samples = self_improver.sample_experiences(n=10, strategy="momentum")
        assert all(s.strategy == "momentum" for s in samples)


class TestWeightUpdates:
    """Test weight update after positive/negative outcomes."""

    def _fill_buffer(self, improver, strategy, pnl_sign, n=25):
        for i in range(n):
            exit_p = 110.0 if pnl_sign > 0 else 90.0
            improver.record_experience(
                symbol="AAPL", strategy=strategy, action="BUY",
                signal_strength=0.5, entry_price=100.0, exit_price=exit_p,
            )

    def test_no_update_with_insufficient_data(self, self_improver):
        old_weights = dict(self_improver.weights)
        new_weights = self_improver.update_weights()
        assert new_weights == old_weights

    def test_weights_update_with_enough_data(self, self_improver):
        # Fill buffer with mixed results
        for strat in self_improver.weights:
            self._fill_buffer(self_improver, strat, 1, n=10)
        new_weights = self_improver.update_weights()
        assert isinstance(new_weights, dict)

    def test_winning_strategy_gains_weight(self, self_improver):
        """Strategy with all wins should gain weight relative to all-loss strategy."""
        # All wins for momentum
        for i in range(25):
            self_improver.record_experience(
                symbol="AAPL", strategy="momentum", action="BUY",
                signal_strength=0.5, entry_price=100.0,
                exit_price=100.0 + np.random.uniform(1, 10),
            )
        # All losses for mean_reversion
        for i in range(25):
            self_improver.record_experience(
                symbol="AAPL", strategy="mean_reversion", action="BUY",
                signal_strength=0.5, entry_price=100.0,
                exit_price=100.0 - np.random.uniform(1, 10),
            )
        # Some data for others
        for strat in ["sentiment", "pattern_recognition"]:
            for i in range(10):
                self_improver.record_experience(
                    symbol="AAPL", strategy=strat, action="BUY",
                    signal_strength=0.5, entry_price=100.0, exit_price=100.5,
                )

        old_momentum = self_improver.weights["momentum"]
        self_improver.update_weights()
        # Momentum should maintain or gain weight
        assert self_improver.weights["momentum"] >= old_momentum - 0.01


class TestSoftmaxNormalization:
    """Test that weights are properly normalized."""

    def test_weights_sum_to_one(self, self_improver):
        total = sum(self_improver.weights.values())
        assert abs(total - 1.0) < 0.01

    def test_enforce_constraints(self, self_improver):
        from config import config

        weights = {"momentum": 0.01, "mean_reversion": 0.90, "sentiment": 0.04, "pattern_recognition": 0.05}
        constrained = self_improver._enforce_constraints(weights)
        # After clipping then normalizing, all weights should be >= min
        for v in constrained.values():
            assert v >= config.min_strategy_weight - 1e-6
        # Sum should be 1.0
        assert abs(sum(constrained.values()) - 1.0) < 1e-6
        # The extreme value (0.90) should be reduced after normalization
        assert constrained["mean_reversion"] < 0.90

    def test_weights_normalized_after_update(self, self_improver):
        for strat in self_improver.weights:
            for _ in range(10):
                self_improver.record_experience(
                    symbol="AAPL", strategy=strat, action="BUY",
                    signal_strength=0.5, entry_price=100.0, exit_price=102.0,
                )
        self_improver.update_weights()
        total = sum(self_improver.weights.values())
        assert abs(total - 1.0) < 0.01


class TestLearningRateEffects:
    """Test that learning rate controls speed of weight changes."""

    def test_small_lr_small_changes(self, self_improver):
        from config import config

        old_lr = config.learning_rate
        config.learning_rate = 0.001
        old_weights = dict(self_improver.weights)

        for strat in self_improver.weights:
            for _ in range(10):
                self_improver.record_experience(
                    symbol="AAPL", strategy=strat, action="BUY",
                    signal_strength=0.5, entry_price=100.0, exit_price=102.0,
                )
        self_improver.update_weights()

        max_change = max(
            abs(self_improver.weights[k] - old_weights[k])
            for k in self_improver.weights
        )
        assert max_change < 0.05  # small learning rate → small changes
        config.learning_rate = old_lr  # restore


class TestPersistence:
    """Test state save/load."""

    def test_save_state(self, self_improver, tmp_log_dir):
        self_improver.record_experience(
            symbol="AAPL", strategy="momentum", action="BUY",
            signal_strength=0.5, entry_price=100.0, exit_price=105.0,
        )
        self_improver._save_state()
        path = os.path.join(tmp_log_dir, "self_improver_state.json")
        assert os.path.exists(path)
        with open(path) as f:
            state = json.load(f)
        assert "weights" in state
        assert "replay_buffer" in state
        assert len(state["replay_buffer"]) == 1

    def test_load_state(self, trade_logger, tmp_log_dir):
        # Create and save
        imp1 = SelfImprover(logger=trade_logger, save_path=tmp_log_dir)
        imp1.record_experience(
            symbol="AAPL", strategy="momentum", action="BUY",
            signal_strength=0.5, entry_price=100.0, exit_price=105.0,
        )
        imp1.weights["momentum"] = 0.50
        imp1._save_state()

        # Load into new instance
        imp2 = SelfImprover(logger=trade_logger, save_path=tmp_log_dir)
        assert len(imp2.replay_buffer) == 1
        assert imp2.weights["momentum"] == 0.50


class TestEdgeCases:
    """Edge cases."""

    def test_all_same_outcome(self, self_improver):
        for strat in self_improver.weights:
            for _ in range(10):
                self_improver.record_experience(
                    symbol="AAPL", strategy=strat, action="BUY",
                    signal_strength=0.5, entry_price=100.0, exit_price=105.0,
                )
        # All strategies perform the same → weights should stay roughly equal
        self_improver.update_weights()
        vals = list(self_improver.weights.values())
        assert max(vals) - min(vals) < 0.3

    def test_report_structure(self, self_improver):
        report = self_improver.get_report()
        assert "current_weights" in report
        assert "total_experiences" in report
        assert "strategy_stats" in report
