"""
P4: test coverage for the previously-untested live decision/learning modules
(REVIEW_AND_IMPROVEMENTS.md §2.7): agent_brain, leverage_manager, weekly_review.

All offline — no API calls. agent_brain and weekly_review are exercised through
their pure parsing functions; leverage_manager through its pure math.
"""

import json

import pytest

from agent_brain import AgentBrain
from leverage_manager import LeverageManager, LeverageConfig
import weekly_review


# ── agent_brain: brain-JSON parsing edge cases ───────────────────────────────

class TestBrainParsing:
    @pytest.fixture
    def brain(self):
        return AgentBrain()

    def _wrap(self, actions):
        return "Here you go:\n" + json.dumps({"actions": actions}) + "\nthanks"

    def test_valid_action_parsed_and_clamped(self, brain):
        raw = self._wrap([{
            "symbol": "AAPL", "action": "BUY", "confidence": 1.7,
            "position_size_pct": 0.9, "rationale": "strong",
        }])
        out = brain._parse_response(raw)
        assert len(out) == 1
        assert out[0]["confidence"] == 1.0      # clamped to [0,1]
        assert out[0]["position_size_pct"] == 0.20  # clamped to [0.01,0.20]

    def test_no_json_returns_none(self, brain):
        assert brain._parse_response("I refuse to answer.") is None

    def test_malformed_json_returns_none(self, brain):
        assert brain._parse_response("{ actions: [ broken ") is None

    def test_unknown_action_dropped(self, brain):
        raw = self._wrap([{
            "symbol": "AAPL", "action": "YOLO", "confidence": 0.8,
            "position_size_pct": 0.1, "rationale": "x",
        }])
        assert brain._parse_response(raw) == []

    def test_incomplete_action_dropped(self, brain):
        raw = self._wrap([{"symbol": "AAPL", "action": "BUY"}])  # missing keys
        assert brain._parse_response(raw) == []

    def test_all_holds_returns_empty_list_not_none(self, brain):
        raw = self._wrap([])
        assert brain._parse_response(raw) == []


# ── leverage_manager: modes + circuit breaker ────────────────────────────────

class TestLeverageManager:
    def test_fixed_mode_returns_multiplier(self):
        lm = LeverageManager(LeverageConfig(mode="fixed", fixed_multiplier=1.5,
                                            max_leverage=2.0))
        assert lm.get_leverage(current_equity=100_000) == pytest.approx(1.5)

    def test_none_mode_is_unlevered(self):
        lm = LeverageManager(LeverageConfig(mode="none"))
        assert lm.get_leverage(current_equity=100_000) == pytest.approx(1.0)

    def test_max_leverage_cap_enforced(self):
        lm = LeverageManager(LeverageConfig(mode="fixed", fixed_multiplier=10.0,
                                            max_leverage=2.0))
        assert lm.get_leverage(current_equity=100_000) <= 2.0

    def test_circuit_breaker_trips_on_drawdown(self):
        lm = LeverageManager(LeverageConfig(
            mode="fixed", fixed_multiplier=2.0, max_leverage=2.0,
            max_drawdown_trigger=0.10, min_leverage=0.5,
        ))
        lm.get_leverage(current_equity=100_000)        # set peak
        levered = lm.get_leverage(current_equity=85_000)  # 15% DD > 10% trigger
        assert lm._circuit_breaker_active is True
        assert levered < 2.0  # deleveraged

    def test_vol_target_falls_back_without_returns(self):
        lm = LeverageManager(LeverageConfig(mode="vol_target", min_leverage=0.5,
                                            max_leverage=3.0))
        # No daily_returns -> safe 1.0 fallback (not a silent huge number).
        assert lm.get_leverage(current_equity=100_000, daily_returns=None) == pytest.approx(1.0)


# ── weekly_review: Opus-response validation / clamping ───────────────────────

class TestWeeklyReviewParsing:
    def _current(self):
        return {
            "strategy_weights": {"mean_reversion": 0.25, "momentum": 0.30,
                                 "sentiment": 0.20, "pattern_recognition": 0.25},
            "buy_threshold": 0.25, "sell_threshold": -0.25,
            "stop_loss_pct": 0.08, "take_profit_pct": 0.20,
        }

    def test_valid_weights_applied_and_normalized(self):
        raw = json.dumps({
            "updated_strategy_weights": {"mean_reversion": 0.20, "momentum": 0.40,
                                         "sentiment": 0.15, "pattern_recognition": 0.25},
            "improvement_memo": "ok",
        })
        params, memo = weekly_review.parse_opus_response(raw, self._current())
        assert sum(params["strategy_weights"].values()) == pytest.approx(1.0, abs=1e-6)
        assert params["strategy_weights"]["momentum"] > params["strategy_weights"]["sentiment"]

    def test_out_of_bounds_stop_rejected(self):
        raw = json.dumps({"stop_loss_pct": 0.90, "improvement_memo": "x"})
        params, _ = weekly_review.parse_opus_response(raw, self._current())
        assert params["stop_loss_pct"] == 0.08  # unchanged

    def test_in_bounds_stop_applied(self):
        raw = json.dumps({"stop_loss_pct": 0.05, "improvement_memo": "x"})
        params, _ = weekly_review.parse_opus_response(raw, self._current())
        assert params["stop_loss_pct"] == 0.05

    def test_garbage_response_keeps_current_params(self):
        params, _ = weekly_review.parse_opus_response("no json here", self._current())
        assert params == self._current()
