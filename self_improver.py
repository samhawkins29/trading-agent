"""
Self-Improvement Module — Reinforcement Learning Loop.

Inspired by:
  - Online learning / bandit algorithms (EXP3, UCB)
  - Experience replay from Deep RL (DQN-style)
  - Meta-learning: "learning to learn" by adjusting strategy allocations
  - Renaissance Technologies' continuous model recalibration

The agent tracks every trade outcome, stores experiences in a replay buffer,
and periodically adjusts strategy weights based on risk-adjusted performance.
"""

import json
import os
import random
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from config import config
from logger import TradeLogger


@dataclass
class Experience:
    """A single trade experience for the replay buffer."""
    timestamp: str
    symbol: str
    strategy: str
    action: str          # BUY or SELL
    signal_strength: float
    entry_price: float
    exit_price: float
    pnl: float
    holding_period: float  # hours
    market_regime: str


class SelfImprover:
    """
    Adjusts strategy weights based on observed performance.

    Algorithm:
      1. Store every trade result in an experience replay buffer
      2. Periodically evaluate each strategy's risk-adjusted return
      3. Shift weights toward better-performing strategies using
         a softmax update (like policy gradient)
      4. Respect min/max weight constraints
    """

    def __init__(self, logger: TradeLogger, save_path: str = "logs"):
        self.logger = logger
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        # Current weights (mutable copy from config)
        self.weights: Dict[str, float] = dict(config.strategy_weights)

        # Experience replay buffer
        self.replay_buffer: deque = deque(
            maxlen=config.experience_replay_size
        )

        # Per-strategy performance trackers
        self.strategy_pnls: Dict[str, List[float]] = {
            name: [] for name in self.weights
        }
        self.strategy_trades: Dict[str, int] = {
            name: 0 for name in self.weights
        }

        # Load previous state if exists
        self._load_state()

    # ── Record Outcome ───────────────────────────────────────────────
    def record_experience(
        self,
        symbol: str,
        strategy: str,
        action: str,
        signal_strength: float,
        entry_price: float,
        exit_price: float,
        holding_period_hours: float = 0.0,
        market_regime: str = "unknown",
    ):
        """Record a completed trade's outcome."""
        pnl = (exit_price - entry_price) / entry_price  # % return

        exp = Experience(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            strategy=strategy,
            action=action,
            signal_strength=signal_strength,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            holding_period=holding_period_hours,
            market_regime=market_regime,
        )
        self.replay_buffer.append(exp)

        if strategy in self.strategy_pnls:
            self.strategy_pnls[strategy].append(pnl)
            self.strategy_trades[strategy] = (
                self.strategy_trades.get(strategy, 0) + 1
            )

        self.logger.info(
            f"EXPERIENCE | {strategy} {action} {symbol}: "
            f"PnL={pnl:+.4f} | buffer_size={len(self.replay_buffer)}"
        )

    # ── Weight Update ────────────────────────────────────────────────
    def update_weights(self) -> Dict[str, float]:
        """
        Recalculate strategy weights based on recent performance.
        Uses a softmax-style policy gradient update.
        """
        if len(self.replay_buffer) < config.evaluation_window:
            self.logger.info(
                f"Not enough experiences to update weights "
                f"({len(self.replay_buffer)}/{config.evaluation_window})"
            )
            return self.weights

        # Evaluate each strategy's risk-adjusted performance
        scores = {}
        for name in self.weights:
            score = self._evaluate_strategy(name)
            scores[name] = score

        # Softmax update
        if all(s == 0 for s in scores.values()):
            self.logger.info("All strategy scores zero — keeping weights")
            return self.weights

        # Compute new weights via softmax on scores
        temperature = 1.0 / max(config.learning_rate, 0.001)
        score_arr = np.array([scores[n] for n in self.weights])
        score_arr = score_arr - score_arr.max()  # numerical stability
        exp_scores = np.exp(score_arr / temperature)
        softmax_weights = exp_scores / exp_scores.sum()

        # Blend old and new weights (smooth transition)
        lr = config.learning_rate
        new_weights = {}
        for i, name in enumerate(self.weights):
            blended = (1 - lr) * self.weights[name] + lr * softmax_weights[i]
            new_weights[name] = blended

        # Enforce min/max constraints
        new_weights = self._enforce_constraints(new_weights)

        # Log changes
        for name in self.weights:
            delta = new_weights[name] - self.weights[name]
            if abs(delta) > 0.001:
                self.logger.info(
                    f"WEIGHT UPDATE | {name}: "
                    f"{self.weights[name]:.3f} -> {new_weights[name]:.3f} "
                    f"(delta={delta:+.3f}, score={scores[name]:.4f})"
                )

        self.weights = new_weights
        self._save_state()
        self.logger.log_strategy_weights(self.weights)

        return self.weights

    def _evaluate_strategy(self, strategy_name: str) -> float:
        """
        Compute risk-adjusted score for a strategy.
        Uses a Sharpe-ratio-like metric on recent trades.
        """
        # Get recent experiences for this strategy
        recent = [
            e for e in list(self.replay_buffer)[-config.evaluation_window * 5:]
            if e.strategy == strategy_name
        ]

        if len(recent) < 3:
            return 0.0  # not enough data

        pnls = [e.pnl for e in recent]
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls, ddof=1) if len(pnls) > 1 else 1e-6

        # Sharpe-like ratio
        sharpe = mean_pnl / max(std_pnl, 1e-6)

        # Win rate bonus
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls)

        # Penalize high drawdown within strategy
        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        max_dd = np.max(peak - cumulative)

        # Composite score
        score = (
            0.50 * sharpe
            + 0.30 * (win_rate - 0.5) * 2  # center around 50%
            - 0.20 * max_dd * 10
        )

        return float(score)

    def _enforce_constraints(
        self, weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Enforce min/max weight constraints and normalize to sum=1."""
        # Clip to bounds
        for name in weights:
            weights[name] = max(
                config.min_strategy_weight,
                min(config.max_strategy_weight, weights[name]),
            )
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights

    # ── Experience Replay Sampling ───────────────────────────────────
    def sample_experiences(
        self, n: int = 32, strategy: Optional[str] = None
    ) -> List[Experience]:
        """
        Sample experiences from the replay buffer.
        Optionally filter by strategy.
        """
        pool = list(self.replay_buffer)
        if strategy:
            pool = [e for e in pool if e.strategy == strategy]
        n = min(n, len(pool))
        return random.sample(pool, n) if n > 0 else []

    # ── Persistence ──────────────────────────────────────────────────
    def _save_state(self):
        """Save weights and replay buffer to disk."""
        state = {
            "weights": self.weights,
            "replay_buffer": [asdict(e) for e in self.replay_buffer],
            "strategy_trades": self.strategy_trades,
            "updated_at": datetime.now().isoformat(),
        }
        path = os.path.join(self.save_path, "self_improver_state.json")
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load previous state from disk."""
        path = os.path.join(self.save_path, "self_improver_state.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                state = json.load(f)
            if "weights" in state:
                # Only load weights for known strategies
                for name in self.weights:
                    if name in state["weights"]:
                        self.weights[name] = state["weights"][name]
            if "replay_buffer" in state:
                for e_dict in state["replay_buffer"]:
                    self.replay_buffer.append(Experience(**e_dict))
            if "strategy_trades" in state:
                self.strategy_trades.update(state["strategy_trades"])
            self.logger.info(
                f"Loaded self-improver state: "
                f"{len(self.replay_buffer)} experiences"
            )
        except Exception as e:
            self.logger.warning(f"Failed to load self-improver state: {e}")

    # ── Reporting ────────────────────────────────────────────────────
    def get_report(self) -> Dict:
        """Return a summary of strategy performance and weights."""
        report = {
            "current_weights": dict(self.weights),
            "total_experiences": len(self.replay_buffer),
            "strategy_stats": {},
        }
        for name in self.weights:
            pnls = self.strategy_pnls.get(name, [])
            report["strategy_stats"][name] = {
                "trades": self.strategy_trades.get(name, 0),
                "mean_pnl": float(np.mean(pnls)) if pnls else 0.0,
                "win_rate": (
                    sum(1 for p in pnls if p > 0) / len(pnls)
                    if pnls
                    else 0.0
                ),
                "weight": self.weights[name],
            }
        return report
