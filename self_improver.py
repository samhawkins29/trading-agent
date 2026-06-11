"""
Self-Improvement Module -- Thompson Sampling + Regime-Aware Learning.

Weight updates use Thompson Sampling with Beta distributions (one per strategy)
instead of the previous softmax scoring. This provides:
  - Natural exploration/exploitation balance
  - Uncertainty-aware allocation (more exploration when data is sparse)
  - James-Stein shrinkage toward equal weights (reduces variance early on)

Per-regime weight profiles are still maintained and blended with the overall
Thompson-sampled weights.

Research basis:
  - Thompson (1933): Sampling for multi-armed bandit selection
  - James & Stein (1961): Shrinkage estimators beat OLS in high dimensions
  - Russo et al. (2018): A Tutorial on Thompson Sampling (arXiv:1707.02038)
  - Regime-switching factor investing (Nystrup et al., 2020)
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
    action: str
    signal_strength: float
    entry_price: float
    exit_price: float
    pnl: float
    holding_period: float
    market_regime: str


class SelfImprover:
    """
    Regime-aware self-improvement module.

    Tracks strategy performance BOTH overall and per-regime.
    When the regime detector identifies the current market state,
    the self-improver provides regime-specific weight recommendations.

    Algorithm:
      1. Store every trade result with its market regime label
      2. Evaluate each strategy's risk-adjusted return overall and per-regime
      3. Maintain separate weight profiles for each regime
      4. Blend regime-specific weights with overall weights
    """

    def __init__(self, logger: TradeLogger, save_path: str = "logs"):
        self.logger = logger
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        # Current weights (mutable copy from config)
        self.weights: Dict[str, float] = dict(config.strategy_weights)

        # Per-regime weight profiles
        self.regime_weights: Dict[str, Dict[str, float]] = {
            "trending_up": dict(config.strategy_weights),
            "trending_down": dict(config.strategy_weights),
            "mean_reverting": dict(config.strategy_weights),
            "crisis": dict(config.strategy_weights),
        }

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

        # Per-regime per-strategy tracking
        self.regime_strategy_pnls: Dict[str, Dict[str, List[float]]] = {}
        for regime in self.regime_weights:
            self.regime_strategy_pnls[regime] = {
                name: [] for name in self.weights
            }

        # Thompson Sampling Beta parameters: alpha=wins+1, beta=losses+1
        # Initialised to Beta(1,1) = Uniform (no prior preference).
        self.ts_alpha: Dict[str, float] = {name: 1.0 for name in self.weights}
        self.ts_beta: Dict[str, float] = {name: 1.0 for name in self.weights}

        self._load_state()

    # -- Record Outcome --

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
        is_short: bool = False,
    ):
        """Record a completed trade with regime label.

        pnl is the NET-OF-COST fractional return, direction-corrected:
          - long:  (exit - entry) / entry
          - short: (entry - exit) / entry   (a short profits when price falls)
        then minus an estimated round-trip cost. Win/loss labelling (which
        drives the Thompson-Sampling Beta posteriors) uses this net figure, so a
        short is no longer mislabelled as a loss when it actually made money, and
        a tiny gross gain that doesn't cover fees is correctly a loss.
        """
        # Infer direction from the closing action too, so callers that haven't
        # been updated to pass is_short still label shorts correctly.
        short = is_short or action.upper() in ("COVER", "SHORT")

        if entry_price > 0:
            gross = (exit_price - entry_price) / entry_price
            if short:
                gross = -gross
        else:
            gross = 0.0
        cost = getattr(config, "round_trip_cost_pct", 0.0)
        pnl = gross - cost   # net-of-cost return

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

        # Overall tracking
        if strategy in self.strategy_pnls:
            self.strategy_pnls[strategy].append(pnl)
            self.strategy_trades[strategy] = (
                self.strategy_trades.get(strategy, 0) + 1
            )

        # Thompson Sampling: update Beta(alpha, beta) for the strategy
        if strategy in self.ts_alpha:
            if pnl > 0:
                self.ts_alpha[strategy] += 1.0   # win
            else:
                self.ts_beta[strategy] += 1.0    # loss

        # Per-regime tracking
        if market_regime in self.regime_strategy_pnls:
            if strategy in self.regime_strategy_pnls[market_regime]:
                self.regime_strategy_pnls[market_regime][strategy].append(pnl)

        self.logger.info(
            f"EXPERIENCE | {strategy} {action} {symbol}: "
            f"PnL={pnl:+.4f} regime={market_regime} | "
            f"TS a={self.ts_alpha.get(strategy, 1):.0f} "
            f"b={self.ts_beta.get(strategy, 1):.0f} | "
            f"buffer_size={len(self.replay_buffer)}"
        )

    # -- Weight Update --

    def update_weights(
        self, regime_name: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Recalculate strategy weights using Thompson Sampling with
        James-Stein shrinkage toward equal weights.

        Posterior mean weight = alpha / (alpha + beta) per strategy,
        then normalised and shrunk toward 1/K as trade count grows.

        If regime_name is provided, also updates the regime-specific
        weight profile for that regime using the same approach.

        FROZEN until min_trades_for_learning closed round-trips exist: moving
        4 strategy weights (×4 regimes) on a handful of trades is curve-fitting
        to noise. Below the threshold this is a no-op that returns the current
        (configured) weights unchanged.
        """
        total_trades = sum(self.strategy_trades.values())
        min_trades = getattr(config, "min_trades_for_learning", 0)
        if total_trades < min_trades:
            self.logger.info(
                f"WEIGHT UPDATE SKIPPED | learning frozen: {total_trades}/"
                f"{min_trades} closed trades — weights held at current values"
            )
            return self.weights

        # 1. Thompson Sampling posterior means
        ts_means = {}
        for name in self.weights:
            a = self.ts_alpha.get(name, 1.0)
            b = self.ts_beta.get(name, 1.0)
            ts_means[name] = a / (a + b)   # posterior mean of Beta(a,b)

        # 2. James-Stein shrinkage toward equal weights (1/K)
        K = len(self.weights)
        equal_weight = 1.0 / K

        # Total trades as a proxy for data richness
        total_trades = sum(self.strategy_trades.values())
        # Shrinkage fades out as we accumulate >=200 trades
        shrinkage = max(0.0, 1.0 - total_trades / 200.0)

        shrunk = {}
        for name in self.weights:
            shrunk[name] = (1.0 - shrinkage) * ts_means[name] + shrinkage * equal_weight

        new_weights = self._enforce_constraints(shrunk)

        for name in self.weights:
            delta = new_weights[name] - self.weights[name]
            a = self.ts_alpha.get(name, 1.0)
            b = self.ts_beta.get(name, 1.0)
            if abs(delta) > 0.001:
                self.logger.info(
                    f"WEIGHT UPDATE | {name}: "
                    f"{self.weights[name]:.3f} -> {new_weights[name]:.3f} "
                    f"(TS a={a:.0f} b={b:.0f} mean={ts_means[name]:.3f} "
                    f"shrink={shrinkage:.2f})"
                )

        self.weights = new_weights

        # 3. Regime-specific weight update (softmax on historical scores --
        #    keep regime profiles updated independently as regime data is sparse)
        if regime_name and regime_name in self.regime_weights:
            regime_scores = {}
            for name in self.weights:
                regime_scores[name] = self._evaluate_strategy(
                    name, regime_filter=regime_name
                )

            if not all(s == 0 for s in regime_scores.values()):
                new_regime_weights = self._softmax_update(
                    self.regime_weights[regime_name], regime_scores
                )
                new_regime_weights = self._enforce_constraints(new_regime_weights)
                self.regime_weights[regime_name] = new_regime_weights

        self._save_state()
        self.logger.log_strategy_weights(self.weights)

        return self.weights

    def sample_weights_for_exploration(self) -> Dict[str, float]:
        """
        Sample weights from Beta posteriors for exploration during weekly review.
        This gives higher-variance weight suggestions than the posterior mean,
        allowing the weekly review to discover better weight combinations.
        """
        sampled = {}
        for name in self.weights:
            a = self.ts_alpha.get(name, 1.0)
            b = self.ts_beta.get(name, 1.0)
            sampled[name] = float(np.random.beta(a, b))

        # Normalise
        total = sum(sampled.values())
        if total > 0:
            sampled = {k: v / total for k, v in sampled.items()}
        return sampled

    def get_regime_weights(self, regime_name: str) -> Dict[str, float]:
        """Get the learned weight profile for a specific regime."""
        return self.regime_weights.get(regime_name, dict(self.weights))

    def _evaluate_strategy(
        self, strategy_name: str, regime_filter: Optional[str] = None
    ) -> float:
        """
        Compute risk-adjusted score for a strategy.

        Optionally filters to only experiences from a specific regime.
        Uses Sharpe-like metric + win rate + drawdown penalty.
        """
        recent = list(self.replay_buffer)[-config.evaluation_window * 5:]
        recent = [
            e for e in recent
            if e.strategy == strategy_name
            and (regime_filter is None or e.market_regime == regime_filter)
        ]

        if len(recent) < 3:
            return 0.0

        pnls = [e.pnl for e in recent]
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls, ddof=1) if len(pnls) > 1 else 1e-6

        # Sharpe-like ratio
        sharpe = mean_pnl / max(std_pnl, 1e-6)

        # Win rate bonus
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls)

        # Max drawdown penalty
        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        max_dd = np.max(peak - cumulative) if len(cumulative) > 0 else 0

        # Composite score
        score = (
            0.50 * sharpe
            + 0.30 * (win_rate - 0.5) * 2
            - 0.20 * max_dd * 10
        )

        return float(score)

    def _softmax_update(
        self, current_weights: Dict[str, float], scores: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply softmax-based weight update blended with current weights."""
        temperature = 1.0 / max(config.learning_rate, 0.001)
        score_arr = np.array([scores[n] for n in current_weights])
        score_arr = score_arr - score_arr.max()
        exp_scores = np.exp(score_arr / temperature)
        softmax_weights = exp_scores / exp_scores.sum()

        lr = config.learning_rate
        new_weights = {}
        for i, name in enumerate(current_weights):
            blended = (1 - lr) * current_weights[name] + lr * softmax_weights[i]
            new_weights[name] = blended

        return new_weights

    def _enforce_constraints(
        self, weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Enforce min/max weight constraints and normalize."""
        for name in weights:
            weights[name] = max(
                config.min_strategy_weight,
                min(config.max_strategy_weight, weights[name]),
            )
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        return weights

    # -- Experience Replay --

    def sample_experiences(
        self, n: int = 32, strategy: Optional[str] = None,
        regime: Optional[str] = None
    ) -> List[Experience]:
        """Sample experiences, optionally filtered by strategy and/or regime."""
        pool = list(self.replay_buffer)
        if strategy:
            pool = [e for e in pool if e.strategy == strategy]
        if regime:
            pool = [e for e in pool if e.market_regime == regime]
        n = min(n, len(pool))
        return random.sample(pool, n) if n > 0 else []

    # -- Persistence --

    def _save_state(self):
        state = {
            "weights": self.weights,
            "regime_weights": self.regime_weights,
            "replay_buffer": [asdict(e) for e in self.replay_buffer],
            "strategy_trades": self.strategy_trades,
            "ts_alpha": self.ts_alpha,
            "ts_beta": self.ts_beta,
            "updated_at": datetime.now().isoformat(),
        }
        path = os.path.join(self.save_path, "self_improver_state.json")
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        path = os.path.join(self.save_path, "self_improver_state.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                state = json.load(f)
            if "weights" in state:
                for name in self.weights:
                    if name in state["weights"]:
                        self.weights[name] = state["weights"][name]
            if "regime_weights" in state:
                for regime in self.regime_weights:
                    if regime in state["regime_weights"]:
                        for name in self.regime_weights[regime]:
                            if name in state["regime_weights"][regime]:
                                self.regime_weights[regime][name] = state["regime_weights"][regime][name]
            if "replay_buffer" in state:
                for e_dict in state["replay_buffer"]:
                    self.replay_buffer.append(Experience(**e_dict))
            if "strategy_trades" in state:
                self.strategy_trades.update(state["strategy_trades"])
            if "ts_alpha" in state:
                for name in self.ts_alpha:
                    if name in state["ts_alpha"]:
                        self.ts_alpha[name] = float(state["ts_alpha"][name])
            if "ts_beta" in state:
                for name in self.ts_beta:
                    if name in state["ts_beta"]:
                        self.ts_beta[name] = float(state["ts_beta"][name])
            self.logger.info(
                f"Loaded self-improver state: "
                f"{len(self.replay_buffer)} experiences | "
                f"TS params: {', '.join(f'{n}(a={self.ts_alpha[n]:.0f},b={self.ts_beta[n]:.0f})' for n in self.ts_alpha)}"
            )
        except Exception as e:
            self.logger.warning(f"Failed to load self-improver state: {e}")

    # -- Reporting --

    def get_report(self) -> Dict:
        report = {
            "current_weights": dict(self.weights),
            "regime_weights": {k: dict(v) for k, v in self.regime_weights.items()},
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
                    if pnls else 0.0
                ),
                "weight": self.weights[name],
            }

        # Per-regime stats
        report["regime_stats"] = {}
        for regime, strat_pnls in self.regime_strategy_pnls.items():
            regime_report = {}
            for name, pnls in strat_pnls.items():
                if pnls:
                    regime_report[name] = {
                        "trades": len(pnls),
                        "mean_pnl": float(np.mean(pnls)),
                        "win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
                    }
            if regime_report:
                report["regime_stats"][regime] = regime_report

        return report
