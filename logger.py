"""
Trade logging and performance tracking.
Logs every decision, trade, and performance metric to file and console.
"""

import csv
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import config


class TradeLogger:
    """Comprehensive trade and performance logger."""

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir or config.log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Set up Python logger
        self.logger = logging.getLogger("TradingAgent")
        self.logger.setLevel(getattr(logging, config.log_level))

        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        self.logger.addHandler(ch)

        # File handler
        log_file = os.path.join(self.log_dir, f"agent_{datetime.now():%Y%m%d}.log")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        self.logger.addHandler(fh)

        # Trade CSV
        self.trades_file = os.path.join(self.log_dir, "trades.csv")
        self._init_trades_csv()

        # Performance JSON
        self.perf_file = os.path.join(self.log_dir, "performance.json")
        self.performance_history: List[Dict] = []
        self._load_performance()

    # ── CSV setup ────────────────────────────────────────────────────────
    def _init_trades_csv(self):
        if not os.path.exists(self.trades_file):
            with open(self.trades_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "symbol", "action", "quantity", "price",
                    "strategy", "signal_strength", "portfolio_value",
                    "reason",
                ])

    def log_trade(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        strategy: str,
        signal_strength: float,
        portfolio_value: float,
        reason: str = "",
    ):
        """Log a trade execution."""
        ts = datetime.now().isoformat()
        self.logger.info(
            f"TRADE | {action} {quantity} {symbol} @ ${price:.2f} | "
            f"strategy={strategy} signal={signal_strength:.3f} | "
            f"portfolio=${portfolio_value:,.2f} | {reason}"
        )
        if config.save_trades_csv:
            with open(self.trades_file, "a", newline="") as f:
                csv.writer(f).writerow([
                    ts, symbol, action, quantity, price,
                    strategy, f"{signal_strength:.4f}",
                    f"{portfolio_value:.2f}", reason,
                ])

    # ── Performance tracking ─────────────────────────────────────────────
    def _load_performance(self):
        if os.path.exists(self.perf_file):
            with open(self.perf_file, "r") as f:
                self.performance_history = json.load(f)

    def _save_performance(self):
        with open(self.perf_file, "w") as f:
            json.dump(self.performance_history, f, indent=2)

    def log_performance_snapshot(self, metrics: Dict[str, Any]):
        """Record a periodic performance snapshot."""
        snapshot = {"timestamp": datetime.now().isoformat(), **metrics}
        self.performance_history.append(snapshot)
        self._save_performance()
        self.logger.info(f"PERF  | {json.dumps(metrics, default=str)}")

    def log_strategy_weights(self, weights: Dict[str, float]):
        """Log current strategy weight allocation."""
        w_str = " | ".join(f"{k}: {v:.3f}" for k, v in weights.items())
        self.logger.info(f"WEIGHTS | {w_str}")

    # ── General logging helpers ──────────────────────────────────────────
    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def debug(self, msg: str):
        self.logger.debug(msg)

    # ── Reporting ────────────────────────────────────────────────────────
    def get_trade_summary(self) -> Dict[str, Any]:
        """Parse trade CSV and return summary statistics."""
        trades = []
        if os.path.exists(self.trades_file):
            with open(self.trades_file, "r") as f:
                reader = csv.DictReader(f)
                trades = list(reader)

        if not trades:
            return {"total_trades": 0}

        buys = [t for t in trades if t["action"] == "BUY"]
        sells = [t for t in trades if t["action"] == "SELL"]
        strategies_used = {}
        for t in trades:
            s = t["strategy"]
            strategies_used[s] = strategies_used.get(s, 0) + 1

        return {
            "total_trades": len(trades),
            "buys": len(buys),
            "sells": len(sells),
            "strategies_used": strategies_used,
            "first_trade": trades[0]["timestamp"] if trades else None,
            "last_trade": trades[-1]["timestamp"] if trades else None,
        }
