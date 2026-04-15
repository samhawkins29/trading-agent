"""
Weekly Review — Claude Opus analysis and strategy improvement.

Reads the trade journal (logs/trade_journal.jsonl) and executed trades
(logs/paper_trades.csv), computes performance statistics, and sends the
full context to Claude Opus for deep analysis.

Opus outputs:
  - Updated strategy weights
  - Revised signal thresholds, stop-loss, take-profit
  - A written improvement memo

Results are saved to:
  - learned_params.json          (loaded by live_trader.py on next start)
  - weekly_reviews/YYYY-MM-DD_review.md  (human-readable memo)

Usage:
    python weekly_review.py                  # Review last 7 days
    python weekly_review.py --days 14        # Review last 14 days
    python weekly_review.py --dry-run        # Print prompt, don't call Opus

Env vars:
    ANTHROPIC_API_KEY  — Required
"""

import argparse
import csv
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OPUS_MODEL = "claude-opus-4-6"
JOURNAL_PATH = os.path.join("logs", "trade_journal.jsonl")
TRADES_CSV = os.path.join("logs", "paper_trades.csv")
LEARNED_PARAMS_PATH = "learned_params.json"
REVIEWS_DIR = "weekly_reviews"


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_journal_entries(days_back: int = 7) -> List[Dict]:
    """Load trade journal entries from the last N days."""
    if not os.path.exists(JOURNAL_PATH):
        logger.warning(f"Trade journal not found: {JOURNAL_PATH}")
        return []
    cutoff = datetime.now() - timedelta(days=days_back)
    entries = []
    with open(JOURNAL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ts = datetime.fromisoformat(entry.get("timestamp", "2000-01-01"))
                if ts >= cutoff:
                    entries.append(entry)
            except (json.JSONDecodeError, ValueError):
                continue
    logger.info(f"Loaded {len(entries)} journal entries from last {days_back} days")
    return entries


def load_executed_trades(days_back: int = 7) -> List[Dict]:
    """Load executed trades from paper_trades.csv."""
    if not os.path.exists(TRADES_CSV):
        logger.warning(f"Trades CSV not found: {TRADES_CSV}")
        return []
    cutoff = datetime.now() - timedelta(days=days_back)
    trades = []
    with open(TRADES_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = datetime.fromisoformat(row["timestamp"])
                if ts >= cutoff:
                    trades.append(row)
            except (ValueError, KeyError):
                continue
    logger.info(f"Loaded {len(trades)} trades from last {days_back} days")
    return trades


# ── Trade Pairing and Stats ──────────────────────────────────────────────────

def pair_trades(trades: List[Dict]) -> List[Dict]:
    """
    Match BUY/SELL trades into round-trips using FIFO per symbol.
    Returns list of completed trades with P&L.
    """
    buys: Dict[str, List[Dict]] = defaultdict(list)
    completed = []

    for trade in sorted(trades, key=lambda t: t.get("timestamp", "")):
        symbol = trade.get("symbol", "")
        action = trade.get("action", "")
        try:
            qty = int(float(trade.get("quantity", 0)))
            price = float(trade.get("filled_price") or trade.get("price", 0))
        except (ValueError, TypeError):
            continue

        if action == "BUY":
            buys[symbol].append({
                "timestamp": trade.get("timestamp"),
                "quantity": qty,
                "price": price,
                "strategy": trade.get("strategy", "unknown"),
                "regime": trade.get("regime", "unknown"),
                "signal_strength": float(trade.get("signal_strength", 0) or 0),
            })
        elif action == "SELL" and buys[symbol]:
            buy = buys[symbol].pop(0)  # FIFO
            pnl = (price - buy["price"]) * min(qty, buy["quantity"])
            pnl_pct = (price - buy["price"]) / buy["price"] if buy["price"] > 0 else 0
            completed.append({
                "symbol": symbol,
                "strategy": buy["strategy"],
                "regime": buy["regime"],
                "entry_price": buy["price"],
                "exit_price": price,
                "quantity": min(qty, buy["quantity"]),
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "win": pnl > 0,
                "entry_ts": buy["timestamp"],
                "exit_ts": trade.get("timestamp"),
                "signal_strength": buy["signal_strength"],
            })

    return completed


def compute_stats(completed: List[Dict]) -> Dict:
    """Compute aggregate and per-strategy/per-symbol stats."""
    if not completed:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "by_strategy": {},
            "by_symbol": {},
            "by_regime": {},
        }

    total = len(completed)
    wins = [t for t in completed if t["win"]]
    losses = [t for t in completed if not t["win"]]
    total_pnl = sum(t["pnl"] for t in completed)

    # Per-strategy breakdown
    by_strategy: Dict[str, Dict] = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0})
    for t in completed:
        s = t["strategy"]
        by_strategy[s]["trades"] += 1
        by_strategy[s]["pnl"] += t["pnl"]
        by_strategy[s]["wins"] += int(t["win"])
    for s, d in by_strategy.items():
        d["win_rate"] = d["wins"] / d["trades"] if d["trades"] > 0 else 0.0
        d["avg_pnl"] = d["pnl"] / d["trades"] if d["trades"] > 0 else 0.0

    # Per-symbol breakdown
    by_symbol: Dict[str, Dict] = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0})
    for t in completed:
        sym = t["symbol"]
        by_symbol[sym]["trades"] += 1
        by_symbol[sym]["pnl"] += t["pnl"]
        by_symbol[sym]["wins"] += int(t["win"])
    for sym, d in by_symbol.items():
        d["win_rate"] = d["wins"] / d["trades"] if d["trades"] > 0 else 0.0
        d["avg_pnl"] = d["pnl"] / d["trades"] if d["trades"] > 0 else 0.0

    # Per-regime breakdown
    by_regime: Dict[str, Dict] = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0})
    for t in completed:
        r = t["regime"]
        by_regime[r]["trades"] += 1
        by_regime[r]["pnl"] += t["pnl"]
        by_regime[r]["wins"] += int(t["win"])
    for r, d in by_regime.items():
        d["win_rate"] = d["wins"] / d["trades"] if d["trades"] > 0 else 0.0

    return {
        "total_trades": total,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / total if total > 0 else 0.0,
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / total,
        "avg_win": sum(t["pnl"] for t in wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(t["pnl"] for t in losses) / len(losses) if losses else 0.0,
        "by_strategy": dict(by_strategy),
        "by_symbol": dict(by_symbol),
        "by_regime": dict(by_regime),
    }


def load_current_params() -> Dict:
    """Load current config defaults and any existing learned params."""
    params = {
        "strategy_weights": {
            "mean_reversion": 0.25,
            "momentum": 0.30,
            "sentiment": 0.20,
            "pattern_recognition": 0.25,
        },
        "buy_threshold": 0.25,
        "sell_threshold": -0.25,
        "stop_loss_pct": 0.08,
        "take_profit_pct": 0.20,
    }
    # Overlay with config.py
    try:
        from config import config as cfg, TradingConfig
        params["strategy_weights"] = dict(cfg.strategy_weights)
        params["stop_loss_pct"] = cfg.stop_loss_pct
        params["take_profit_pct"] = cfg.take_profit_pct
    except Exception:
        pass
    # Overlay with previously learned params
    if os.path.exists(LEARNED_PARAMS_PATH):
        try:
            with open(LEARNED_PARAMS_PATH) as f:
                learned = json.load(f)
            params.update(learned)
        except Exception:
            pass
    return params


# ── Opus Prompt ──────────────────────────────────────────────────────────────

def build_opus_prompt(
    stats: Dict,
    current_params: Dict,
    journal_entries: List[Dict],
    completed_trades: List[Dict],
    days_back: int,
) -> str:
    lines = [
        f"You are a senior quantitative portfolio manager conducting a {days_back}-day",
        "performance review of an AI trading system.",
        "",
        "## Current Strategy Parameters",
        f"Strategy weights: {json.dumps(current_params.get('strategy_weights', {}), indent=2)}",
        f"Buy threshold: {current_params.get('buy_threshold', 0.25)}",
        f"Sell threshold: {current_params.get('sell_threshold', -0.25)}",
        f"Stop loss: {current_params.get('stop_loss_pct', 0.08):.1%}",
        f"Take profit: {current_params.get('take_profit_pct', 0.20):.1%}",
        "",
        f"## Performance Summary ({days_back} days)",
        f"Total completed trades: {stats['total_trades']}",
        f"Win rate: {stats['win_rate']:.1%}  ({stats['wins']} wins / {stats['losses']} losses)",
        f"Total P&L: ${stats['total_pnl']:+,.2f}",
        f"Average P&L per trade: ${stats['avg_pnl']:+.2f}",
        f"Average win: ${stats['avg_win']:+.2f}  |  Average loss: ${stats['avg_loss']:+.2f}",
        "",
    ]

    # Per-strategy stats
    if stats.get("by_strategy"):
        lines.append("## Performance by Strategy")
        for strat, d in sorted(stats["by_strategy"].items(), key=lambda x: -x[1]["pnl"]):
            lines.append(
                f"- {strat}: {d['trades']} trades, "
                f"win rate {d['win_rate']:.1%}, "
                f"total P&L ${d['pnl']:+,.2f}, "
                f"avg ${d['avg_pnl']:+.2f}"
            )
        lines.append("")

    # Per-symbol stats
    if stats.get("by_symbol"):
        lines.append("## Performance by Symbol")
        for sym, d in sorted(stats["by_symbol"].items(), key=lambda x: -x[1]["pnl"]):
            lines.append(
                f"- {sym}: {d['trades']} trades, "
                f"win rate {d['win_rate']:.1%}, "
                f"total P&L ${d['pnl']:+,.2f}"
            )
        lines.append("")

    # Per-regime stats
    if stats.get("by_regime"):
        lines.append("## Performance by Market Regime")
        for regime, d in stats["by_regime"].items():
            lines.append(
                f"- {regime}: {d['trades']} trades, "
                f"win rate {d['win_rate']:.1%}, "
                f"total P&L ${d['pnl']:+,.2f}"
            )
        lines.append("")

    # Recent journal context (last 20 brain decisions)
    if journal_entries:
        lines.append(f"## Recent AI Brain Decisions (last {min(20, len(journal_entries))})")
        for entry in journal_entries[-20:]:
            ts = entry.get("timestamp", "")[:16]
            regime = entry.get("regime", "?")
            pv = entry.get("portfolio_value", 0)
            assessment = entry.get("market_assessment", "")
            actions = entry.get("actions", [])
            action_strs = [
                f"{a['symbol']}:{a['action']}({a['confidence']:.0%})"
                for a in actions if a.get("action") in ("BUY", "SELL")
            ]
            lines.append(
                f"[{ts}] {regime} | ${pv:,.0f} | "
                + (", ".join(action_strs) if action_strs else "HOLD")
                + (f" | {assessment}" if assessment else "")
            )
        lines.append("")

    # Worst trades for pattern analysis
    if completed_trades:
        worst = sorted(completed_trades, key=lambda t: t["pnl"])[:5]
        best = sorted(completed_trades, key=lambda t: -t["pnl"])[:5]
        lines.append("## Best Trades")
        for t in best:
            lines.append(
                f"  {t['symbol']} via {t['strategy']}: "
                f"${t['pnl']:+.2f} ({t['pnl_pct']:+.1%}) in {t['regime']} regime"
            )
        lines.append("## Worst Trades")
        for t in worst:
            lines.append(
                f"  {t['symbol']} via {t['strategy']}: "
                f"${t['pnl']:+.2f} ({t['pnl_pct']:+.1%}) in {t['regime']} regime"
            )
        lines.append("")

    lines += [
        "## Your Task",
        "Analyze the above performance data and provide concrete improvement recommendations.",
        "Consider:",
        "  1. Which strategies are over/underperforming? Should any weights change?",
        "  2. Are the signal thresholds (0.25 buy / -0.25 sell) appropriate?",
        "  3. Should stop-loss or take-profit levels change based on win/loss patterns?",
        "  4. Any regime-specific observations? (e.g., momentum worse in mean-reverting markets?)",
        "  5. Any symbols consistently losing? Should any be traded with lower size?",
        "",
        "Provide your output as ONLY valid JSON in this exact format:",
        "",
        "{",
        '  "updated_strategy_weights": {',
        '    "mean_reversion": 0.25,',
        '    "momentum": 0.30,',
        '    "sentiment": 0.20,',
        '    "pattern_recognition": 0.25',
        "  },",
        '  "buy_threshold": 0.25,',
        '  "sell_threshold": -0.25,',
        '  "stop_loss_pct": 0.08,',
        '  "take_profit_pct": 0.20,',
        '  "improvement_memo": "Multi-paragraph analysis and recommendations (plain text, can be long)"',
        "}",
        "",
        "Weights must sum to 1.0. All values must be realistic (weights 0.05-0.60,"
        " thresholds 0.10-0.50, stop_loss 0.02-0.20, take_profit 0.05-0.40).",
    ]

    return "\n".join(lines)


# ── Opus Call ────────────────────────────────────────────────────────────────

def call_opus(prompt: str) -> Optional[str]:
    """Call Claude Opus via direct HTTPS — no anthropic SDK or pydantic needed."""
    import requests as _requests

    api_key = os.getenv("ANTHROPIC_API_KEY") or ""
    if not api_key.strip():
        try:
            from config import ANTHROPIC_API_KEY
            api_key = ANTHROPIC_API_KEY or ""
        except (ImportError, AttributeError):
            pass

    if not api_key.strip():
        logger.error("ANTHROPIC_API_KEY not set — cannot run weekly review")
        return None

    logger.info(f"Calling Claude Opus ({OPUS_MODEL})...")
    try:
        resp = _requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": OPUS_MODEL,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=120,
        )
        if resp.status_code != 200:
            logger.error(f"Opus API returned HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        return resp.json()["content"][0]["text"]
    except Exception as e:
        logger.error(f"Opus API call failed: {e}")
        return None


# ── Response Parsing ──────────────────────────────────────────────────────────

def parse_opus_response(raw: str, current_params: Dict) -> Tuple[Dict, str]:
    """
    Parse Opus JSON response.
    Returns (updated_params_dict, improvement_memo_str).
    Falls back to current_params if parsing fails.
    """
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end <= start:
        logger.warning("Could not find JSON in Opus response — keeping current params")
        return current_params, raw

    try:
        data = json.loads(raw[start:end])
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error in Opus response: {e}")
        return current_params, raw

    updated_params = dict(current_params)

    # Validate and apply strategy weights
    new_weights = data.get("updated_strategy_weights", {})
    if isinstance(new_weights, dict) and len(new_weights) == 4:
        total = sum(new_weights.values())
        if 0.95 <= total <= 1.05 and all(0.04 <= v <= 0.62 for v in new_weights.values()):
            # Renormalize to ensure exact sum of 1
            updated_params["strategy_weights"] = {
                k: round(v / total, 4) for k, v in new_weights.items()
            }
            logger.info(f"Updated strategy weights: {updated_params['strategy_weights']}")
        else:
            logger.warning(f"Opus weights failed validation (sum={total:.3f}) — keeping current")

    # Apply thresholds with bounds checking
    for field, lo, hi in [
        ("buy_threshold", 0.10, 0.50),
        ("stop_loss_pct", 0.02, 0.20),
        ("take_profit_pct", 0.05, 0.40),
    ]:
        if field in data:
            val = float(data[field])
            if lo <= val <= hi:
                updated_params[field] = round(val, 4)

    if "sell_threshold" in data:
        val = float(data["sell_threshold"])
        if -0.50 <= val <= -0.10:
            updated_params["sell_threshold"] = round(val, 4)

    memo = data.get("improvement_memo", "No improvement memo provided.")
    return updated_params, memo


# ── Persistence ──────────────────────────────────────────────────────────────

def save_learned_params(params: Dict, memo: str):
    """Save updated params to learned_params.json."""
    params["updated_at"] = datetime.now().strftime("%Y-%m-%d")
    params["memo"] = memo
    with open(LEARNED_PARAMS_PATH, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved learned params to {LEARNED_PARAMS_PATH}")


def save_review_memo(memo: str, stats: Dict, params: Dict):
    """Save human-readable review memo to weekly_reviews/."""
    os.makedirs(REVIEWS_DIR, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(REVIEWS_DIR, f"{date_str}_review.md")

    lines = [
        f"# Weekly Trading Review — {date_str}",
        "",
        "## Performance Summary",
        f"- Trades: {stats['total_trades']} | Win rate: {stats['win_rate']:.1%}",
        f"- Total P&L: ${stats['total_pnl']:+,.2f}",
        f"- Avg win: ${stats['avg_win']:+.2f} | Avg loss: ${stats['avg_loss']:+.2f}",
        "",
        "## Updated Parameters",
        "```json",
        json.dumps({
            k: v for k, v in params.items()
            if k not in ("updated_at", "memo")
        }, indent=2),
        "```",
        "",
        "## Opus Improvement Analysis",
        "",
        memo,
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Review memo saved to {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def run_weekly_review(days_back: int = 7, dry_run: bool = False):
    """Run the full weekly review pipeline."""
    logger.info(f"Starting weekly review (last {days_back} days)...")

    # Load data
    journal_entries = load_journal_entries(days_back)
    executed_trades = load_executed_trades(days_back)

    # Compute stats
    completed = pair_trades(executed_trades)
    stats = compute_stats(completed)

    logger.info(
        f"Stats: {stats['total_trades']} trades, "
        f"win rate {stats['win_rate']:.1%}, "
        f"total P&L ${stats['total_pnl']:+,.2f}"
    )

    if stats["total_trades"] < 5:
        logger.warning(
            f"Only {stats['total_trades']} completed trades — "
            "review may not be meaningful yet. Continuing anyway."
        )

    # Load current params
    current_params = load_current_params()

    # Build prompt
    prompt = build_opus_prompt(
        stats, current_params, journal_entries, completed, days_back
    )

    if dry_run:
        print("\n" + "=" * 60 + " OPUS PROMPT " + "=" * 60)
        print(prompt)
        print("=" * 133)
        logger.info("Dry run — not calling Opus API")
        return

    # Call Opus
    raw_response = call_opus(prompt)
    if not raw_response:
        logger.error("Weekly review failed — no response from Opus")
        return

    logger.info("Opus response received")

    # Parse response
    updated_params, memo = parse_opus_response(raw_response, current_params)

    # Save outputs
    save_learned_params(updated_params, memo)
    memo_path = save_review_memo(memo, stats, updated_params)

    logger.info("Weekly review complete.")
    logger.info(f"  Params saved to: {LEARNED_PARAMS_PATH}")
    logger.info(f"  Memo saved to:   {memo_path}")
    logger.info("\nUpdated strategy weights:")
    for name, weight in updated_params.get("strategy_weights", {}).items():
        current = current_params.get("strategy_weights", {}).get(name, 0)
        delta = weight - current
        logger.info(f"  {name}: {weight:.3f} ({delta:+.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly Opus trading review")
    parser.add_argument(
        "--days", type=int, default=7, help="Days of history to review (default: 7)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the prompt without calling Opus"
    )
    args = parser.parse_args()
    run_weekly_review(days_back=args.days, dry_run=args.dry_run)
