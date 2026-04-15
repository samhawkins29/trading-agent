"""
Agent Brain — Claude Sonnet real-time decision layer.

Takes raw strategy signals, portfolio state, market regime, and optional news
headlines, packages them into a structured prompt, and calls the Anthropic
Messages API directly via `requests` (no pydantic/anthropic SDK required).

Falls back gracefully to None (triggering the legacy threshold system) if:
  - ANTHROPIC_API_KEY is not set or blank
  - The API call fails for any reason

All decisions are logged to logs/trade_journal.jsonl for the weekly Opus review.

Env vars:
    ANTHROPIC_API_KEY  — Required for this module to activate
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


def _get_api_key() -> str:
    """Return the API key, checking env var then config.py fallback."""
    key = os.getenv("ANTHROPIC_API_KEY") or ""
    if not key.strip():
        try:
            from config import ANTHROPIC_API_KEY as cfg_key
            key = cfg_key or ""
        except (ImportError, AttributeError):
            pass
    return key.strip()


class AgentBrain:
    """
    Claude Sonnet decision layer for the trading agent.

    Calls the Anthropic API directly over HTTPS using `requests` — no SDK,
    no pydantic dependency. Each call to decide() packages strategy signals
    into a prompt, queries claude-sonnet-4-6, parses the JSON response, and
    logs the decision to the trade journal. Returns None if disabled or on
    any error.
    """

    MODEL = "claude-sonnet-4-6"
    JOURNAL_PATH = os.path.join("logs", "trade_journal.jsonl")

    def __init__(self):
        self.api_key = _get_api_key()
        self.enabled = bool(self.api_key)
        if self.enabled:
            logger.info("AgentBrain: Claude Sonnet decision layer ENABLED")
        else:
            logger.info(
                "AgentBrain: DISABLED (set ANTHROPIC_API_KEY env var to activate)"
            )
        os.makedirs("logs", exist_ok=True)

    def decide(
        self,
        signals: Dict[str, Dict],
        positions: Dict,
        portfolio: Dict,
        regime: str,
        news: Optional[Dict[str, List[Dict]]] = None,
    ) -> Optional[List[Dict]]:
        """
        Ask Claude Sonnet for trade decisions.

        Args:
            signals:   {symbol: {combined_strength, combined_action, per_strategy}}
            positions: {symbol: {quantity, entry_price, unrealized_pnl_pct}}
            portfolio: {portfolio_value, cash, buying_power, equity}
            regime:    Market regime string (e.g. "trending_up")
            news:      Optional {symbol: [{headline, sentiment, published_at}]}

        Returns:
            List of action dicts or None if disabled/failed.
            Each action: {symbol, action, confidence, position_size_pct, rationale}
        """
        # Refresh key in case env var was set after init
        self.api_key = _get_api_key()
        self.enabled = bool(self.api_key)
        if not self.enabled:
            return None

        prompt = self._build_prompt(signals, positions, portfolio, regime, news)

        try:
            resp = requests.post(
                _API_URL,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": _ANTHROPIC_VERSION,
                    "content-type": "application/json",
                },
                json={
                    "model": self.MODEL,
                    "max_tokens": 2048,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=30,
            )
        except requests.exceptions.Timeout:
            logger.error("AgentBrain: API request timed out (30s) — skipping cycle")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"AgentBrain: Network error: {e}")
            return None

        if resp.status_code != 200:
            logger.error(
                f"AgentBrain: API returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
            return None

        try:
            data = resp.json()
            raw_text = data["content"][0]["text"]
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"AgentBrain: Unexpected response shape: {e} | {resp.text[:200]}")
            return None

        actions = self._parse_response(raw_text)
        if actions is not None:
            self._log_to_journal(actions, regime, portfolio, raw_text)
        return actions

    # ── Prompt Construction ─────────────────────────────────────────────

    def _build_prompt(
        self,
        signals: Dict,
        positions: Dict,
        portfolio: Dict,
        regime: str,
        news: Optional[Dict],
    ) -> str:
        lines = [
            "You are a quantitative trading AI managing a live paper trading portfolio.",
            "Review the strategy signals and market context, then output final trade decisions.",
            "",
            f"## Current Market Regime: {regime.replace('_', ' ').title()}",
            "",
            "## Portfolio State",
            f"- Total value: ${portfolio.get('portfolio_value', 0):,.2f}",
            f"- Available cash: ${portfolio.get('cash', 0):,.2f}",
            f"- Buying power: ${portfolio.get('buying_power', 0):,.2f}",
            "",
        ]

        if positions:
            lines.append("## Open Positions")
            for sym, pos in positions.items():
                pnl_pct = pos.get("unrealized_pnl_pct", 0) * 100
                lines.append(
                    f"- {sym}: {pos.get('quantity', 0)} shares @ "
                    f"${pos.get('entry_price', 0):.2f} ({pnl_pct:+.1f}% unrealized)"
                )
            lines.append("")

        lines += [
            "## Strategy Signals",
            "Combined strength ranges from -1.0 (strong sell) to +1.0 (strong buy).",
            "Per-strategy breakdown shows each algorithm's individual reading.",
            "",
        ]

        for symbol, sig in sorted(signals.items()):
            strength = sig.get("combined_strength", 0.0)
            action = sig.get("combined_action", "HOLD")
            bar_len = int(abs(strength) * 8)
            direction = "\u25b2" if strength > 0.1 else ("\u25bc" if strength < -0.1 else "\u2014")
            lines.append(
                f"**{symbol}**: {direction} {strength:+.3f} {'|' * bar_len} [{action}]"
            )
            per = sig.get("per_strategy", {})
            for strat, details in per.items():
                s = details.get("strength", 0)
                r = details.get("reason", "")[:80]
                lines.append(f"  * {strat}: {s:+.3f}  {r}")
            lines.append("")

        if news:
            lines.append("## Recent News Headlines")
            for symbol, headlines in news.items():
                if headlines:
                    lines.append(f"**{symbol}:**")
                    for h in headlines[:3]:
                        sentiment = h.get("sentiment", "neutral")
                        headline = h.get("headline", "")[:120]
                        lines.append(f"  [{sentiment.upper()}] {headline}")
            lines.append("")

        lines += [
            "## Decision Instructions",
            "1. Output BUY only when you have strong conviction (confidence >= 0.60).",
            "2. Output SELL only for symbols currently in open positions.",
            "3. HOLD is the default — omit symbols where action would be HOLD.",
            "4. Position size should be 5-15% of portfolio value per new position.",
            "5. Consider the regime: in trending markets favor momentum signals;",
            "   in mean-reverting markets favor mean-reversion signals.",
            "6. If news is negative, reduce confidence for BUY; increase for SELL.",
            "",
            "## Required Output Format",
            "Respond with ONLY valid JSON — no markdown, no explanation outside the JSON.",
            "",
            "{",
            '  "actions": [',
            "    {",
            '      "symbol": "AAPL",',
            '      "action": "BUY",',
            '      "confidence": 0.72,',
            '      "position_size_pct": 0.10,',
            '      "rationale": "One-sentence explanation of the decision"',
            "    }",
            "  ],",
            '  "market_assessment": "Brief overall market view in one sentence",',
            '  "risk_notes": "Any key risks or caveats"',
            "}",
        ]

        return "\n".join(lines)

    # ── Response Parsing ────────────────────────────────────────────────

    def _parse_response(self, raw: str) -> Optional[List[Dict]]:
        """Extract and validate the JSON action list from Claude's response."""
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end <= start:
            logger.warning(f"AgentBrain: No JSON object found in response:\n{raw[:300]}")
            return None

        try:
            data = json.loads(raw[start:end])
        except json.JSONDecodeError as e:
            logger.warning(f"AgentBrain: JSON parse error ({e}). Raw:\n{raw[:400]}")
            return None

        raw_actions = data.get("actions", [])
        if not isinstance(raw_actions, list):
            return None

        validated = []
        for a in raw_actions:
            if not isinstance(a, dict):
                continue
            if not all(k in a for k in ("symbol", "action", "confidence", "position_size_pct", "rationale")):
                logger.warning(f"AgentBrain: Incomplete action dict: {a}")
                continue
            if a["action"] not in ("BUY", "SELL", "HOLD"):
                logger.warning(f"AgentBrain: Unknown action '{a['action']}' for {a.get('symbol')}")
                continue
            a["confidence"] = max(0.0, min(1.0, float(a["confidence"])))
            a["position_size_pct"] = max(0.01, min(0.20, float(a["position_size_pct"])))
            validated.append(a)

        if not validated:
            logger.info("AgentBrain: No actionable decisions returned (all HOLDs or empty)")
            return []

        assessment = data.get("market_assessment", "")
        risk_notes = data.get("risk_notes", "")
        if assessment:
            logger.info(f"AgentBrain market assessment: {assessment}")
        if risk_notes:
            logger.info(f"AgentBrain risk notes: {risk_notes}")

        return validated

    # ── Trade Journal ───────────────────────────────────────────────────

    def _log_to_journal(
        self,
        actions: List[Dict],
        regime: str,
        portfolio: Dict,
        raw_response: str,
    ):
        """Append a decision record to the JSONL trade journal."""
        market_assessment = ""
        risk_notes = ""
        try:
            start = raw_response.find("{")
            end = raw_response.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(raw_response[start:end])
                market_assessment = data.get("market_assessment", "")
                risk_notes = data.get("risk_notes", "")
        except Exception:
            pass

        entry = {
            "timestamp": datetime.now().isoformat(),
            "regime": regime,
            "portfolio_value": portfolio.get("portfolio_value", 0),
            "cash": portfolio.get("cash", 0),
            "actions": actions,
            "market_assessment": market_assessment,
            "risk_notes": risk_notes,
        }
        try:
            with open(self.JOURNAL_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"AgentBrain: Failed to write trade journal: {e}")
