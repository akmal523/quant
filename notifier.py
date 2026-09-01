"""
notifier.py — Daily Execution Push Notification (Phase 4, Module 4.2).

Intent: since this is solo use, the user should not manually check the terminal
every day. At the end of main.py, this module sends a push notification to the
user's phone summarizing exact trades, cash allocation, and risk warnings.

Supports two channels (configure via .env):
  - Telegram Bot API:  TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
  - Discord Webhook:   DISCORD_WEBHOOK_URL

Invariants:
  - Sending is best-effort; failures are logged, never crash the pipeline.
  - Message is plain-text (Telegram) or embed (Discord).

Dependencies: requests, config, risk.daily_risk_free_rate.
"""
from __future__ import annotations

import os
import logging

import requests

from config import BROKER_CASH_APY
from risk import daily_risk_free_rate

logger = logging.getLogger(__name__)


def _telegram_token() -> str:
    return os.getenv("TELEGRAM_BOT_TOKEN", "")


def _telegram_chat_id() -> str:
    return os.getenv("TELEGRAM_CHAT_ID", "")


def _discord_webhook() -> str:
    return os.getenv("DISCORD_WEBHOOK_URL", "")


def is_configured() -> bool:
    """True if at least one notification channel is configured."""
    return bool(_telegram_token() and _telegram_chat_id()) or bool(_discord_webhook())


def build_daily_summary(
    total_value: float,
    cash_allocation: float,
    instructions: list[dict],
    risk_warnings: list[str],
) -> str:
    """Build the daily briefing text.

    Intent: single source of truth for the push message. Includes cash yield,
    total portfolio value, exact execution instructions, and risk warnings.
    Invariants: returns a non-empty string.
    """
    lines = [
        "📊 DAILY BRIEFING — Trade Republic",
        "=" * 40,
        f"Cash yield (APY): {BROKER_CASH_APY*100:.2f}%  (daily {daily_risk_free_rate()*100:.4f}%)",
        f"Total portfolio value: €{total_value:,.2f}",
        f"Cash allocation: {cash_allocation*100:.1f}%",
        "",
        "📋 EXECUTION INSTRUCTIONS:",
    ]

    if instructions:
        for inst in instructions:
            lines.append(f"  • {inst.get('instruction', '')}")
    else:
        lines.append("  • No trades required today.")

    if risk_warnings:
        lines.append("")
        lines.append("⚠️ RISK WARNINGS:")
        for w in risk_warnings:
            lines.append(f"  • {w}")

    return "\n".join(lines)


def send_telegram(message: str) -> bool:
    """Send a message via Telegram Bot API. Returns True on success."""
    token = _telegram_token()
    chat_id = _telegram_chat_id()
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=15)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.warning("[NOTIFIER] Telegram send failed: %s", e)
        return False


def send_discord(message: str) -> bool:
    """Send a message via Discord webhook. Returns True on success."""
    webhook = _discord_webhook()
    if not webhook:
        return False
    try:
        resp = requests.post(webhook, json={"content": message}, timeout=15)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.warning("[NOTIFIER] Discord send failed: %s", e)
        return False


def notify_daily(
    total_value: float,
    cash_allocation: float,
    instructions: list[dict],
    risk_warnings: list[str],
) -> bool:
    """Send the daily briefing to all configured channels.

    Intent: end-of-pipeline hook. Best-effort; never raises.
    Invariants: returns True if at least one channel delivered.
    """
    if not is_configured():
        logger.info("[NOTIFIER] No notification channel configured — skipping push.")
        return False

    message = build_daily_summary(total_value, cash_allocation, instructions, risk_warnings)
    delivered = False
    if send_telegram(message):
        delivered = True
    if send_discord(message):
        delivered = True
    return delivered