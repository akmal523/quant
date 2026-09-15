"""
notifier.py — Daily Execution Push Notification (Phase 4, Module 4.2).

Intent: since this is solo use, the user should not manually check the terminal
every day. At the end of main.py, this module sends a push notification to the
user's phone summarizing exact trades, cash allocation, and risk warnings.

Supports two channels (configure via .env):
  - Telegram Bot API:  TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID
  - Discord Webhook:   DISCORD_WEBHOOK_URL

Phase 5 (v10.2):
  - Message text contains zero emoji characters.
  - If no channel is configured, log the exact missing variable name.

Invariants:
  - Sending is best-effort; failures are logged, never crash the pipeline.
  - Message is plain-text (Telegram) or embed (Discord).

Dependencies: requests, config, risk.daily_risk_free_rate.
"""
from __future__ import annotations

import os
import logging

import requests

from quant.portfolio.cash_rate import current_cash_apy

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


def missing_config() -> list[str]:
    """Return the names of missing notification env variables.

    Intent (Phase 5 / v10.2): make the missing-variable diagnosis explicit so
    the user knows exactly what to set in .env.
    Invariants: returns a list of variable names; empty if fully configured.
    """
    missing = []
    if not _telegram_token():
        missing.append("TELEGRAM_BOT_TOKEN")
    if not _telegram_chat_id():
        missing.append("TELEGRAM_CHAT_ID")
    if not _discord_webhook():
        missing.append("DISCORD_WEBHOOK_URL")
    return missing


def build_daily_summary(
    total_value: float,
    cash_allocation: float,
    instructions: list[dict],
    risk_warnings: list[str],
    regime: str = "unknown",
    data_health: list[str] | None = None,
) -> str:
    """Build the daily briefing text.

    Intent: single source of truth for the push message. Includes run date,
    market regime, cash yield, total portfolio value, exact execution
    instructions (route/symbol/ISIN/amount), bucket violations, and data-health
    warnings. Phase 5 (v10.2): zero emoji characters.
    Invariants: returns a non-empty string.
    """
    from datetime import date
    lines = [
        "Daily briefing - Trade Republic",
        "=" * 40,
        f"Date: {date.today().isoformat()}",
        f"Market trend: {regime}",
        f"Cash rate: {current_cash_apy()*100:.2f} percent per year",
        f"Portfolio value: {total_value:.2f} EUR",
        f"Cash share: {cash_allocation*100:.0f} percent",
        "",
        "What to do:",
    ]

    if instructions:
        for inst in instructions:
            lines.append(
                f"  - {inst.get('route', '')} {inst.get('symbol', '')} "
                f"ISIN {inst.get('isin', 'n/a')} amount {inst.get('min_trade_size_eur', 0):.0f} EUR"
            )
    else:
        lines.append("  - No trades required today.")

    if risk_warnings:
        lines.append("")
        lines.append("BUCKET VIOLATIONS:")
        for w in risk_warnings:
            lines.append(f"  - {w}")

    if data_health:
        lines.append("")
        lines.append("DATA HEALTH WARNINGS:")
        for h in data_health:
            lines.append(f"  - {h}")

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
        missing = ", ".join(missing_config())
        logger.info("[NOTIFIER] No notification channel configured - skipping push. "
                    "Missing env vars: %s", missing)
        return False

    message = build_daily_summary(total_value, cash_allocation, instructions, risk_warnings)
    delivered = False
    if send_telegram(message):
        delivered = True
    if send_discord(message):
        delivered = True
    return delivered