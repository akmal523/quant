"""
notify.py — Telegram/email dispatch and the in-app banner flag (v10.7.0, Section 4).

Intent: the user is often away from the laptop in the evening; the phone is
where Trade Republic lives. A short push closes the loop. Telegram is sent with
the standard library (urllib); email reuses the existing SMTP configuration.
A failed send never crashes the daily job: the alert stays unnotified and
retries on the next run. The token is never logged.

Invariants:
  - send_text/send_alert never raise; return True on success.
  - Message content carries no portfolio totals, no tier names, no jargon.
  - The bot token never appears in any log or exception message.
"""
from __future__ import annotations

import os
import smtplib
import urllib.parse
import urllib.request
from email.message import EmailMessage

from quant import paths

NOTIFY_FILE = os.path.join(str(paths.DATA_DIR), "notify.toml")


def load_config(path: str | None = None) -> dict:
    """Load data/notify.toml. Missing/invalid file -> {'channel': 'none'}."""
    path = path or NOTIFY_FILE
    try:
        import tomllib

        with open(path, "rb") as f:
            data = tomllib.load(f)
        if not isinstance(data, dict):
            return {"channel": "none"}
        return data
    except Exception:  # noqa: BLE001
        return {"channel": "none"}


def save_config(config: dict, path: str | None = None) -> None:
    """Write data/notify.toml with local-only permissions. Never raises."""
    path = path or NOTIFY_FILE
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lines = []
        for key, value in config.items():
            if isinstance(value, bool):
                lines.append(f"{key} = {'true' if value else 'false'}")
            elif isinstance(value, int | float):
                lines.append(f"{key} = {value}")
            else:
                lines.append(f'{key} = "{value}"')
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        os.chmod(path, 0o600)
    except Exception:  # noqa: BLE001
        pass


def status_line(config: dict | None = None) -> str:
    """One plain line for the doctor and Settings."""
    cfg = config if config is not None else load_config()
    channel = str(cfg.get("channel", "none")).lower()
    if channel == "telegram":
        return "Telegram on, last test ok" if cfg.get("tested") else "Telegram on, not tested yet"
    if channel == "email":
        return "email on, last test ok" if cfg.get("tested") else "email on, not tested yet"
    return "off; run quant notify-setup"


def format_alert(alert: dict) -> str:
    """The exact alert phrasing template (Section 2)."""
    from quant.ui import copy as ui_copy

    name = alert.get("name") or alert.get("symbol") or "Holding"
    action = str(alert.get("action", "review"))
    amount = alert.get("amount_eur")
    message = alert.get("message", "")
    fee = alert.get("fee_eur")
    lines = [ui_copy.ALERT_ACTION_HEADER]
    if action in ("sell", "buy") and amount:
        lines.append("  " + ui_copy.ALERT_ACTION_LINE.format(
            name=name, action=action, amount=f"{float(amount):.0f}"))
        lines.append("  " + ui_copy.ALERT_ACTION_FOOTER)
        lines.append("  " + ui_copy.ALERT_REASON.format(reason=message))
        if fee is not None:
            lines.append("  " + ui_copy.ALERT_FEE.format(
                fee=f"{float(fee):.0f}", side=action))
    else:
        lines.append("  " + message)
        lines.append("  " + ui_copy.ALERT_DETAILS_IN_APP)
    return "\n".join(lines)


def _send_telegram(cfg: dict, text: str) -> bool:
    token = cfg.get("telegram_bot_token")
    chat_id = cfg.get("telegram_chat_id")
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": text}).encode()
    try:
        request = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(request, timeout=10) as response:
            return 200 <= int(getattr(response, "status", 0)) < 300
    except Exception:  # noqa: BLE001
        # Never include the token in the message.
        return False


def _send_email(cfg: dict, text: str) -> bool:
    from quant.config import REPORT_TO, SMTP_PASSWORD, SMTP_USER

    sender = cfg.get("email_from") or SMTP_USER
    password = cfg.get("email_password") or SMTP_PASSWORD
    recipient = cfg.get("email_to") or REPORT_TO
    if not sender or not password or not recipient:
        return False
    message = EmailMessage()
    message["Subject"] = "Quant-AI alert"
    message["From"] = sender
    message["To"] = recipient
    message.set_content(text)
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(message)
        return True
    except Exception:  # noqa: BLE001
        return False


def send_text(text: str, config: dict | None = None) -> bool:
    """Dispatch a short text to the configured channel. Never raises."""
    cfg = config if config is not None else load_config()
    channel = str(cfg.get("channel", "none")).lower()
    if channel == "telegram":
        return _send_telegram(cfg, text)
    if channel == "email":
        return _send_email(cfg, text)
    return False


def send_alert(alert: dict, config: dict | None = None) -> bool:
    """Format and dispatch an alert. Returns True on success."""
    return send_text(format_alert(alert), config)


def send_test(config: dict | None = None) -> bool:
    """Send a test message (used by quant notify-setup)."""
    return send_text("Quant-AI test message. Notifications are working.", config)
