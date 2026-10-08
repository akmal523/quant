"""
notify.py — Telegram dispatch and the notify rule (v10.7.0; rule v10.8.2, section 7).

Intent: the user is often away from the laptop; the phone is where Trade Republic
lives. Telegram is the only channel, sent with the standard library (urllib). A
short message is sent after a confirmed save or the daily check when there is
something to do AND the list changed since the last message, never more than two
per local calendar day. A failed send never crashes the job and never reports
success.

Invariants:
  - ``send`` never raises; returns True only on a real 2xx response.
  - At most two messages per local calendar day; identical content is not resent.
  - The bot token never appears in any log or exception message.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

from quant import paths

NOTIFY_FILE = os.path.join(str(paths.DATA_DIR), "notify.toml")
STATE_NAME = "notify_state.json"
MAX_PER_DAY = 2
FINGERPRINT_ROUND = 10
MAX_MESSAGE_LINES = 8


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
        return ("Telegram on, last test ok" if cfg.get("tested")
                else "Telegram on, not tested yet")
    return "Telegram off"


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


def send(text: str, config: dict | None = None) -> bool:
    """The single dispatch function (Telegram). Never raises."""
    cfg = config if config is not None else load_config()
    if str(cfg.get("channel", "none")).lower() != "telegram":
        return False
    return _send_telegram(cfg, text)


def send_text(text: str, config: dict | None = None) -> bool:
    """Backward-compatible alias for :func:`send`."""
    return send(text, config)


def format_alert(alert: dict) -> str:
    """The legacy alert phrasing (kept until the alert system is folded into
    the decision list, v10.8.2). Never raises."""
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


def send_alert(alert: dict, config: dict | None = None) -> bool:
    """Legacy: format and dispatch an alert. Never raises."""
    return send(format_alert(alert), config)


def send_test(config: dict | None = None) -> bool:
    """Send a test message (used by the inline Settings setup)."""
    return send("Quant-AI test message. Notifications are working.", config)


# ── The notify rule (v10.8.2, section 7) ──────────────────────────────────────

def _state_path() -> str:
    return os.path.join(str(paths.OUTPUTS_DIR), STATE_NAME)


def _load_state() -> dict:
    try:
        with open(_state_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _save_state(state: dict) -> None:
    try:
        os.makedirs(str(paths.OUTPUTS_DIR), exist_ok=True)
        with open(_state_path(), "w", encoding="utf-8") as f:
            json.dump(state, f)
    except Exception:  # noqa: BLE001
        pass


def _round10(amount) -> int:
    if isinstance(amount, bool) or not isinstance(amount, int | float):
        return 0
    return int(round(float(amount) / FINGERPRINT_ROUND) * FINGERPRINT_ROUND)


def message_fingerprint(items: list[dict]) -> str:
    """A stable identity of an item list (symbol + verb + amount to the 10 EUR)."""
    parts = []
    for it in items or []:
        sym = str(it.get("symbol") or it.get("label") or "")
        verb = str(it.get("verb") or it.get("kind") or "")
        parts.append(f"{verb}:{sym}:{_round10(it.get('amount_eur'))}")
    return "|".join(sorted(parts))


def build_message(items: list[dict]) -> str:
    """The plain-text message: at most 8 lines, one item per line."""
    from quant.engine.decisions import format_item_line

    lines = [f"Quant-AI: {len(items)} things to do"]
    for it in items:
        lines.append(format_item_line(it))
        if len(lines) >= MAX_MESSAGE_LINES:
            break
    return "\n".join(lines)


def notify_decisions(items: list[dict], today=None, config: dict | None = None,
                     sender=None) -> bool:
    """Send one message if the actionable list is non-empty, new, and under cap.

    Returns True only when a message was actually sent (v10.8.2, section 7).
    """
    from datetime import date as _date

    from quant.engine.decisions import actionable_items

    actionable = actionable_items(items)
    if not actionable:
        return False
    today = today or _date.today()
    day = today.isoformat()
    state = _load_state()
    if state.get("day") != day:
        state = {"day": day, "count": 0, "fingerprint": state.get("fingerprint")}
    if int(state.get("count", 0)) >= MAX_PER_DAY:
        return False
    fingerprint = message_fingerprint(actionable)
    if fingerprint == state.get("fingerprint"):
        return False  # nothing changed since the last message
    text = build_message(actionable)
    fn = sender or send
    ok = fn(text, config) if config is not None else fn(text)
    if ok:
        state["count"] = int(state.get("count", 0)) + 1
        state["fingerprint"] = fingerprint
        _save_state(state)
    return bool(ok)
