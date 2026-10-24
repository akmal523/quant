"""
notify.py — Telegram: the daily analysis digest and the Settings test message.

Intent: the phone is where Trade Republic lives. Once per trading day the 07:45
run (after every market of the previous day has closed) sends ONE plain-text
message: portfolio value and change, the market trend, the ONE list (Today /
This month), each holding against its target with its score, up to three
notable headlines for the holdings, and any data problem. It is sent even when
there is nothing to do ("Nothing to do"), because it is also the analysis.
Deduplicated by price date: a re-run for the same prices never resends.

v10.8.10 replaces three senders: an uncapped briefing on every ``quant run``,
one message per level-triggered alert, and a "list changed" rule that was never
wired up.

Invariants:
  - ``send`` never raises; returns True only on a real 2xx response.
  - At most one digest per price date.
  - The bot token never appears in any log or exception message.
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from typing import Any

from quant import paths

NOTIFY_FILE = os.path.join(str(paths.DATA_DIR), "notify.toml")
STATE_NAME = "notify_state.json"
DIGEST_HOUR_BEFORE = 12   # the morning slot sends; the evening slot only updates
MAX_ITEMS = 10
MAX_NEWS = 3


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
    """The plain Telegram state, without the "Telegram" prefix (v10.8.5, D14).

    Callers prepend the label, so the rendered line is "Telegram: on, last test
    ok" — never "Telegram: Telegram on, ...".
    """
    cfg = config if config is not None else load_config()
    channel = str(cfg.get("channel", "none")).lower()
    if channel == "telegram":
        return ("on, last test ok" if cfg.get("tested")
                else "on, not tested yet")
    return "off"


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


def send_test(config: dict | None = None) -> bool:
    """Send a test message (used by the inline Settings setup)."""
    return send("Quant-AI test message. Notifications are working.", config)


# ── The daily digest ──────────────────────────────────────────────────────────

def _fmt_eur(v: Any) -> str:
    return f"{float(v or 0):,.0f} EUR"


def build_digest(snap: dict) -> str:
    """The digest text from a snapshot (see :func:`digest_snapshot`). Pure."""
    from quant.engine.decisions import format_item_line
    from quant.ui import copy as C

    lines = [f"Quant-AI · {C.fmt_date(snap['today'])}"
             + (f" (prices of {C.fmt_date(snap['prices_date'])})"
                if snap.get("prices_date") else "")]
    if snap.get("total") is not None:
        _icon, _color, badge = C.delta_badge(snap.get("day_eur"), snap.get("day_pct"))
        since = snap.get("since_pct")
        lines.append(f"Portfolio {_fmt_eur(snap['total'])} · day {badge}"
                     + (f" · since first purchase {since:+.1f}%" if since is not None
                        else ""))
    if snap.get("regime_word"):
        lines.append(f"Market: {snap['regime_word']}")
    items = snap.get("items") or []
    lines.append("")
    if not items:
        lines.append("Nothing to do.")
    for horizon in (C.HORIZON_TODAY, C.HORIZON_MONTH):
        group = [i for i in items if i.get("horizon") == horizon][:MAX_ITEMS]
        if group:
            lines.append(f"{horizon}:")
            lines += [f"- {format_item_line(i)}: {i.get('reason') or ''}".rstrip(": ")
                      for i in group]
    if snap.get("holdings"):
        lines.append("")
        lines.append("Holdings:")
        lines += [f"- {h}" for h in snap["holdings"]]
    if snap.get("news"):
        lines.append("")
        lines.append("News:")
        lines += [f"- {n}" for n in snap["news"][:MAX_NEWS]]
    if snap.get("issues"):
        lines.append("")
        lines += [f"Note: {i}" for i in snap["issues"]]
    return "\n".join(lines)


def digest_snapshot(conn: Any, items: list[dict], holdings: list[dict],
                    today: date | None = None) -> dict:
    """Collect the digest inputs from the stores (reads only)."""
    from quant.engine import mtm
    from quant.engine.flows import load_flows, since_first_return
    from quant.portfolio.regime_constraints import regime_key
    from quant.reporting.artifacts import read_regime, read_score_row
    from quant.ui import copy as C

    today = today or date.today()
    total = mtm.latest_total(conn)
    day_eur, day_pct, _d = mtm.day_delta(conn)
    series = [(r["date"], r["value_eur"]) for r in mtm.read_total_series(conn)]
    since = since_first_return(series, load_flows(conn))
    regime = read_regime()
    key = regime_key(regime)
    rows = []
    for h in sorted(holdings, key=lambda x: -float(x.get("value_eur") or 0)):
        tgt = h.get("target_weight")
        score = (read_score_row(str(h["symbol"])) or {}).get("composite")
        now_pct = float(h.get("current_weight") or 0) * 100
        rows.append(f"{h.get('name') or h['symbol']}: {now_pct:.0f}%"
                    + (f" (target {float(tgt) * 100:.0f}%)" if tgt is not None else "")
                    + (f", score {float(score):.0f}" if score is not None else ""))
    news = []
    held = [str(h["symbol"]) for h in holdings]
    if held:
        marks = ",".join("?" for _ in held)
        try:
            for sym, title, score in conn.execute(
                    "SELECT symbol, title, score FROM nlp_evidence WHERE symbol IN "
                    f"({marks}) AND score IS NOT NULL AND retrieved_at >= ? "
                    "ORDER BY abs(score) DESC LIMIT ?",
                    [*held, datetime.combine(today, datetime.min.time())
                     - timedelta(days=2), MAX_NEWS]).fetchall():
                tone = "positive" if score > 0.15 else "negative" if score < -0.15 else "neutral"
                news.append(f"{sym}: {title} ({tone})")
        except Exception:  # noqa: BLE001
            news = []
    issues = []
    try:
        for stage, status, error in conn.execute(
                "SELECT stage, status, error FROM run_manifest WHERE run_id = "
                "(SELECT MAX(run_id) FROM run_manifest) AND status IN ('failed','missing')"
        ).fetchall():
            issues.append(f"{stage} {status}" + (f" ({error})" if error else ""))
    except Exception:  # noqa: BLE001
        issues = []
    return {"today": today, "prices_date": total.get("date"),
            "total": total.get("value_eur") if total else None,
            "day_eur": day_eur, "day_pct": day_pct,
            "since_pct": None if since is None else since * 100.0,
            "regime_word": (f"{C.regime_word(key)} ({regime.get('confidence')} confidence)"
                            if key else ""),
            "items": items, "holdings": rows, "news": news, "issues": issues}


def _state_path() -> str:
    return os.path.join(str(paths.OUTPUTS_DIR), STATE_NAME)


def _read_state() -> dict:
    try:
        with open(_state_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def send_daily_digest(conn: Any, items: list[dict], holdings: list[dict],
                      today: date | None = None, now: datetime | None = None,
                      sender: Any = None, force: bool = False) -> bool:
    """Send the digest once per price date, from the morning slot. Never raises."""
    now = now or datetime.now()
    if not force and now.hour >= DIGEST_HOUR_BEFORE:
        return False
    try:
        snap = digest_snapshot(conn, items, holdings, today)
    except Exception:  # noqa: BLE001
        return False
    key = str(snap.get("prices_date") or snap["today"])
    if not force and _read_state().get("last_digest_prices") == key:
        return False
    ok = bool((sender or send)(build_digest(snap)))
    if ok:
        try:
            os.makedirs(str(paths.OUTPUTS_DIR), exist_ok=True)
            with open(_state_path(), "w", encoding="utf-8") as f:
                json.dump({"last_digest_prices": key, "sent_at": now.isoformat()}, f)
        except Exception:  # noqa: BLE001
            pass
    return ok
