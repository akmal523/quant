"""
alerts.py — Level-triggered alert conditions and the honesty ledger (v10.7.0).

Intent: the laptop sleeps; edge-triggered alerts would be missed. Every
condition inspects the CURRENT state, so a monitoring gap delays an alert but
never loses it. An alert stays open until the user resolves it. When resolved,
the price is recorded; a scheduled check prices it 30 days later and records a
verdict (the advice hit-rate ledger).

Conditions (Section 5):
  1. structural_break   structural grade below 40, or dropped >= 20 since the
                        last monthly decision (EQUITY holdings only).
  2. tactical_collapse  tactical grade dropped >= 25 within 7 days.
  3. position_crash     holding value down >= 15 percent within 7 days.
  4. regime_flip        market regime changed to bear since the previous run.
  5. speculative_stop   a SPECULATIVE holding down 50 percent from entry.

Invariants:
  - No duplicate OPEN alert for the same (symbol, kind).
  - evaluate_alerts never raises; returns the list of newly created alerts.
  - I/O helpers take an explicit connection.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from quant.config import (
    ALERT_POSITION_CRASH,
    ALERT_SPECULATIVE_STOP,
    ALERT_STRUCTURAL_DROP,
    ALERT_STRUCTURAL_FLOOR,
    ALERT_TACTICAL_DROP,
    SPARPLAN_SELL_FEE_EUR,
)
from quant.engine import sizing

KIND_STRUCTURAL = "structural_break"
KIND_TACTICAL = "tactical_collapse"
KIND_CRASH = "position_crash"
KIND_REGIME = "regime_flip"
KIND_SPECULATIVE = "speculative_stop"

OPEN_STATUS = "new"
DONE_STATUS = "done"
DECLINED_STATUS = "declined"


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sell_action(tier: str, value_eur: float) -> dict | None:
    """A sell action obeying the sizing laws, or None when forbidden."""
    amount = sizing.can_sell(tier, value_eur, value_eur)
    if amount is None:
        return None
    return {"action": "sell", "amount_eur": amount, "fee_eur": SPARPLAN_SELL_FEE_EUR}


def has_open_alert(conn, symbol: str | None, kind: str) -> bool:
    """True when an open alert already exists for (symbol, kind)."""
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM alerts WHERE status = ? AND kind = ? "
            "AND ((symbol IS NULL AND ? IS NULL) OR symbol = ?)",
            [OPEN_STATUS, kind, symbol, symbol],
        ).fetchone()
    except Exception:  # noqa: BLE001
        return False
    return bool(row and row[0])


def _insert(conn, alert: dict, today: date) -> None:
    conn.execute(
        "INSERT INTO alerts (created_date, symbol, kind, message, action, "
        "amount_eur, status, notified) VALUES (?, ?, ?, ?, ?, ?, ?, FALSE)",
        [
            today,
            alert.get("symbol"),
            alert["kind"],
            alert.get("message"),
            alert.get("action"),
            alert.get("amount_eur"),
            OPEN_STATUS,
        ],
    )


def evaluate_alerts(
    conn,
    holdings: list[dict],
    regime: str | None = None,
    prev_regime: str | None = None,
    today: date | None = None,
) -> list[dict]:
    """Evaluate the five conditions and create new alerts. Returns new alerts."""
    today = today or date.today()
    created: list[dict] = []

    def _emit(alert: dict) -> None:
        if has_open_alert(conn, alert.get("symbol"), alert["kind"]):
            return
        _insert(conn, alert, today)
        created.append(alert)

    for holding in holdings or []:
        symbol = str(holding.get("symbol", "")).strip()
        if not symbol:
            continue
        name = holding.get("name") or symbol
        tier = str(holding.get("tier", "")).upper()
        value = _num(holding.get("value_eur")) or 0.0
        structural = _num(holding.get("structural"))
        tactical = _num(holding.get("tactical"))
        prev_structural = _num(holding.get("prev_structural"))
        prev_tactical = _num(holding.get("prev_tactical_7d"))
        prev_value = _num(holding.get("prev_value_7d"))
        entry = _num(holding.get("entry_price"))
        price = _num(holding.get("current_price"))

        # 1. Structural break (EQUITY only; ETFs use the ETF structural grade).
        if structural is not None:
            broke = structural < ALERT_STRUCTURAL_FLOOR
            dropped = (prev_structural is not None
                       and (prev_structural - structural) >= ALERT_STRUCTURAL_DROP)
            if broke or dropped:
                action = _sell_action(tier, value)
                if action is None:
                    action = {"action": "review", "amount_eur": None, "fee_eur": None}
                _emit({
                    "symbol": symbol,
                    "kind": KIND_STRUCTURAL,
                    "message": f"{name}: structure fell to {structural:.0f}.",
                    **action,
                })

        # 2. Tactical collapse.
        if tactical is not None and prev_tactical is not None:
            if (prev_tactical - tactical) >= ALERT_TACTICAL_DROP:
                action = _sell_action(tier, value)
                if action is None:
                    action = {"action": "review", "amount_eur": None, "fee_eur": None}
                _emit({
                    "symbol": symbol,
                    "kind": KIND_TACTICAL,
                    "message": (f"{name}: tactics fell from {prev_tactical:.0f} to "
                                f"{tactical:.0f} in 7 days."),
                    **action,
                })

        # 3. Position crash.
        if prev_value and prev_value > 0 and value > 0:
            drop = (prev_value - value) / prev_value
            if drop >= ALERT_POSITION_CRASH:
                action = _sell_action(tier, value)
                if action is None:
                    action = {"action": "review", "amount_eur": None, "fee_eur": None}
                _emit({
                    "symbol": symbol,
                    "kind": KIND_CRASH,
                    "message": f"{name}: value fell {drop * 100:.0f} percent in 7 days.",
                    **action,
                })

        # 5. Speculative stop-loss.
        if tier == "SPECULATIVE" and entry and entry > 0 and price is not None:
            change = price / entry - 1.0
            if change <= ALERT_SPECULATIVE_STOP:
                action = _sell_action(tier, value)
                if action is None:
                    action = {"action": "review", "amount_eur": None, "fee_eur": None}
                _emit({
                    "symbol": symbol,
                    "kind": KIND_SPECULATIVE,
                    "message": (f"{name}: down {abs(change) * 100:.0f} percent from "
                                f"entry; stop-loss."),
                    **action,
                })

    # 4. Regime flip (market-level, one alert).
    if str(regime).lower() == "bear" and str(prev_regime).lower() != "bear":
        _emit({
            "symbol": None,
            "kind": KIND_REGIME,
            "message": ("Market regime turned falling. Affects only the Active part; "
                        "new active money goes to cash until the regime recovers. "
                        "Long-term savings plan continues."),
            "action": "hold",
            "amount_eur": None,
            "fee_eur": None,
        })

    return created


def open_alerts(conn) -> list[dict]:
    """All open alerts, newest first."""
    try:
        rows = conn.execute(
            "SELECT id, created_date, symbol, kind, message, action, amount_eur "
            "FROM alerts WHERE status = ? ORDER BY id DESC",
            [OPEN_STATUS],
        ).fetchall()
    except Exception:  # noqa: BLE001
        return []
    return [
        {
            "id": r[0],
            "created_date": r[1],
            "symbol": r[2],
            "kind": r[3],
            "message": r[4],
            "action": r[5],
            "amount_eur": r[6],
        }
        for r in rows
    ]


def resolve_alert(
    conn,
    alert_id: int,
    status: str,
    reason: str | None = None,
    price: float | None = None,
    today: date | None = None,
) -> bool:
    """Resolve an alert (done/declined). Returns True when a row was updated."""
    today = today or date.today()
    if status not in (DONE_STATUS, DECLINED_STATUS):
        return False
    try:
        conn.execute(
            "UPDATE alerts SET status = ?, resolve_date = ?, resolve_reason = ?, "
            "price_at_resolve = ? WHERE id = ?",
            [status, today, reason, price, alert_id],
        )
    except Exception:  # noqa: BLE001
        return False
    return True


def score_resolved_alerts(
    conn,
    price_lookup: Any,
    today: date | None = None,
    horizon_days: int = 30,
) -> int:
    """Price resolved alerts at 30 days and record a verdict. Returns rows scored.

    For a "sell" advice that was done, correct if the price 30 days later is
    lower; for a declined sell, correct if the price is higher.
    """
    today = today or date.today()
    try:
        rows = conn.execute(
            "SELECT id, symbol, action, status, resolve_date, price_at_resolve "
            "FROM alerts WHERE status IN (?, ?) AND verdict IS NULL",
            [DONE_STATUS, DECLINED_STATUS],
        ).fetchall()
    except Exception:  # noqa: BLE001
        return 0
    scored = 0
    for alert_id, symbol, action, status, resolve_date, price_at_resolve in rows:
        if resolve_date is None:
            continue
        resolved = resolve_date
        if isinstance(resolved, str):
            resolved = date.fromisoformat(resolved[:10])
        if (today - resolved).days < horizon_days:
            continue
        price_now = None
        try:
            price_now = price_lookup(symbol) if callable(price_lookup) else price_lookup.get(symbol)
        except Exception:  # noqa: BLE001
            price_now = None
        if price_now is None or price_at_resolve is None:
            continue
        went_down = float(price_now) < float(price_at_resolve)
        if str(action) == "sell":
            correct = went_down if status == DONE_STATUS else (not went_down)
        else:
            correct = True
        verdict = "correct" if correct else "wrong"
        conn.execute(
            "UPDATE alerts SET price_30d = ?, verdict = ? WHERE id = ?",
            [float(price_now), verdict, alert_id],
        )
        scored += 1
    return scored


def advice_record_line(conn, months: int = 12) -> str:
    """The Settings one-line advice record (Section 5)."""
    from quant.ui import copy as ui_copy

    try:
        rows = conn.execute(
            "SELECT verdict FROM alerts WHERE verdict IS NOT NULL"
        ).fetchall()
    except Exception:  # noqa: BLE001
        rows = []
    total = len(rows)
    correct = sum(1 for r in rows if r[0] == "correct")
    wrong = total - correct
    return ui_copy.ADVICE_RECORD.format(n=total, correct=correct, wrong=wrong)
