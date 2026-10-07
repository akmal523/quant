"""
flows.py — Money flows and honest performance math (v10.7.0, Section 7).

Intent: the old "+2.5 percent since 13 Sep" is a naive value change and lies
whenever money enters. Modified Dietz excludes deposits:

    R = (V_end - V_start - F) / (V_start + sum(w_i * F_i))
    w_i = (days_in_period - days_from_start_to_flow) / days_in_period

Deposits do not count as profit. Flow signs: buy = +amount (money entered
invested), sell = -amount (money left invested), dividend = -amount (left
invested, landed in cash).

Invariants:
  - modified_dietz is pure (no I/O).
  - I/O helpers take an explicit connection; they never open their own.
  - Sparplan auto-flows are tagged with a note so actuals can replace them.
"""
from __future__ import annotations

from datetime import date
from typing import Any

FLOW_TYPES = ("buy", "sell", "dividend")
_FLOW_SIGN = {"buy": 1.0, "sell": -1.0, "dividend": -1.0}


def flow_sign(flow_type: str) -> float:
    """Signed direction of a flow for the return math."""
    return _FLOW_SIGN.get(str(flow_type).lower(), 0.0)


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def record_flow(
    conn: Any,
    flow_date: date,
    flow_type: str,
    amount_eur: float,
    symbol: str | None = None,
    note: str | None = None,
) -> None:
    """Insert one flow row. amount_eur is stored positive; type carries the sign."""
    conn.execute(
        "INSERT INTO flows (date, type, amount_eur, symbol, note) VALUES (?, ?, ?, ?, ?)",
        [flow_date, str(flow_type), float(amount_eur), symbol, note],
    )


def load_flows(conn: Any, start: date | None = None, end: date | None = None) -> list[dict]:
    """Load flows in [start, end] (inclusive), ordered by date."""
    query = "SELECT id, date, type, amount_eur, symbol, note FROM flows"
    clauses: list[str] = []
    params: list = []
    if start is not None:
        clauses.append("date >= ?")
        params.append(start)
    if end is not None:
        clauses.append("date <= ?")
        params.append(end)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY date"
    try:
        rows = conn.execute(query, params).fetchall()
    except Exception:  # noqa: BLE001
        return []
    return [
        {
            "id": r[0],
            "date": r[1],
            "type": r[2],
            "amount_eur": r[3],
            "symbol": r[4],
            "note": r[5],
        }
        for r in rows
    ]


def modified_dietz(
    v_start: float,
    v_end: float,
    flows: list[dict],
    start: date,
    end: date,
) -> float | None:
    """Modified Dietz return over [start, end]. Returns a fraction or None.

    flows: list of dicts with 'date', 'type', and 'amount_eur' (positive). The
    sign is applied from the type. Returns None when the period is empty or the
    denominator is non-positive.
    """
    start_d = _as_date(start)
    end_d = _as_date(end)
    total_days = (end_d - start_d).days
    if total_days <= 0:
        return None
    net_flow = 0.0
    weighted = 0.0
    for flow in flows:
        amount = float(flow.get("amount_eur", 0) or 0) * flow_sign(flow.get("type", "buy"))
        flow_date = _as_date(flow.get("date"))
        days_from_start = (flow_date - start_d).days
        weight = (total_days - days_from_start) / total_days
        net_flow += amount
        weighted += weight * amount
    denominator = float(v_start) + weighted
    if denominator <= 0:
        return None
    return (float(v_end) - float(v_start) - net_flow) / denominator


def performance_line(
    v_start: float,
    v_end: float,
    flows: list[dict],
    start: date,
    end: date,
) -> str:
    """The Overview sentence: change, deposits, market move, and return.

    Example: "Since 13 Sep: +20.84 EUR. Of that: you added 0.00 EUR, market
    moved +20.84 EUR. Return: +2.4 percent."
    """
    from quant.ui import copy as ui_copy

    net_flow = sum(
        float(f.get("amount_eur", 0) or 0) * flow_sign(f.get("type", "buy"))
        for f in flows
    )
    change = float(v_end) - float(v_start)
    market = change - net_flow
    ret = modified_dietz(v_start, v_end, flows, start, end)
    ret_str = "n/a" if ret is None else f"{ret * 100:+.1f}"
    return ui_copy.PERFORMANCE_LINE.format(
        date=ui_copy.fmt_date(start),
        change=f"{change:+.2f}",
        added=f"{net_flow:+.2f}",
        market=f"{market:+.2f}",
        ret=ret_str,
    )


def _planned_note(month: str) -> str:
    return f"planned sparplan {month}"


def _actual_note(month: str) -> str:
    return f"actual sparplan {month}"


def write_planned_sparplan_flows(
    conn: Any,
    month: str,
    legs: list[dict],
    execution_date: date,
) -> int:
    """Write planned buy flows for an approved monthly plan. Returns rows written."""
    written = 0
    for leg in legs:
        record_flow(
            conn,
            execution_date,
            "buy",
            leg.get("amount_eur", 0),
            leg.get("symbol"),
            note=_planned_note(month),
        )
        written += 1
    return written


def replace_auto_flows_with_actuals(
    conn: Any,
    month: str,
    actuals: list[dict],
) -> list[dict]:
    """Replace planned Sparplan flows with actuals. Returns reconciliation rows.

    Each reconciliation row: {symbol, planned, actual, deviation}. Conscious
    deviations are recorded, not judged.
    """
    planned = {
        f["symbol"]: float(f["amount_eur"] or 0)
        for f in load_flows(conn)
        if f.get("note") == _planned_note(month)
    }
    conn.execute("DELETE FROM flows WHERE note = ?", [_planned_note(month)])
    out: list[dict] = []
    for actual in actuals:
        symbol = actual.get("symbol")
        amount = float(actual.get("amount_eur", 0) or 0)
        record_flow(
            conn,
            actual.get("date"),
            "buy",
            amount,
            symbol,
            note=_actual_note(month),
        )
        planned_amount = planned.get(symbol, 0.0)
        out.append(
            {
                "symbol": symbol,
                "planned": planned_amount,
                "actual": amount,
                "deviation": amount - planned_amount,
            }
        )
    return out
