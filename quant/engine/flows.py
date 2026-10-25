"""
flows.py — honest performance math (v10.7.0, Section 7; trimmed v10.8.2).

Intent: the old "+2.5 percent since 13 Sep" is a naive value change and lies
whenever money enters. Modified Dietz excludes deposits:

    R = (V_end - V_start - F) / (V_start + sum(w_i * F_i))
    w_i = (days_in_period - days_from_start_to_flow) / days_in_period

Deposits do not count as profit. Flow signs: buy = +amount (money entered
invested), sell = -amount (money left invested), dividend = -amount (left
invested, landed in cash).

v10.8.2: the manual Sparplan flow writers were removed with the Monthly decision
page. The ledger is written by quant.engine.confirm and quant.engine.ledger.

Invariants:
  - modified_dietz is pure (no I/O).
  - load_flows takes an explicit connection; it never opens its own.
"""
from __future__ import annotations

from datetime import date
from typing import Any

_BUY = "buy"   # a flow without a type is a buy (money entered invested)
_FLOW_SIGN = {_BUY: 1.0, "sell": -1.0, "dividend": -1.0}


def flow_sign(flow_type: str) -> float:
    """Signed direction of a flow for the return math."""
    return _FLOW_SIGN.get(str(flow_type).lower(), 0.0)


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def load_flows(conn: Any, start: date | None = None, end: date | None = None) -> list[dict]:
    """Load flows in [start, end] (inclusive), ordered by date."""
    # SQL keywords and column names are case-insensitive, so mutating their
    # case cannot change behavior; the SQL text lines are excluded from mutation.
    query = "SELECT id, date, type, amount_eur, symbol, note FROM flows"  # pragma: no mutate
    clauses: list[str] = []
    params: list = []
    if start is not None:
        clauses.append("date >= ?")  # pragma: no mutate
        params.append(start)
    if end is not None:
        clauses.append("date <= ?")  # pragma: no mutate
        params.append(end)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)  # pragma: no mutate
    query += " ORDER BY date"  # pragma: no mutate
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
        amount = float(flow.get("amount_eur") or 0) * flow_sign(flow.get("type", _BUY))
        flow_date = _as_date(flow.get("date"))
        days_from_start = (flow_date - start_d).days
        weight = (total_days - days_from_start) / total_days
        net_flow += amount
        weighted += weight * amount
    denominator = float(v_start) + weighted
    if denominator <= 0:
        return None
    return (float(v_end) - float(v_start) - net_flow) / denominator


def since_first_return(series: list[tuple], flows: list[dict]) -> float | None:
    """Flow-adjusted return over a value series, as a fraction (v10.8.10).

    ``series`` is ``[(date, value_eur), ...]`` in date order. The start value
    already contains every flow up to and including the first date, so only the
    flows strictly after it enter the Modified Dietz sum: a deposit (buy) is
    never counted as gain. None when there are fewer than two points.
    """
    if len(series or []) < 2:
        return None
    (d0, v0), (d1, v1) = series[0], series[-1]
    start, end = _as_date(d0), _as_date(d1)
    later = [f for f in flows or [] if start < _as_date(f.get("date")) <= end]
    return modified_dietz(float(v0), float(v1), later, start, end)
