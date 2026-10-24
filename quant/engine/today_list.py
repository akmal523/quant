"""
today_list.py — the ONE list's items, computed once and persisted by writers only.

Intent (v10.8.10, R-C4): the Portfolio page used to compute the list AND write the
advice log on every render. The computation now lives here; the render path calls
:func:`build_items` and :func:`quant.engine.advice_state.annotate` (read-only), and
only the writers (the daily job, confirm and refresh) call :func:`persist`, which
reconciles the advice log via :func:`quant.engine.advice_state.sync`.

The list is: drift advice (:func:`quant.engine.advice.build_advice` ->
:func:`quant.engine.decisions.timed_items`) plus the threat / swap / plan-change
classes of :mod:`quant.engine.action_taxonomy`. A symbol with a swap never also
carries a standalone buy/sell (the swap is the paired action).

Invariants:
  - ``build_items`` / ``scores_map`` / ``idea_rows`` / ``months_under_target`` are
    reads only.
  - ``persist`` never raises; it returns the number of items synced.
"""
from __future__ import annotations

from typing import Any

from quant.ui import copy as ui_copy


def _rows(sql: str, params: list | None = None, conn: Any = None) -> list:
    try:
        if conn is not None:
            return conn.execute(sql, params or []).fetchall()
        from quant.data.database import read_only_connection

        with read_only_connection() as c:
            return c.execute(sql, params or []).fetchall()
    except Exception:  # noqa: BLE001
        return []


def _display(sym: str) -> str:
    try:
        from quant.data.names import display_name

        return display_name(sym)
    except Exception:  # noqa: BLE001
        return sym


def scores_map(conn: Any = None) -> dict:
    """The scores store as ``{symbol: {composite, structural_grade, confidence}}``."""
    rows = _rows("SELECT symbol, composite, structural_grade, confidence FROM scores",
                 conn=conn)
    return {str(sym): {"composite": comp, "structural_grade": grade, "confidence": conf}
            for sym, comp, grade, conf in rows}


def idea_rows(holdings: list[dict] | None, scores: dict | None = None,
              limit: int = 50) -> list[dict]:
    """Watched (non-held) symbols with a composite >= ``IDEA_FLOOR``, best first.

    Only high-confidence scores qualify: a stock needs its fundamentals, an ETF
    needs news, so a pure price trend never becomes an Idea.
    """
    from quant.config import IDEA_FLOOR

    scores = scores_map() if scores is None else scores
    held = {str(h.get("symbol")) for h in holdings or []}
    out = [{"symbol": sym, "name": _display(sym), "score": float(s["composite"])}
           for sym, s in scores.items()
           if sym not in held and s.get("composite") is not None
           and str(s.get("confidence") or "high") == "high"
           and float(s["composite"]) >= IDEA_FLOOR]
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:limit]


def months_under_target(holdings: list[dict] | None, gap: float | None = None,
                        conn: Any = None) -> dict:
    """Per Long-term symbol: consecutive month-ends (newest first) under target.

    A month counts when the symbol's share of the invested total on the last
    valuation date of that month is more than ``gap`` below its target weight.
    Read from the valuation store, so it needs no extra state.
    """
    from quant.config import PLAN_CHANGE_GAP

    gap = PLAN_CHANGE_GAP if gap is None else gap
    targets = {str(h.get("symbol")): float(h.get("target_weight") or 0)
               for h in holdings or []
               if str(h.get("tier", "")).upper() == "FORTRESS"}
    if not targets:
        return {}
    rows = _rows(
        "WITH month_end AS (SELECT max(date) AS d FROM valuation "
        "GROUP BY date_trunc('month', date)) "
        "SELECT v.date, v.symbol, v.value_eur FROM valuation v "
        "JOIN month_end m ON v.date = m.d ORDER BY v.date DESC", conn=conn)
    by_date: dict = {}
    for d, sym, value in rows:
        by_date.setdefault(d, {})[str(sym)] = float(value or 0)
    out = dict.fromkeys(targets, 0)
    open_syms = set(targets)
    for d in sorted(by_date, reverse=True):
        values = by_date[d]
        total = sum(values.values())
        if total <= 0:
            break
        for sym in list(open_syms):
            if targets[sym] - values.get(sym, 0.0) / total > gap:
                out[sym] += 1
            else:
                open_syms.discard(sym)
        if not open_syms:
            break
    return out


def _taxonomy_items(holdings: list[dict], regime: Any, scores: dict,
                    ideas: list[dict], months_under: dict) -> list[dict]:
    """Threat / swap / plan-change items in the ONE-list shape.

    Rebalance stays with ``build_advice`` (it yields the drift buy/sell).
    """
    from quant.engine import action_taxonomy as T

    directions = {T.VERB_SWAP: ui_copy.DIRECTION_SELL, T.VERB_THREAT: "Review",
                  T.VERB_PLAN: ui_copy.DIRECTION_BUY}
    out: list[dict] = []
    for t in T.build(holdings, scores=scores, ideas=ideas, regime=regime,
                     months_under=months_under):
        if t["verb"] == T.VERB_REBALANCE:
            continue
        out.append({
            "horizon": t["horizon"], "class": t["class"],
            "direction": directions.get(t["verb"], ui_copy.DIRECTION_BUY),
            "verb": t["verb"], "label": t["label"], "amount_eur": t["amount_eur"],
            "reason": t["reason"], "symbol": t["symbol"],
            "evidence": t.get("evidence"),
            "target_symbol": t.get("target_symbol"),
            "target_name": t.get("target_name"),
        })
    return out


def build_items(holdings: list[dict] | None, regime: Any = None,
                scores: dict | None = None, ideas: list[dict] | None = None,
                months_under: dict | None = None) -> list[dict]:
    """The ONE list (Today before This month). Reads only; never raises."""
    from quant.engine.advice import build_advice
    from quant.engine.decisions import timed_items

    holdings = list(holdings or [])
    scores = scores_map() if scores is None else scores
    ideas = idea_rows(holdings, scores) if ideas is None else ideas
    if months_under is None:
        months_under = months_under_target(holdings)
    try:
        advice, _rejected = build_advice(holdings=holdings, regime=regime)
    except Exception:  # noqa: BLE001
        advice = []
    taxo = _taxonomy_items(holdings, regime, scores, ideas, months_under)
    # A swappable position never gets a standalone buy/sell (R-T4).
    swap_syms = {str(t["symbol"]) for t in taxo if t["verb"] == "swap"}
    drift = [i for i in timed_items(holdings, advice)
             if str(i.get("symbol")) not in swap_syms]
    order = {ui_copy.HORIZON_TODAY: 0, ui_copy.HORIZON_MONTH: 1}
    items = drift + taxo
    items.sort(key=lambda i: order.get(i.get("horizon"), 2))
    return items


def persist(conn: Any = None) -> int:
    """Writers only: rebuild the list from stored data and sync the advice log."""
    try:
        from quant.engine import advice_state
        from quant.engine.holdings_view import holdings_view
        from quant.reporting.artifacts import read_regime

        holdings = holdings_view()
        items = build_items(holdings, regime=read_regime())
        if conn is not None:
            return len(advice_state.sync(conn, items))
        from quant.data.database import write_connection

        with write_connection() as c:
            return len(advice_state.sync(c, items))
    except Exception:  # noqa: BLE001
        return 0
