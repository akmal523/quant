"""
today_list.py — the ONE list's items, computed once and persisted by writers only.

Intent (v10.8.10, R-C4): the Portfolio page used to compute the list AND write the
advice log on every render. The computation now lives here; the render path calls
:func:`build_items` and :func:`quant.engine.advice_state.annotate` (read-only), and
only the writers (the daily job, confirm and refresh) call :func:`persist`, which
reconciles the advice log via :func:`quant.engine.advice_state.sync`.

The list is the one rule set in :mod:`quant.engine.action_taxonomy` (rebalance,
plan change, threat, swap) over the live holdings, their 7-day price change,
the scores and the ideas.

Invariants:
  - ``build_items`` / ``scores_map`` / ``idea_rows`` / ``change_7d`` are reads only.
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


def change_7d(symbols: list[str], conn: Any = None) -> dict:
    """Per symbol: the close-to-close change over the last 7 trading days."""
    if not symbols:
        return {}
    marks = ",".join("?" for _ in symbols)
    rows = _rows(
        "SELECT Symbol, Close FROM (SELECT Symbol, Close, ROW_NUMBER() OVER "
        "(PARTITION BY Symbol ORDER BY Date DESC) AS rn FROM market_history "
        f"WHERE Symbol IN ({marks}) AND Close IS NOT NULL) WHERE rn IN (1, 8) "
        "ORDER BY Symbol, rn", list(symbols), conn=conn)
    closes: dict = {}
    for sym, close in rows:
        closes.setdefault(str(sym), []).append(float(close))
    return {s: c[0] / c[1] - 1.0 for s, c in closes.items() if len(c) == 2 and c[1] > 0}


_DIRECTIONS = {"sell_part": ui_copy.DIRECTION_SELL, "swap": ui_copy.DIRECTION_SELL,
               "threat": "Review"}


def build_items(holdings: list[dict] | None, regime: Any = None,
                scores: dict | None = None, ideas: list[dict] | None = None) -> list[dict]:
    """The ONE list (Today before This month) from the one rule set. Reads only."""
    from quant.engine import action_taxonomy as T

    holdings = list(holdings or [])
    scores = scores_map() if scores is None else scores
    ideas = idea_rows(holdings, scores) if ideas is None else ideas
    moves = change_7d([str(h.get("symbol")) for h in holdings])
    holdings = [{**h, "change_7d": moves.get(str(h.get("symbol")))} for h in holdings]
    return [{**t, "direction": _DIRECTIONS.get(t["verb"], ui_copy.DIRECTION_BUY)}
            for t in T.build(holdings, scores=scores, ideas=ideas, regime=regime)]


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
