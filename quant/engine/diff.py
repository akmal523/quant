"""
diff.py — the Review diff against the last saved table (v10.8.2, section 3).

Intent: saving the table is the only way data enters the system. Before writing
anything, the app shows what changed per ticker: Bought, Sold (with an estimated
realized gain/loss), or a market move. Derived, never typed: invested = Value -
Profit; shares = invested / Einstandskurs.

Invariants:
  - Pure: no I/O.
  - A share change below 0.5 percent of the position, or below 1 EUR of invested,
    counts as unchanged (rounding).
  - A sell's realized gain is an estimate; Trade Republic's tax report is the
    authority. Proceeds use the latest EUR price when known.
"""
from __future__ import annotations

from typing import Any

TOL_SHARE_FRAC = 0.005
TOL_INVESTED_EUR = 1.0
# v10.8.2 (B6): a buy within this many EUR of the saved plan is labeled a
# savings-plan buy.
PLAN_TOLERANCE_EUR = 5.0


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def position(row: dict) -> dict:
    """One position with derived invested and shares (Einstandskurs based)."""
    symbol = str(row.get("Symbol") or row.get("symbol") or "").strip()
    entry = _num(row.get("Avg_Entry_Price", row.get("entry")))
    value = _num(row.get("Current_Value_EUR", row.get("value")))
    profit = _num(row.get("Broker_PnL_EUR", row.get("profit")))
    invested = value - profit
    shares = invested / entry if entry > 0 else 0.0
    return {"symbol": symbol, "entry": entry, "value": value, "profit": profit,
            "invested": invested, "shares": shares}


def _index(rows: list[dict] | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in rows or []:
        p = position(r)
        if p["symbol"]:
            out[p["symbol"]] = p
    return out


def _shares_unchanged(prev: dict, d_shares: float) -> bool:
    """Rounding tolerance on the share count: below 0.5 percent or below 1 EUR."""
    return (abs(d_shares) < TOL_SHARE_FRAC * prev["shares"]
            or abs(d_shares) * max(prev["entry"], 1.0) < TOL_INVESTED_EUR)


def _matches_plan(symbol: str, amount: float, plans: dict | None) -> bool:
    """True when a buy amount matches the saved monthly plan (v10.8.2, B6)."""
    if not plans:
        return False
    plan = _num(plans.get(symbol))
    return plan > 0 and abs(amount - plan) <= PLAN_TOLERANCE_EUR


def _sell(prev: dict, shares_sold: float, prices: dict) -> dict:
    price = _num(prices.get(prev["symbol"]))
    proceeds = shares_sold * price if price > 0 else 0.0
    gain = (proceeds - shares_sold * prev["entry"]) if price > 0 else 0.0
    return {"kind": "Sold", "symbol": prev["symbol"], "shares": round(shares_sold, 6),
            "amount_eur": round(proceeds, 2), "realized_eur": round(gain, 2),
            "estimate": True}


def diff_tables(old_rows: list[dict] | None, new_rows: list[dict] | None,
                prices: dict | None = None, plans: dict | None = None) -> dict:
    """Diff the new table against the last saved one. Pure; no I/O.

    Returns ``{changes: [...], summary: {...}}`` where each change is one of
    ``Bought`` / ``Sold`` / ``Market move``. A buy that matches the saved plan
    (``plans``: {symbol: EUR/month}) carries ``savings_plan: True``.
    """
    prices = prices or {}
    old = _index(old_rows)
    new = _index(new_rows)
    changes: list[dict] = []
    bought = sold = realized = 0.0

    for sym, p in new.items():
        prev = old.get(sym)
        if prev is None:
            if p["invested"] != 0 or p["shares"] > 0:
                changes.append({"kind": "Bought", "symbol": sym,
                                "shares": round(p["shares"], 6),
                                "amount_eur": round(p["invested"], 2),
                                "savings_plan": _matches_plan(
                                    sym, p["invested"], plans)})
                bought += p["invested"]
            continue
        d_shares = p["shares"] - prev["shares"]
        if _shares_unchanged(prev, d_shares):
            # Same shares: a value/profit move is a market move (no record).
            if abs(p["value"] - prev["value"]) >= TOL_INVESTED_EUR:
                changes.append({"kind": "Market move", "symbol": sym})
            continue
        if d_shares > 0:
            amount = p["invested"] - prev["invested"]
            changes.append({"kind": "Bought", "symbol": sym,
                            "shares": round(d_shares, 6),
                            "amount_eur": round(amount, 2),
                            "savings_plan": _matches_plan(sym, amount, plans)})
            bought += amount
        else:
            change = _sell(prev, -d_shares, prices)
            changes.append(change)
            sold += change["amount_eur"]
            realized += change["realized_eur"]

    for sym, prev in old.items():
        if sym in new:
            continue
        if prev["invested"] != 0 or prev["shares"] > 0:
            change = _sell(prev, prev["shares"], prices)
            changes.append(change)
            sold += change["amount_eur"]
            realized += change["realized_eur"]

    return {
        "changes": changes,
        "summary": {
            "bought_eur": round(bought, 2),
            "sold_eur": round(sold, 2),
            "realized_eur": round(realized, 2),
            "changed": bool(changes),
        },
    }
