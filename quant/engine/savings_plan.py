"""
savings_plan.py — the "How much do you want to invest?" block (v10.8.2, B2).

Intent: the user enters one amount and a cadence; the app proposes savings-plan
rates (monthly) or one-off orders using the existing allocation rules. It is a
suggestion only: the app never controls the broker's plans and never stores the
"Once" amount.

Invariants:
  - Reuses :func:`quant.engine.allocator.allocate` (one allocation function).
  - Monthly: rows are Name (TICKER) | Plan now | Suggested | Change; the
    suggested rates sum to the amount minus any unallocated part.
  - "Once" returns whole orders at or above the minimum order size; never a sell.
  - Pure: no I/O.
"""
from __future__ import annotations

EQUAL_TOLERANCE_EUR = 5.0


def propose_plans(amount_eur: float, holdings: list[dict] | None = None,
                  current_plans: dict | None = None, regime: str | None = None,
                  candidates: list[dict] | None = None,
                  tolerance: float = EQUAL_TOLERANCE_EUR) -> dict:
    """Propose monthly savings-plan rates for ``amount_eur``.

    Returns ``{rows, not_allocated, already_fit, total}`` where each row is
    ``{symbol, name, current, suggested, change, reason}``.
    """
    from quant.engine.allocator import allocate

    amount = float(amount_eur or 0.0)
    current = {str(k): float(v or 0.0) for k, v in (current_plans or {}).items()}
    suggested: dict[str, float] = {}
    names: dict[str, str] = {}
    not_allocated = 0.0
    for leg in allocate(amount, holdings, regime, candidates):
        if leg.get("kind") == "cash" or not leg.get("symbol"):
            not_allocated += float(leg.get("amount_eur") or 0.0)
            continue
        sym = str(leg["symbol"])
        suggested[sym] = suggested.get(sym, 0.0) + float(leg.get("amount_eur") or 0.0)
        names[sym] = leg.get("name") or names.get(sym, sym)

    # Every holding gets a row (an overweight one shows 0 suggested).
    held: dict[str, str] = {}
    for h in holdings or []:
        sym = str(h.get("symbol") or "")
        if sym:
            held[sym] = h.get("name") or sym
    rows = []
    for sym in sorted(set(suggested) | set(current) | set(held)):
        cur = current.get(sym, 0.0)
        sug = round(suggested.get(sym, 0.0), 2)
        rows.append({"symbol": sym, "name": names.get(sym) or held.get(sym, sym),
                     "current": cur, "suggested": sug,
                     "change": round(sug - cur, 2)})
    rows.sort(key=lambda r: -abs(r["change"]))
    already_fit = bool(rows) and all(abs(r["change"]) <= tolerance for r in rows)
    return {"rows": rows, "not_allocated": round(not_allocated, 2),
            "already_fit": already_fit, "total": round(sum(suggested.values()), 2)}


def propose_once(amount_eur: float, holdings: list[dict] | None = None,
                 regime: str | None = None, candidates: list[dict] | None = None,
                 min_order_eur: float = 0.0) -> list[dict]:
    """One-off purchases: whole orders at or above the minimum; never a sell.

    The amount is a suggestion and is not stored.
    """
    from quant.engine.allocator import allocate

    out: list[dict] = []
    for leg in allocate(float(amount_eur or 0.0), holdings, regime, candidates):
        if leg.get("kind") == "cash" or not leg.get("symbol"):
            continue
        amt = float(leg.get("amount_eur") or 0.0)
        if amt >= float(min_order_eur or 0.0) and amt > 0:
            out.append({"symbol": str(leg["symbol"]),
                        "name": leg.get("name") or leg["symbol"],
                        "amount_eur": amt})
    return out
