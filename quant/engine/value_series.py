"""value_series.py — the daily invested-value series (v10.8.0, Phase 1).

Intent: the Overview chart must show the invested value over time without
requiring a completed review. The series is built from append-only position
snapshots, recorded flows, and daily closes in ``market_history`` (converted to
EUR). Cash is never part of the series.

Invariants:
  - Pure: no I/O, no database writes.
  - A later snapshot is truth from its date forward; the difference from the
    flow-derived shares is a neutral correction (never profit or loss).
  - A missing close on a day carries the previous close.
"""
from __future__ import annotations

from datetime import date, timedelta


def _as_date(value) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _days(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _close_eur(symbol, day, closes, fx, last_close) -> float | None:
    """The EUR close on a day, carrying the previous close when missing."""
    native = closes.get((symbol, day.isoformat()))
    if native is None:
        native = last_close.get(symbol)
    else:
        last_close[symbol] = native
    if native is None:
        return None
    rate = fx.get(symbol, 1.0) if isinstance(fx, dict) else 1.0
    return float(native) * float(rate)


def build_value_series(snapshots, flows, closes, fx) -> list[dict]:
    """Build the daily invested-value series. Pure; no I/O.

    Args:
        snapshots: ``[{date, symbol, shares}]`` (append-only, any order).
        flows: ``[{date, type, amount_eur, symbol}]`` (buy/sell/dividend).
        closes: ``{(symbol, date_iso): close_native}``.
        fx: ``{symbol: rate_to_eur}`` (the EUR multiplier).

    Returns ``[{date, value_eur}]`` oldest first. Empty when there is no data.
    """
    snaps = sorted(
        ({"date": _as_date(s["date"]), "symbol": str(s["symbol"]),
          "shares": float(s["shares"] or 0.0)} for s in snapshots),
        key=lambda s: s["date"])
    flws = sorted(
        ({"date": _as_date(f["date"]), "type": str(f["type"]),
          "amount_eur": float(f["amount_eur"] or 0.0),
          "symbol": str(f["symbol"])} for f in flows if f.get("symbol")),
        key=lambda f: f["date"])
    if not snaps and not flws:
        return []

    dates = [s["date"] for s in snaps] + [f["date"] for f in flws]
    start, end = min(dates), max(dates)

    by_sym: dict[str, list] = {}
    for s in snaps:
        by_sym.setdefault(s["symbol"], []).append(s)

    out: list[dict] = []
    last_close: dict[str, float] = {}
    for day in _days(start, end):
        total = 0.0
        for sym, rows in by_sym.items():
            base = None
            for s in rows:
                if s["date"] <= day:
                    base = s
                else:
                    break
            if base is None:
                continue
            shares = base["shares"]
            for f in flws:
                if (f["symbol"] != sym or f["date"] <= base["date"]
                        or f["date"] > day):
                    continue
                price = _close_eur(sym, f["date"], closes, fx, last_close)
                if price and price > 0:
                    if f["type"] == "buy":
                        shares += f["amount_eur"] / price
                    elif f["type"] == "sell":
                        shares -= f["amount_eur"] / price
            price = _close_eur(sym, day, closes, fx, last_close)
            if price is not None:
                total += shares * price
        out.append({"date": day, "value_eur": round(total, 2)})
    return out
