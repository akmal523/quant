"""value_series.py — the daily invested-value series (v10.8.0, fixed v10.8.2).

Intent: the Portfolio chart must show the invested value over time without
requiring a completed review, and must never draw a flat zero line. The series
runs from the first saved table to the LATEST available price date, one point per
calendar day. A day on which any held symbol has never yet had a price is
skipped, not zeroed; once a symbol has a first price, its last close is carried
forward. Cash is never part of the series.

Invariants:
  - Pure: no I/O, no database writes.
  - The series ends at the latest price date, so new prices extend it.
  - A missing close carries the previous close forward.
  - All values are EUR (native close * the symbol's EUR rate).
"""
from __future__ import annotations

import bisect
from datetime import date, timedelta
from typing import Any


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _days(start: date, end: date) -> Any:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _close_series(closes: dict, symbols: set[str]) -> dict[str, list[tuple[date, float]]]:
    """{symbol: [(date, native_close), ...] sorted} for held symbols only."""
    out: dict[str, list[tuple[date, float]]] = {}
    for (sym, day), value in (closes or {}).items():
        if sym not in symbols or value is None:
            continue
        try:
            d = _as_date(day)
        except ValueError:
            continue
        out.setdefault(str(sym), []).append((d, float(value)))
    for sym in out:
        out[sym].sort(key=lambda t: t[0])
    return out


def _price_on_or_before(series: list[tuple[date, float]], day: date) -> float | None:
    """The close on ``day`` or the most recent close before it (None if none yet)."""
    if not series:
        return None
    idx = bisect.bisect_right(series, (day, float("inf"))) - 1
    return series[idx][1] if idx >= 0 else None


def _rate(fx: dict, symbol: str) -> float:
    if not isinstance(fx, dict):
        return 1.0
    value = fx.get(symbol, 1.0)
    return float(value) if isinstance(value, int | float) else 1.0


def build_value_series(snapshots: list[dict], flows: list[dict],
                       closes: dict, fx: dict) -> list[dict]:
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

    symbols = {s["symbol"] for s in snaps}
    by_sym: dict[str, list] = {}
    for s in snaps:
        by_sym.setdefault(s["symbol"], []).append(s)

    close_by_sym = _close_series(closes, symbols)

    # Start at the first saved table; end at the latest price date (so new
    # prices extend the series). With no prices, there is no series.
    start = min(s["date"] for s in snaps) if snaps else min(f["date"] for f in flws)
    latest = [s[-1][0] for s in close_by_sym.values() if s]
    if not latest:
        return []
    end = max([start] + latest)

    out: list[dict] = []
    for day in _days(start, end):
        total = 0.0
        day_ok = True
        for sym, rows in by_sym.items():
            base = None
            for s in rows:
                if s["date"] <= day:
                    base = s
                else:
                    break
            if base is None:
                continue  # not held yet
            shares = base["shares"]
            for f in flws:
                if (f["symbol"] != sym or f["date"] <= base["date"]
                        or f["date"] > day):
                    continue
                price = _price_on_or_before(close_by_sym.get(sym, []), f["date"])
                if price and price > 0:
                    if f["type"] == "buy":
                        shares += f["amount_eur"] / price
                    elif f["type"] == "sell":
                        shares -= f["amount_eur"] / price
            native = _price_on_or_before(close_by_sym.get(sym, []), day)
            if native is None:
                day_ok = False  # a held symbol has never had a price yet
                break
            total += shares * native * _rate(fx, sym)
        if not day_ok:
            continue
        out.append({"date": day, "value_eur": round(total, 2)})
    return out
