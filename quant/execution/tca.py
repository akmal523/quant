"""
tca.py — Transaction Cost Analysis / Implementation Shortfall (v10.4.0, Phase 2).

Intent: a backtest is a theory; execution is reality. This module measures the
gap between the price when a signal was generated and the actual fill price from
Trade Republic, in basis points. It also provides an empirical slippage estimate
so the fee hurdle can be sized against realistic execution cost.

Invariants:
  - implementation_shortfall is pure: (signal, fill, side) -> bps.
  - record_trade persists one row to trade_log with computed slippage.
  - slippage_summary reads trade_log and returns aggregate stats.
  - Pure computation separated from I/O.

Dependencies: database.get_connection, config.
"""
from __future__ import annotations

import datetime as dt



def implementation_shortfall(signal_price: float, fill_price: float, side: str) -> float:
    """Implementation Shortfall in basis points.

    Intent: quantify slippage cost. For a BUY, paying above the signal price is a
    cost (positive bps). For a SELL, filling below the signal price is a cost.
    Invariants: returns bps (positive = worse execution); pure function (no I/O).
    """
    if not signal_price or signal_price <= 0 or fill_price is None:
        return 0.0
    side = (side or "").upper()
    if side == "SELL":
        diff = signal_price - fill_price
    else:  # BUY (default)
        diff = fill_price - signal_price
    return float(diff / signal_price * 10000.0)


def estimate_slippage_bps(
    participation: float,
    spread_bps: float = 5.0,
    impact_coeff: float = 10.0,
) -> float:
    """Empirical slippage model: half-spread + square-root market impact.

    Intent: estimate expected slippage before a trade. Impact grows with the
    square root of participation (fraction of ADV), a standard market-impact form.
    Invariants: returns >= 0 bps; pure function (no I/O).
    """
    p = max(0.0, min(participation, 1.0))
    return float(spread_bps / 2.0 + impact_coeff * (p ** 0.5) * 100.0)


def record_trade(
    symbol: str,
    side: str,
    signal_price: float,
    fill_price: float,
    fee_eur: float = 0.0,
    signal_ts: str | None = None,
    fill_ts: str | None = None,
) -> float:
    """Persist a trade to trade_log and return its slippage in bps.

    Intent: build the empirical dataset for TCA. The signal price/time is the
    price when the signal fired; the fill price/time is the broker's actual fill.
    Invariants: returns slippage_bps; best-effort (never raises).
    Dependencies: database.get_connection.
    """
    slip = implementation_shortfall(signal_price, fill_price, side)
    try:
        from quant.data.database import get_connection
        conn = get_connection()
        conn.execute(
            """INSERT INTO trade_log
               (symbol, side, signal_price, signal_ts, fill_price, fill_ts,
                slippage_bps, fee_eur)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                symbol, (side or "").upper(), signal_price,
                signal_ts or dt.datetime.now().isoformat(),
                fill_price, fill_ts or dt.datetime.now().isoformat(),
                slip, fee_eur,
            ],
        )
    except Exception:
        pass
    return slip


def slippage_summary() -> dict:
    """Aggregate TCA stats from trade_log.

    Intent: report average/median slippage and total fees so execution quality is
    observable. Invariants: returns a dict; empty-safe (never raises).
    """
    empty = {"trades": 0, "avg_slippage_bps": 0.0, "median_slippage_bps": 0.0,
             "total_fee_eur": 0.0}
    try:
        from quant.data.database import get_connection
        conn = get_connection()
        rows = conn.execute(
            "SELECT slippage_bps, fee_eur FROM trade_log"
        ).fetchall()
    except Exception:
        return empty
    if not rows:
        return empty

    slips = sorted(float(r[0]) for r in rows if r[0] is not None)
    fees = sum(float(r[1] or 0.0) for r in rows)
    n = len(slips)
    if n == 0:
        return empty
    median = slips[n // 2] if n % 2 else (slips[n // 2 - 1] + slips[n // 2]) / 2.0
    return {
        "trades": n,
        "avg_slippage_bps": round(sum(slips) / n, 2),
        "median_slippage_bps": round(median, 2),
        "total_fee_eur": round(fees, 2),
    }
