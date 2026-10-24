"""store_readers.py — independent readers of the valuation store for invariant tests.

The product reads the store through ``quant.reporting.artifacts`` and
``quant.engine.mtm``; the invariant tests check those against these plain SQL
reads, so a bug in a product reader cannot hide behind itself.
"""
from __future__ import annotations

from datetime import date


def _one(sql: str, params: list | None = None):
    from quant.data.database import read_only_connection

    with read_only_connection() as conn:
        row = conn.execute(sql, params or []).fetchone()
    return row[0] if row else None


def portfolio_total(as_of: date) -> float:
    """The stored invested total for exactly ``as_of`` (0.0 when absent)."""
    value = _one("SELECT value_eur FROM portfolio_total WHERE date = ?", [as_of])
    return round(float(value or 0.0), 2)


def latest_market_date() -> date | None:
    """The newest date in ``market_history``."""
    value = _one("SELECT MAX(Date) FROM market_history")
    if value is None:
        return None
    return value if isinstance(value, date) else date.fromisoformat(str(value)[:10])
