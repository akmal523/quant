"""
history.py — Portfolio value history (v10.5.1, spec 5.1).

Intent: one row per review feeds the Today value chart. Written by the review
step (writer path); read by the UI through a short-lived read-only connection
(spec 4.1). Never raises on read.

Invariants:
  - record_review appends exactly one row per call.
  - load_history returns a DataFrame ordered oldest -> newest.
  - Reads use read_only_connection so they never hold the write lock.

Dependencies: quant.data.database, pandas, datetime.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd


def record_review(
    value_eur: float,
    invested_eur: float,
    cash_eur: float | None,
    pnl_eur: float,
    review_ts: datetime | None = None,
    conn=None,
) -> None:
    """Append one review row. Uses the writer connection if supplied."""
    ts = review_ts or datetime.now()
    if conn is None:
        from quant.data.database import get_connection

        conn = get_connection()
    conn.execute(
        "INSERT INTO portfolio_history "
        "(review_ts, value_eur, invested_eur, cash_eur, pnl_eur) VALUES (?, ?, ?, ?, ?)",
        [ts, float(value_eur), float(invested_eur),
         None if cash_eur is None else float(cash_eur), float(pnl_eur)],
    )


def load_history(limit: int | None = None) -> pd.DataFrame:
    """Load review history oldest -> newest via a read-only connection."""
    from quant.data.database import read_only_connection

    try:
        with read_only_connection() as conn:
            df = conn.execute(
                "SELECT review_ts, value_eur, invested_eur, cash_eur, pnl_eur "
                "FROM portfolio_history ORDER BY review_ts ASC"
            ).df()
    except Exception:  # noqa: BLE001
        return pd.DataFrame(
            columns=["review_ts", "value_eur", "invested_eur", "cash_eur", "pnl_eur"]
        )
    if limit is not None and len(df) > limit:
        df = df.tail(limit).reset_index(drop=True)
    return df
