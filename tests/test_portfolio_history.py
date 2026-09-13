"""
test_portfolio_history.py — Review history contract (v10.5.1, P3d).

Asserts: two sequential reviews produce two rows and a non-empty chart series.
"""
from __future__ import annotations

from datetime import datetime, timedelta


def test_two_reviews_produce_two_rows():
    from quant.data.database import get_connection
    from quant.portfolio.history import load_history, record_review

    conn = get_connection()
    conn.execute("DELETE FROM portfolio_history")
    t0 = datetime(2026, 9, 12, 17, 0)
    record_review(1000.0, 900.0, 100.0, 100.0, review_ts=t0, conn=conn)
    record_review(1030.0, 900.0, 90.0, 130.0, review_ts=t0 + timedelta(days=1), conn=conn)

    hist = load_history()
    assert len(hist) == 2
    assert list(hist["value_eur"]) == [1000.0, 1030.0]
    series = hist["value_eur"].dropna()
    assert not series.empty
