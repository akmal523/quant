import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# test_db.py
# DB isolation is provided by tests/conftest.py (ephemeral DuckDB). Never
# delete the production store from a test.
from quant.data.database import get_connection, init_db


def test_db_init():
    # Act
    init_db()
    conn = get_connection()

    # Test Fundamentals Schema
    conn.execute("INSERT INTO fundamentals (symbol, pe) VALUES ('AAPL', 25.5)")
    res_fund = conn.execute("SELECT pe FROM fundamentals WHERE symbol = 'AAPL'").fetchone()
    assert res_fund[0] == 25.5, "Fundamentals I/O failed."

    # News evidence schema (written by refresh, read by scoring and the drawer)
    conn.execute("INSERT INTO nlp_evidence (symbol, source, title, score) "
                 "VALUES ('AAPL', 'news', 'db test headline', 0.5)")
    res_nlp = conn.execute("SELECT score FROM nlp_evidence WHERE title = "
                           "'db test headline'").fetchone()
    assert res_nlp[0] == 0.5, "NLP evidence I/O failed."
    conn.execute("DELETE FROM nlp_evidence WHERE title = 'db test headline'")
