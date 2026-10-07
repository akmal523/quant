"""test_v10_8_0_ledger.py — one transactional ledger write (v10.8.0, 2.4).

A recorded transaction writes both ledgers in one transaction; a sell computes
its FIFO realized gain; the fee default is zero for a savings plan.
"""
from __future__ import annotations

import pytest

from quant.engine.ledger import default_fee, fifo_realized_gain, record_transaction


def test_default_fee_savings_plan_is_zero():
    assert default_fee("buy", savings_plan=True) == 0.0
    assert default_fee("buy") > 0.0
    assert default_fee("dividend") == 0.0


def test_fifo_realized_gain_consumes_lots():
    lots = [{"shares": 10.0, "price_eur": 100.0}]
    assert fifo_realized_gain(lots, 10.0, 150.0) == pytest.approx(500.0)


def test_fifo_seeded_by_entry_price():
    # No ledger lots -> the broker entry price prices the whole sell.
    assert fifo_realized_gain([], 5.0, 150.0, entry_price=100.0) == pytest.approx(250.0)


def test_record_transaction_writes_both_ledgers():
    from quant.data import database

    database.init_db()
    with database.write_connection() as conn:
        conn.execute("DELETE FROM flows WHERE symbol = 'TESTX'")
        conn.execute("DELETE FROM trades WHERE symbol = 'TESTX'")
    res = record_transaction(
        when="2026-10-01", action="buy", symbol="TESTX", amount_eur=200.0,
        shares=2.0, price_eur=100.0)
    assert res["fee_eur"] > 0
    with database.write_connection() as conn:
        f = conn.execute(
            "SELECT COUNT(*) FROM flows WHERE symbol = 'TESTX'").fetchone()[0]
        t = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE symbol = 'TESTX'").fetchone()[0]
    assert f == 1 and t == 1


def test_sell_computes_fifo_gain():
    from quant.data import database

    database.init_db()
    with database.write_connection() as conn:
        conn.execute("DELETE FROM flows WHERE symbol = 'TESTY'")
        conn.execute("DELETE FROM trades WHERE symbol = 'TESTY'")
    # A far-future year keeps this gain out of the current-year tax tests.
    record_transaction(when="2099-10-01", action="buy", symbol="TESTY",
                       amount_eur=1000.0, shares=10.0, price_eur=100.0)
    res = record_transaction(when="2099-10-05", action="sell", symbol="TESTY",
                             amount_eur=1500.0, shares=10.0, price_eur=150.0)
    assert res["realized_pnl_eur"] == pytest.approx(500.0)
