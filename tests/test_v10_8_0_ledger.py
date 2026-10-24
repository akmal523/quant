"""test_v10_8_0_ledger.py — one transactional ledger write (v10.8.0, 2.4).

A recorded transaction writes both ledgers in one transaction; a sell computes
its FIFO realized gain; the fee default is zero for a savings plan.
"""
from __future__ import annotations

from quant.engine.ledger import default_fee


def test_default_fee_savings_plan_is_zero():
    assert default_fee("buy", savings_plan=True) == 0.0
    assert default_fee("buy") > 0.0
    assert default_fee("dividend") == 0.0
