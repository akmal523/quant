"""test_v10_8_4_phase3.py — T2/T3/T8 page-level rules (v10.8.4, Phase 3).

T8: the class plan for an amount sums to that amount (rows + unallocated).
T3: the day delta is the total change from the store (the frozen cash_eur column is
    never read; v10.8.7, R-C1).
T9 (partial): one verb, one ticker; one class-word mapping.
"""
from __future__ import annotations

from datetime import date

from quant.engine.decisions import format_item_line
from quant.engine.invest_plan import propose_by_class
from quant.ui import copy as C


def _holdings():
    return [
        {"symbol": "EUNL.DE", "name": "MSCI World", "tier": "FORTRESS",
         "value_eur": 300.0},
        {"symbol": "SXRV.DE", "name": "Nasdaq 100", "tier": "ALPHA",
         "value_eur": 400.0},
    ]


def test_invest_plan_sums_to_the_amount():
    plan = propose_by_class(200.0, _holdings(), "balanced")
    assert round(plan["total"] + plan["not_allocated"], 2) == 200.0
    # The three class rows are always present and sum to the invested total.
    assert {r["tier"] for r in plan["classes"]} == {
        "FORTRESS", "ALPHA", "SPECULATIVE"}


def test_invest_plan_never_exceeds_the_amount():
    plan = propose_by_class(500.0, _holdings(), "conservative")
    assert plan["total"] <= 500.0
    assert all(o["amount_eur"] > 0 for o in plan["orders"])


def test_day_delta_uses_the_store_total():
    from quant.data.database import write_connection
    from quant.engine import mtm

    with write_connection() as conn:
        conn.execute("DELETE FROM portfolio_total")
        conn.execute("INSERT INTO portfolio_total (date, value_eur, cash_eur) "
                     "VALUES (?, ?, ?), (?, ?, ?)",
                     [date(2026, 6, 1), 1000.0, 100.0,
                      date(2026, 6, 2), 1050.0, 100.0])
    eur, pct, when = mtm.day_delta()
    assert round(eur, 2) == 50.0
    assert when == date(2026, 6, 2)


def test_sell_line_is_one_verb_one_ticker():
    line = format_item_line({"verb": "sell_part", "label": "Nasdaq 100 (SXRV.DE)",
                             "amount_eur": 65})
    assert line == "Sell ~65 EUR of Nasdaq 100 (SXRV.DE)"
    assert "about" not in line


def test_one_class_word_mapping():
    # R6/A1: one mapping module, no synonym drift.
    assert C.CLASS_LABELS["FORTRESS"] == "Long-term"
    assert C.CLASS_LABELS["ALPHA"] == "Active"
    assert C.CLASS_LABELS["SPECULATIVE"] == "Speculative"
