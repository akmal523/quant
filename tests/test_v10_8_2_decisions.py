"""test_v10_8_2_decisions.py — the ONE list item format (v10.8.2, section 6).

Every item line is "Verb Name (TICKER) amount", never a bare reason, and a
position that needs nothing is not an item at all.
"""
from __future__ import annotations

from quant.engine.decisions import (
    actionable_items,
    build_decision_list,
    format_item_line,
)
from quant.ui import copy as C


def _item(verb, label, amount=None):
    return {"group": C.DECISION_GROUP_RECOMMENDED, "verb": verb,
            "label": label, "amount_eur": amount}


def test_buy_line_has_verb_name_ticker_amount():
    line = format_item_line(
        _item("buy", "iShares Core MSCI World (EUNL)", 140))
    assert line == "Buy 140 EUR of iShares Core MSCI World (EUNL)"


def test_sell_line_has_about_and_amount():
    line = format_item_line(
        _item("sell_part", "iShares NASDAQ 100 (SXRV)", 75))
    assert line == "Sell about 75 EUR of iShares NASDAQ 100 (SXRV)"


def test_amount_uses_thousands_separator():
    line = format_item_line(_item("buy", "Broad ETF (EUNL)", 1234))
    assert "1,234 EUR" in line


def test_no_bare_reason_line_is_always_label_plus_verb():
    # A change_savings_plan item has no amount but still names the instrument.
    line = format_item_line(_item("change_savings_plan", "Alpha (AAA)"))
    assert line == "Change the savings plan for Alpha (AAA)"


def test_actionable_items_exclude_optional():
    decisions = build_decision_list(
        [{"kind": "buy", "symbol": "AAA", "company_name": "Alpha", "eur": 10.0,
          "why": "under target"},
         {"kind": "keep", "symbol": "BBB", "company_name": "Beta", "eur": None,
          "why": "long-term holding"}],
        [], [])
    act = actionable_items(decisions)
    assert {d["symbol"] for d in act} == {"AAA"}
