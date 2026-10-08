"""test_v10_8_2_diff.py — the Review diff (v10.8.2, section 3).

Saving changes a table; the diff records Bought / Sold / Market move per ticker,
derives shares from invested / Einstandskurs (never a price lookup), and treats
sub-tolerance changes as unchanged.
"""
from __future__ import annotations

from quant.engine.diff import diff_tables, position


def _row(symbol, entry, value, profit):
    return {"Symbol": symbol, "Avg_Entry_Price": entry,
            "Current_Value_EUR": value, "Broker_PnL_EUR": profit}


def test_position_derives_shares_and_invested():
    p = position(_row("EUNL.DE", 95.5, 287.58, 9.58))
    assert p["invested"] == 278.0            # 287.58 - 9.58
    assert p["shares"] == 278.0 / 95.5       # invested / Einstandskurs


def test_empty_old_to_new_is_bought():
    out = diff_tables([], [_row("AMZN", 150.0, 160.0, 10.0)])
    assert out["changes"][0]["kind"] == "Bought"
    assert out["summary"]["bought_eur"] == 150.0   # invested = 160 - 10


def test_removed_row_is_sold_with_estimated_gain():
    old = [_row("AMZN", 150.0, 160.0, 10.0)]       # 1 share, invested 150
    out = diff_tables(old, [], prices={"AMZN": 180.0})
    change = out["changes"][0]
    assert change["kind"] == "Sold"
    assert change["amount_eur"] == 180.0           # 1 * 180
    assert change["realized_eur"] == 30.0          # 180 - 150
    assert change["estimate"] is True
    assert out["summary"]["realized_eur"] == 30.0


def test_more_shares_is_bought_amount_is_increase_in_invested():
    old = [_row("AMZN", 100.0, 100.0, 0.0)]        # 1 share, invested 100
    new = [_row("AMZN", 100.0, 200.0, 0.0)]        # 2 shares, invested 200
    out = diff_tables(old, new, prices={"AMZN": 100.0})
    change = out["changes"][0]
    assert change["kind"] == "Bought"
    assert change["amount_eur"] == 100.0
    assert out["summary"]["bought_eur"] == 100.0


def test_same_shares_different_value_is_market_move_no_record():
    old = [_row("AMZN", 100.0, 100.0, 0.0)]
    new = [_row("AMZN", 100.0, 120.0, 20.0)]       # 1 share, value up, no flow
    out = diff_tables(old, new)
    assert out["changes"][0]["kind"] == "Market move"
    assert out["summary"]["bought_eur"] == 0.0
    assert out["summary"]["sold_eur"] == 0.0


def test_sub_tolerance_change_is_unchanged():
    old = [_row("AMZN", 100.0, 100.0, 0.0)]
    # +0.4 EUR invested: below the 1 EUR tolerance -> no change recorded.
    new = [_row("AMZN", 100.0, 100.4, 0.0)]
    out = diff_tables(old, new)
    assert out["changes"] == []
    assert out["summary"]["changed"] is False


def test_no_changes_summary():
    rows = [_row("AMZN", 100.0, 100.0, 0.0)]
    out = diff_tables(rows, rows)
    assert out["summary"] == {"bought_eur": 0.0, "sold_eur": 0.0,
                              "realized_eur": 0.0, "changed": False}


# ── v10.8.2 (B6): a buy matching the saved plan is labeled a savings-plan buy ──

def test_buy_matching_plan_is_labeled_savings_plan():
    old = [_row("AMZN", 100.0, 100.0, 0.0)]        # 1 share, invested 100
    new = [_row("AMZN", 100.0, 200.0, 0.0)]        # 2 shares, invested 200
    out = diff_tables(old, new, plans={"AMZN": 100.0})
    assert out["changes"][0]["savings_plan"] is True


def test_buy_not_matching_plan_is_not_labeled():
    old = [_row("AMZN", 100.0, 100.0, 0.0)]
    new = [_row("AMZN", 100.0, 200.0, 0.0)]
    out = diff_tables(old, new, plans={"AMZN": 40.0})
    assert out["changes"][0]["savings_plan"] is False


def test_buy_without_plan_is_not_labeled():
    out = diff_tables([], [_row("AMZN", 150.0, 160.0, 10.0)])
    assert out["changes"][0]["savings_plan"] is False
