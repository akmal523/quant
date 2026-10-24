"""test_decision_rules.py — the one rule set behind the ONE list (v10.8.10).

quant/engine/action_taxonomy.py: rebalance (Active/Speculative), plan change
(Long-term), threats, swaps; sizing laws; no target -> no drift advice.
"""
from __future__ import annotations

from quant.engine import action_taxonomy as T


def _h(sym, tier, cur, tgt, value, **kw):
    return {"symbol": sym, "name": sym, "tier": tier, "current_weight": cur,
            "target_weight": tgt, "value_eur": value, **kw}


def _book(*rows):
    return list(rows)


# ── Rebalance ─────────────────────────────────────────────────────────────────

def test_overweight_active_gets_a_sized_sell_today():
    book = _book(_h("ACT", "ALPHA", 0.60, 0.40, 600.0), _h("LT", "FORTRESS", 0.40, 0.60, 400.0))
    sells = [i for i in T.rebalance_items(book) if i["verb"] == T.VERB_SELL]
    assert len(sells) == 1 and sells[0]["horizon"] == T.HORIZON_TODAY
    assert sells[0]["amount_eur"] == 200.0          # 20pp of 1000, 5 EUR steps


def test_underweight_active_gets_a_buy_this_month_but_not_in_a_bear_market():
    book = _book(_h("ACT", "ALPHA", 0.20, 0.40, 200.0), _h("LT", "FORTRESS", 0.80, 0.60, 800.0))
    buys = [i for i in T.rebalance_items(book) if i["verb"] == T.VERB_BUY]
    assert buys and buys[0]["amount_eur"] == 200.0 and buys[0]["horizon"] == T.HORIZON_MONTH
    assert not [i for i in T.rebalance_items(book, regime="bear") if i["verb"] == T.VERB_BUY]


def test_no_target_means_no_drift_advice():
    book = _book(_h("NEW", "ALPHA", 0.90, None, 900.0), _h("LT", "FORTRESS", 0.10, 0.10, 100.0))
    assert T.build(book) == []


def test_sizing_laws_small_positions_and_small_amounts_are_left_alone():
    book = _book(_h("TINY", "ALPHA", 0.90, 0.10, 90.0), _h("LT", "FORTRESS", 0.10, 0.90, 10.0))
    assert not [i for i in T.rebalance_items(book) if i["verb"] == T.VERB_SELL]  # < 100 EUR


# ── Long-term ─────────────────────────────────────────────────────────────────

def test_long_term_is_never_sold_only_its_plan_changes():
    under = _book(_h("LT", "FORTRESS", 0.30, 0.60, 300.0), _h("ACT", "ALPHA", 0.70, 0.40, 700.0))
    over = _book(_h("LT", "FORTRESS", 0.85, 0.60, 850.0), _h("ACT", "ALPHA", 0.15, 0.40, 150.0))
    up = [i for i in T.build(under) if i["symbol"] == "LT"]
    down = [i for i in T.build(over) if i["symbol"] == "LT"]
    assert [i["verb"] for i in up] == [T.VERB_PLAN_UP]
    assert [i["verb"] for i in down] == [T.VERB_PLAN_DOWN]


# ── Threats ───────────────────────────────────────────────────────────────────

def test_price_threats_from_what_was_paid_and_the_last_week():
    book = _book(
        _h("SPC", "SPECULATIVE", 0.01, None, 10.0, pnl_pct=-0.55),
        _h("ACT", "ALPHA", 0.50, None, 500.0, pnl_pct=-0.25),
        _h("CRS", "ALPHA", 0.49, None, 490.0, pnl_pct=0.05, change_7d=-0.18),
        _h("LT", "FORTRESS", 0.0, None, 0.0, pnl_pct=-0.40))
    reasons = {i["symbol"]: i["reason"] for i in T.threat_items(book)}
    assert "stop-loss" in reasons["SPC"]
    assert "from what you paid" in reasons["ACT"]
    assert "7 days" in reasons["CRS"]
    assert "LT" not in reasons                    # Long-term drawdown is no sell signal


def test_speculative_cap_and_bear_market_are_market_level_threats():
    book = _book(_h("SPC", "SPECULATIVE", 0.10, None, 100.0), _h("ACT", "ALPHA", 0.90, None, 900.0))
    items = T.threat_items(book, regime={"label": "falling"})
    labels = {i["label"] for i in items if i["symbol"] is None}
    assert labels == {"the Speculative part", "the Active part"}


def test_score_floor_needs_a_trusted_score_and_skips_long_term():
    book = _book(_h("LT", "FORTRESS", 0.4, None, 400.0), _h("ACT", "ALPHA", 0.3, None, 300.0),
                 _h("THIN", "ALPHA", 0.3, None, 300.0))
    scores = {"LT": {"composite": 20, "confidence": "high"},
              "ACT": {"composite": 20, "confidence": "medium"},
              "THIN": {"composite": 20, "confidence": "low"}}
    assert {i["symbol"] for i in T.threat_items(book, scores)} == {"ACT"}


# ── Swaps ─────────────────────────────────────────────────────────────────────

def test_swap_needs_a_scored_active_holding_and_replaces_its_rebalance():
    book = _book(_h("ACT", "ALPHA", 0.60, 0.40, 600.0), _h("LT", "FORTRESS", 0.40, 0.60, 400.0),
                 _h("UNS", "SPECULATIVE", 0.0, None, 0.0))
    ideas = [{"symbol": "IDEA", "name": "Idea", "score": 90.0}]
    items = T.build(book, scores={"ACT": {"composite": 50}, "LT": {"composite": 10}},
                    ideas=ideas)
    act = [i["verb"] for i in items if i["symbol"] == "ACT"]
    assert act == [T.VERB_SWAP]                     # no separate sell
    assert not [i for i in items if i["symbol"] in ("LT", "UNS") and i["verb"] == T.VERB_SWAP]
    swap = next(i for i in items if i["verb"] == T.VERB_SWAP)
    assert swap["target_symbol"] == "IDEA"


def test_every_item_is_complete_and_today_comes_first():
    book = _book(_h("ACT", "ALPHA", 0.60, 0.40, 600.0, pnl_pct=-0.3),
                 _h("LT", "FORTRESS", 0.40, 0.60, 400.0))
    items = T.build(book)
    for it in items:
        assert {"verb", "symbol", "label", "amount_eur", "reason", "horizon", "class",
                "evidence"} <= set(it)
    horizons = [i["horizon"] for i in items]
    assert horizons == sorted(horizons, key=lambda h: 0 if h == T.HORIZON_TODAY else 1)
