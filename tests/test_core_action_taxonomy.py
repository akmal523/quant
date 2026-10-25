"""test_core_action_taxonomy.py — every item of the ONE list, exactly.

quant/engine/action_taxonomy.py is in the mutation scope. Each test compares
whole items (verb, label, amount, reason text, horizon, class, evidence) and
probes each threshold on both sides, so a changed word, number or comparison
fails a test. Thresholds come from quant/config.py.
"""
from __future__ import annotations

import pytest

from quant.engine import action_taxonomy as T


def hold(sym, tier, cur, tgt, value, name=None, **kw):
    return {"symbol": sym, "name": name or sym, "tier": tier, "current_weight": cur,
            "target_weight": tgt, "value_eur": value, **kw}


def item(verb, sym, label, amount, reason, horizon, cls, evidence, **extra):
    return {"verb": verb, "symbol": sym, "label": label, "amount_eur": amount,
            "reason": reason, "horizon": horizon, "class": cls, "evidence": evidence,
            **extra}


TODAY, MONTH = "Today", "This month"


# ── helpers ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ({"symbol": "ACT", "name": "Alpha Star"}, "Alpha Star (ACT)"),
    ({"symbol": "ACT"}, "ACT"),
    ({"symbol": "ACT", "name": None}, "ACT"),
    ({"name": "Lonely"}, "Lonely"),
    ({}, ""),
    ({"symbol": None, "name": None}, ""),
])
def test_label(raw, expected):
    assert T._label(raw) == expected


def test_num_and_tier_and_score_of():
    assert T._num(None) == 0.0 and T._num("x") == 0.0
    assert T._num("x", 5.0) == 5.0 and T._num("2") == 2.0
    assert T._tier({"tier": "alpha"}) == "ALPHA" and T._tier({}) == ""
    assert T._score_of({"composite": 0, "score": 9}) == 0
    assert T._score_of({"score": 9, "structural_grade": 3}) == 9
    assert T._score_of({"structural_grade": 3}) == 3
    assert T._score_of({}) is None


# ── rebalance ─────────────────────────────────────────────────────────────────

def test_overweight_active_sell_item():
    book = [hold("LT", "FORTRESS", 0.4, 0.6, 400.0),
            hold("NEW", "ALPHA", 0.1, None, 100.0),
            hold("ACT", "ALPHA", 0.6, 0.4, 600.0, "Alpha Star")]
    assert T.rebalance_items(book) == [item(
        "sell_part", "ACT", "Alpha Star (ACT)", 220.0, "20pp above its 40% target.",
        TODAY, "ALPHA", "Holdings analysis")]


def test_underweight_buy_item_and_the_bear_market():
    book = [hold("LT", "FORTRESS", 0.6, 0.6, 600.0),
            hold("ACT", "ALPHA", 0.2, 0.4, 200.0, "Alpha Star"),
            hold("SPC", "SPECULATIVE", 0.2, 0.4, 200.0, "Space Bet")]
    buys = T.rebalance_items(book)
    assert buys == [
        item("buy", "ACT", "Alpha Star (ACT)", 200.0, "20pp below its 40% target.",
             MONTH, "ALPHA", "Holdings analysis"),
        item("buy", "SPC", "Space Bet (SPC)", 200.0, "20pp below its 40% target.",
             MONTH, "SPECULATIVE", "Holdings analysis")]
    assert [i["symbol"] for i in T.rebalance_items(book, regime="bear")] == ["SPC"]
    assert [i["symbol"] for i in T.rebalance_items(book, regime="falling")] == ["SPC"]


def test_rebalance_drift_exactly_at_the_threshold_is_left_alone():
    base = hold("LT", "FORTRESS", 0.95, 0.95, 9500.0)
    assert T.rebalance_items([base, hold("A", "ALPHA", 0.05, 0.0, 500.0)]) == []
    assert T.rebalance_items([base, hold("A", "ALPHA", 0.0, 0.05, 500.0)]) == []
    over = T.rebalance_items([base, hold("A", "ALPHA", 0.06, 0.0, 600.0)])
    under = T.rebalance_items([base, hold("A", "ALPHA", 0.0, 0.06, 0.0)])
    assert [(i["verb"], i["amount_eur"]) for i in over] == [("sell_part", 595.0)]
    assert [(i["verb"], i["amount_eur"]) for i in under] == [("buy", 570.0)]


def test_rebalance_threshold_argument():
    book = [hold("LT", "FORTRESS", 0.4, 0.6, 400.0), hold("ACT", "ALPHA", 0.6, 0.4, 600.0)]
    assert T.rebalance_items(book, threshold=0.25) == []
    assert len(T.rebalance_items(book, threshold=0.15)) == 1


def test_rebalance_sizing_laws():
    small = [hold("LT", "FORTRESS", 0.0, 0.0, 9901.0), hold("A", "ALPHA", 0.6, 0.0, 99.0)]
    assert T.rebalance_items(small) == []                     # untouchable < 100 EUR
    edge = [hold("LT", "FORTRESS", 0.0, 0.0, 900.0), hold("A", "ALPHA", 0.6, 0.0, 100.0)]
    assert [i["amount_eur"] for i in T.rebalance_items(edge)] == [95.0]  # value - 1
    tiny = [hold("LT", "FORTRESS", 0.0, 0.0, 400.0), hold("A", "ALPHA", 0.0, 0.06, 0.0)]
    assert T.rebalance_items(tiny) == []                      # 24 EUR rounds to 20 < 25
    ok = [hold("LT", "FORTRESS", 0.0, 0.0, 500.0), hold("A", "ALPHA", 0.0, 0.06, 0.0)]
    assert [i["amount_eur"] for i in T.rebalance_items(ok)] == [30.0]


def test_rebalance_empty():
    assert T.rebalance_items(None) == [] and T.rebalance_items([]) == []


# ── plan change ───────────────────────────────────────────────────────────────

def test_plan_up_and_down_items():
    up = [hold("ACT", "ALPHA", 0.7, 0.4, 700.0), hold("LT", "FORTRESS", 0.3, 0.6, 300.0,
                                                "World Star")]
    down = [hold("LT", "FORTRESS", 0.85, 0.6, 850.0, "World Star")]
    assert T.plan_change_items(up) == [item(
        "plan_up", "LT", "World Star (LT)", None, "30pp below its 60% target.",
        MONTH, "FORTRESS", "Savings plan")]
    assert T.plan_change_items(down) == [item(
        "plan_down", "LT", "World Star (LT)", None,
        "25pp above its 60% target; Long-term holdings are never sold.",
        MONTH, "FORTRESS", "Savings plan")]


def test_plan_change_gap_edges_and_skips():
    assert T.plan_change_items([hold("LT", "FORTRESS", 0.0, 0.1, 0.0)]) == []
    assert T.plan_change_items([hold("LT", "FORTRESS", 0.1, 0.0, 0.0)]) == []
    skipped_first = [hold("N", "FORTRESS", 0.0, None, 0.0), hold("LT", "FORTRESS", 0.0, 0.2, 0.0)]
    assert [i["verb"] for i in T.plan_change_items(skipped_first)] == ["plan_up"]
    assert T.plan_change_items([hold("LT", "FORTRESS", 0.0, 0.2, 0.0)], gap=0.25) == []
    assert T.plan_change_items(None) == []


# ── threats ───────────────────────────────────────────────────────────────────

# A large Active holding keeps a single Speculative position under the 2% cap.
BIG = {"symbol": "BIG", "tier": "ALPHA", "value_eur": 10000.0}


def _threat(sym, label, reason, cls, evidence="Holdings analysis"):
    return item("threat", sym, label, None, reason, TODAY, cls, evidence)


@pytest.mark.parametrize("tier, kw, reason", [
    ("SPECULATIVE", {"pnl_pct": -0.5}, "stop-loss hit (-50%)."),
    ("SPECULATIVE", {"pnl_pct": -0.49}, "down 49% from what you paid."),
    ("SPECULATIVE", {"pnl_pct": 1.0}, "take-profit hit (+100%)."),
    ("SPECULATIVE", {"pnl_pct": 0.99}, None),
    ("ALPHA", {"pnl_pct": 1.5}, None),
    ("ALPHA", {"pnl_pct": -0.2}, "down 20% from what you paid."),
    ("ALPHA", {"pnl_pct": -0.19}, None),
    ("FORTRESS", {"pnl_pct": -0.6}, None),
    ("ALPHA", {"change_7d": -0.15}, "fell 15% in the last 7 days."),
    ("FORTRESS", {"change_7d": -0.3}, "fell 30% in the last 7 days."),
    ("ALPHA", {"change_7d": -0.14}, None),
    ("ALPHA", {"change_7d": 0.1}, None),
    ("ALPHA", {"pnl_pct": -0.6, "change_7d": -0.3}, "down 60% from what you paid."),
])
def test_price_threats(tier, kw, reason):
    h = {"symbol": "X", "name": "Xeno Star", "tier": tier, "value_eur": 100.0, **kw}
    expected = [] if reason is None else [_threat("X", "Xeno Star (X)", reason, tier)]
    assert T.threat_items([h, BIG]) == expected


@pytest.mark.parametrize("tier, score, reason", [
    ("ALPHA", {"composite": 35.0, "confidence": "medium"}, "score 35 below the 40 floor."),
    ("ALPHA", {"composite": 35.0, "confidence": "high"}, "score 35 below the 40 floor."),
    ("ALPHA", {"composite": 35.0}, "score 35 below the 40 floor."),
    ("ALPHA", {"composite": 35.0, "confidence": "low"}, None),
    ("ALPHA", {"composite": 40.0, "confidence": "high"}, None),
    ("ALPHA", {"composite": 70.0, "structural_grade": 30.0}, None),
    ("ALPHA", {"score": 30.0}, "score 30 below the 40 floor."),
    ("SPECULATIVE", {"structural_grade": 20.0}, "score 20 below the 40 floor."),
    ("FORTRESS", {"composite": 10.0, "confidence": "high"}, None),
    ("ALPHA", {}, None),
])
def test_score_threats(tier, score, reason):
    h = {"symbol": "X", "name": "Xeno Star", "tier": tier, "value_eur": 100.0}
    expected = [] if reason is None else [_threat("X", "Xeno Star (X)", reason, tier)]
    assert T.threat_items([h, BIG], {"X": score}) == expected


def test_threat_without_a_class_keeps_an_empty_class():
    h = {"symbol": "X", "name": "X", "change_7d": -0.2, "value_eur": 1.0}
    assert T.threat_items([h]) == [_threat("X", "X", "fell 20% in the last 7 days.", "")]


def test_speculative_cap_threat():
    def book(spec, other):
        return [{"symbol": "A", "tier": "ALPHA", "value_eur": other},
                {"symbol": "S", "tier": "SPECULATIVE", "value_eur": spec}]

    assert T.threat_items(book(3.0, 97.0)) == [_threat(
        None, "the Speculative part", "3% of your money, above the 2% cap.",
        "SPECULATIVE")]
    assert T.threat_items(book(2.0, 98.0)) == []          # exactly the cap
    assert T.threat_items(book(1.0, 99.0)) == []
    assert T.threat_items(book(0.5, 0.0)) == [_threat(
        None, "the Speculative part", "100% of your money, above the 2% cap.",
        "SPECULATIVE")]
    assert T.threat_items([]) == [] and T.threat_items(None) == []


def test_bear_market_threat():
    expected = [_threat(None, "the Active part",
                        "the market is falling; no new Active buys, review risk.",
                        "ALPHA", evidence="Market")]
    assert T.threat_items([], regime="bear") == expected
    assert T.threat_items([], regime={"label": "falling"}) == expected
    assert T.threat_items([], regime="bull") == []


# ── swap ──────────────────────────────────────────────────────────────────────

IDEAS = [{"symbol": "LOW", "name": "Low Star", "composite": 60.0},
         {"symbol": "IDEA", "name": "Big Idea", "composite": 70.0},
         {"symbol": "NOSC", "name": "No Score"},
         {"symbol": "ACT", "name": "Alpha Star", "composite": 99.0}]


def test_swap_item():
    book = [hold("ACT", "ALPHA", 0.5, 0.5, 300.0, "Alpha Star")]
    assert T.swap_items(book, IDEAS, {"ACT": {"composite": 50.0}}) == [item(
        "swap", "ACT", "Alpha Star (ACT)", 300.0,
        "Big Idea (IDEA) scores 70 vs 50 for Alpha Star (ACT), more than the "
        "fee/tax hurdle.", TODAY, "ALPHA", "Research",
        target_symbol="IDEA", target_name="Big Idea")]


def test_swap_reads_the_score_from_the_holding_when_not_in_scores():
    book = [hold("SPC", "SPECULATIVE", 0.5, 0.5, 80.0, "Space Bet", composite=40.0)]
    got = T.swap_items(book, IDEAS)
    assert [(i["symbol"], i["target_symbol"], i["class"]) for i in got] == \
        [("SPC", "ACT", "SPECULATIVE")]


@pytest.mark.parametrize("held, swapped", [(65.0, False), (64.96, True), (64.9, True)])
def test_swap_hurdle_edge(held, swapped):
    book = [hold("ACT", "ALPHA", 0.5, 0.5, 300.0)]
    assert bool(T.swap_items(book, IDEAS[:2], {"ACT": {"composite": held}})) is swapped


def test_swap_hurdle_argument():
    book = [hold("ACT", "ALPHA", 0.5, 0.5, 300.0)]
    assert T.swap_items(book, IDEAS[:2], {"ACT": {"composite": 50.0}}, hurdle=0.25) == []


def test_swap_skips_without_stopping():
    book = [hold("LT", "FORTRESS", 0.5, 0.5, 100.0),
            hold("UNS", "ALPHA", 0.1, 0.1, 100.0),
            hold("ACT", "ALPHA", 0.4, 0.4, 300.0)]
    got = T.swap_items(book, IDEAS[:2], {"LT": {"composite": 1.0}, "ACT": {"composite": 50.0}})
    assert [i["symbol"] for i in got] == ["ACT"]
    only_self = [{"symbol": "ACT", "composite": 99.0}]
    two = [hold("ACT", "ALPHA", 0.5, 0.5, 1.0), hold("B", "ALPHA", 0.5, 0.5, 1.0)]
    assert [i["symbol"] for i in T.swap_items(two, only_self, {
        "ACT": {"composite": 10.0}, "B": {"composite": 10.0}})] == ["B"]
    assert T.swap_items(book, None, {"ACT": {"composite": 50.0}}) == []
    assert T.swap_items(None, IDEAS) == []


# ── the ONE list ──────────────────────────────────────────────────────────────

def test_build_combines_all_rules_today_first():
    book = [hold("LT", "FORTRESS", 0.3, 0.6, 300.0),
            hold("ACT", "ALPHA", 0.1, 0.3, 100.0),
            hold("SW", "ALPHA", 0.6, 0.1, 600.0, composite=30.0, confidence="high")]
    got = T.build(book, {"SW": {"composite": 30.0, "confidence": "high"}},
                  IDEAS[:2], regime="bull")
    assert [(i["verb"], i["symbol"], i["horizon"]) for i in got] == [
        ("threat", "SW", TODAY), ("swap", "SW", TODAY),
        ("buy", "ACT", MONTH), ("plan_up", "LT", MONTH)]


def test_build_passes_the_regime_and_scores_through():
    book = [hold("LT", "FORTRESS", 0.8, 0.8, 800.0), hold("ACT", "ALPHA", 0.2, 0.4, 200.0)]
    bear = T.build(book, regime="bear")
    assert [(i["verb"], i["label"]) for i in bear] == [("threat", "the Active part")]
    scored = T.build(book, {"ACT": {"composite": 10.0, "confidence": "high"}})
    assert [i["verb"] for i in scored] == ["threat", "buy"]


def test_build_empty():
    assert T.build(None) == [] and T.build([]) == []


# ── reason numbers large enough that a changed factor shows ───────────────────

def test_reason_percentages_are_exact_for_large_numbers():
    sell = [hold("LT", "FORTRESS", 0.1, 0.1, 100.0), hold("A", "ALPHA", 0.9, 0.1, 900.0)]
    assert T.rebalance_items(sell)[0]["reason"] == "80pp above its 10% target."
    sell2 = [hold("LT", "FORTRESS", 0.1, 0.1, 100.0), hold("A", "ALPHA", 0.9, 0.7, 900.0)]
    assert T.rebalance_items(sell2)[0]["reason"] == "20pp above its 70% target."
    buy = [hold("LT", "FORTRESS", 0.9, 0.9, 900.0), hold("A", "ALPHA", 0.1, 0.9, 100.0)]
    assert T.rebalance_items(buy)[0]["reason"] == "80pp below its 90% target."
    up = [hold("LT", "FORTRESS", 0.1, 0.7, 100.0)]
    assert T.plan_change_items(up)[0]["reason"] == "60pp below its 70% target."
    down = [hold("LT", "FORTRESS", 0.9, 0.3, 900.0)]
    assert T.plan_change_items(down)[0]["reason"].startswith("60pp above its 30% target;")
    stop = {"symbol": "S", "tier": "SPECULATIVE", "pnl_pct": -0.6, "value_eur": 1.0}
    assert T.threat_items([stop, BIG])[0]["reason"] == "stop-loss hit (-60%)."
    fell = {"symbol": "F", "tier": "ALPHA", "change_7d": -0.6, "value_eur": 1.0}
    assert T.threat_items([fell, BIG])[0]["reason"] == "fell 60% in the last 7 days."


def test_swap_keeps_going_after_a_holding_inside_the_hurdle():
    book = [hold("A", "ALPHA", 0.5, 0.5, 1.0), hold("B", "ALPHA", 0.5, 0.5, 1.0)]
    got = T.swap_items(book, IDEAS[:2], {"A": {"composite": 69.0}, "B": {"composite": 50.0}})
    assert [i["symbol"] for i in got] == ["B"]
