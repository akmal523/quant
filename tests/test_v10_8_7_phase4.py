"""test_v10_8_7_phase4.py — v10.8.7 Phase 4: T4 (decision taxonomy).

Each verb (Rebalance / Threat / Swap / Plan change) is ONE item in the ONE list
with a verb, an amount, one reason and an evidence pointer. Today items rank
before This-month. A swappable position never gets a standalone sell, so a swap
item and a rebalance item for the same symbol never coexist. All thresholds are
named constants in one config module.

Synthetic fixtures only; the user's real DB and data dir are never touched.
"""
from __future__ import annotations

from quant.engine import action_taxonomy as T

# ── Per-verb triggers ─────────────────────────────────────────────────────────

def test_rebalance_trigger_yields_one_item():
    h = [{"symbol": "AAA", "name": "AAA", "tier": "ALPHA",
          "current_weight": 0.30, "target_weight": 0.10}]
    items = T.rebalance_items(h)
    assert len(items) == 1
    it = items[0]
    assert it["verb"] == T.VERB_REBALANCE
    assert it["reason"] and it["evidence"]


def test_threat_stop_loss_trigger():
    h = [{"symbol": "BBB", "name": "BBB", "tier": "SPECULATIVE", "pnl_pct": -0.55}]
    items = T.threat_items(h)
    assert len(items) == 1 and items[0]["verb"] == T.VERB_THREAT
    assert "stop-loss" in items[0]["reason"]


def test_threat_drawdown_trigger():
    h = [{"symbol": "CCC", "name": "CCC", "tier": "ALPHA", "pnl_pct": -0.25}]
    items = T.threat_items(h)
    assert len(items) == 1 and items[0]["verb"] == T.VERB_THREAT
    assert "from entry" in items[0]["reason"]


def test_threat_score_floor_trigger():
    h = [{"symbol": "DDD", "name": "DDD", "tier": "ALPHA"}]
    items = T.threat_items(h, scores={"DDD": {"active_score": 30}})
    assert len(items) == 1 and items[0]["verb"] == T.VERB_THREAT
    assert "floor" in items[0]["reason"]


def test_threat_bear_regime_trigger():
    items = T.threat_items([], regime="bear")
    assert len(items) == 1 and items[0]["verb"] == T.VERB_THREAT
    assert "de-risk" in items[0]["reason"]


def test_swap_trigger_is_paired():
    h = [{"symbol": "EEE", "name": "EEE", "tier": "ALPHA", "value_eur": 100.0}]
    ideas = [{"symbol": "FFF", "name": "FFF", "score": 90}]
    items = T.swap_items(h, ideas, scores={"EEE": {"active_score": 30}})
    assert len(items) == 1
    it = items[0]
    assert it["verb"] == T.VERB_SWAP
    assert it["target_symbol"] == "FFF"
    assert it["amount_eur"] == 100.0
    assert "after hurdle" in it["reason"]


def test_plan_change_requires_k_months():
    h = [{"symbol": "GGG", "name": "GGG", "tier": "FORTRESS",
          "current_weight": 0.20, "target_weight": 0.50}]
    assert T.plan_change_items(h, months_under=2) == []
    items = T.plan_change_items(h, months_under=3)
    assert len(items) == 1 and items[0]["verb"] == T.VERB_PLAN


# ── Ranking + dedupe ──────────────────────────────────────────────────────────

def test_today_ranks_before_this_month():
    h = [{"symbol": "AAA", "tier": "SPECULATIVE", "pnl_pct": -0.6, "value_eur": 50},
         {"symbol": "GGG", "tier": "FORTRESS", "current_weight": 0.2,
          "target_weight": 0.5}]
    items = T.build(h, months_under=3)
    horizons = [i["horizon"] for i in items]
    assert horizons == sorted(horizons, key=lambda x: 0 if x == "Today" else 1)


def test_swap_and_rebalance_never_coexist():
    """A swappable position never gets a standalone sell/rebalance (T4)."""
    h = [{"symbol": "DDD", "name": "DDD", "tier": "ALPHA", "value_eur": 100.0,
          "current_weight": 0.30, "target_weight": 0.10}]  # drift 0.20 > 0.05
    ideas = [{"symbol": "EEE", "name": "EEE", "score": 90}]
    items = T.build(h, scores={"DDD": {"active_score": 30}}, ideas=ideas)
    verbs = {i["verb"] for i in items if i["symbol"] == "DDD"}
    assert T.VERB_SWAP in verbs
    assert T.VERB_REBALANCE not in verbs


def test_every_item_has_verb_amount_reason_evidence():
    h = [{"symbol": "AAA", "name": "AAA", "tier": "ALPHA",
          "current_weight": 0.30, "target_weight": 0.10, "value_eur": 100.0}]
    for it in T.build(h):
        assert it["verb"]
        assert "amount_eur" in it
        assert it["reason"]
        assert it["evidence"]


# ── Thresholds live in one config module ──────────────────────────────────────

def test_thresholds_are_named_config_constants():
    from quant import config

    for name in ("THREAT_DRAWDOWN", "SCORE_FLOOR", "SPEC_STOP", "SPEC_TAKE",
                 "SWAP_HURDLE", "PLAN_CHANGE_MONTHS", "PLAN_CHANGE_GAP",
                 "IDEA_FLOOR"):
        assert hasattr(config, name), f"missing config constant {name}"
    # The taxonomy reads them from config, not local magic numbers.
    assert T.SCORE_FLOOR == config.SCORE_FLOOR
    assert T.SWAP_HURDLE == config.SWAP_HURDLE
    assert T.THREAT_DRAWDOWN == config.THREAT_DRAWDOWN
