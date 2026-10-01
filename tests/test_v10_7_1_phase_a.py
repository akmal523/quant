"""
test_v10_7_1_phase_a.py — One advice pipeline (v10.7.1, Part 1).

Intent: lock the historical failures. Sells come only from build_advice, which
calls the sizing laws; a FORTRESS sell cannot be expressed by any consumer.

Invariants:
  - 143.53 EUR position, 6.9 percent drift: no sell_part anywhere.
  - A considered-but-suppressed sell becomes a rejected note.
  - Cooldown blocks sells, never buys; alerts override cooldown.
  - No legacy tier label appears as a tier value.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from quant.engine.advice import build_advice
from quant.portfolio.account import AccountState
from quant.reporting.actions import build_actions
from quant.reporting.briefing import build_briefing_md
from quant.ui import copy as C


def _holding(tier, drift, value=143.53, conviction=0.0, cooldown=None):
    return {
        "symbol": "5J50.DE", "name": "Global Aero & Defense", "tier": tier,
        "value_eur": value, "current_weight": 0.5 + drift, "target_weight": 0.5,
        "conviction": conviction, "cooldown_until": cooldown,
    }


def test_fortress_small_drift_never_sells():
    advice, rejected = build_advice([_holding("FORTRESS", 0.069)])
    assert not [a for a in advice if a["kind"] == "sell_part"]
    assert not [r for r in rejected if r["considered_action"] == C.ADVICE_SELL_PART]


def test_alpha_small_position_sell_is_rejected():
    advice, rejected = build_advice([_holding("ALPHA", 0.069)])
    assert not [a for a in advice if a["kind"] == "sell_part"]
    assert any(r["considered_action"] == C.ADVICE_SELL_PART for r in rejected)


def test_cooldown_blocks_sell_not_buy():
    # A sell in cooldown is rejected.
    _advice, rejected = build_advice(
        [_holding("ALPHA", 0.20, value=1000.0, cooldown=date(2099, 1, 1))],
        as_of=date(2026, 10, 1))
    assert any("cooldown" in r["plain_reason"] for r in rejected)
    # A buy in cooldown is NOT blocked.
    buy_holding = {
        "symbol": "AMZN", "name": "Amazon.com", "tier": "ALPHA",
        "value_eur": 1000.0, "current_weight": 0.30, "target_weight": 0.50,
        "conviction": 90.0, "cooldown_until": date(2099, 1, 1),
    }
    advice, _rej = build_advice([buy_holding], as_of=date(2026, 10, 1))
    assert any(a["kind"] == "buy" for a in advice)


def test_alert_sell_overrides_cooldown():
    holding = _holding("ALPHA", 0.0, value=1000.0, cooldown=date(2099, 1, 1))
    advice, _rej = build_advice(
        [holding],
        open_alerts=[{"symbol": "5J50.DE", "action": "sell", "amount_eur": 75.0,
                      "fee_eur": 1.0, "message": "tactics fell."}],
        as_of=date(2026, 10, 1))
    assert any(a["kind"] == "sell_part" and a["source"] == "alert" for a in advice)


def test_briefing_has_no_legacy_tier_values():
    audit = pd.DataFrame([
        {"Symbol": "5J50.DE", "Name": "Global Aero & Defense", "Tier": "SECTOR",
         "Target_Weight": "10.0%", "Drift": "6.9%", "Value_EUR": 143.53,
         "Recommendation": "SELL 20 EUR (SECTOR drift 6.9% exceeds 6.0% threshold)"},
    ])
    md = build_briefing_md(
        as_of="2026-10-01", version="10.7.1", regime_label="rising", regime_prob=0.8,
        regime_source="HMM", audit_df=audit,
        account=AccountState("EUR", 0.0, "balanced", True),
        total_value=143.53, pnl_eur=0.0, pnl_pct=0.0, with_news=0, without_news=0,
        latest_bar="2026-10-01")
    for token in ("SECTOR", "SATELLITE", "CORE", "ACTIVE"):
        assert token not in md
    assert "Active (may sell)" in md


def test_build_actions_uses_dictionary_words():
    audit = pd.DataFrame([
        {"Symbol": "AMZN", "Tier": "ACTIVE", "Target_Weight": "20.0%", "Drift": "-16.1%",
         "Recommendation": "BUY 150 EUR (ACTIVE drift -16.1% exceeds 5.0% threshold)"},
    ])
    acts = {a["symbol"]: a for a in build_actions(audit)}
    assert acts["AMZN"]["action"] == C.ADVICE_BUY


# ── v10.7.2 (Part 5.1): FORTRESS sell violation regression ────────────────────

def test_fortress_never_sells_real_scenario():
    """Real-world bug: 5J50.DE is FORTRESS in tiers.csv but the audit's legacy
    Tier column says SECTOR (-> ALPHA). The system must NOT generate sell_part
    advice for it, regardless of drift."""
    holdings = [{
        "symbol": "5J50.DE", "name": "Global Aero & Defense",
        "tier": "ALPHA",  # the audit's legacy tier (SECTOR -> ALPHA)
        "value_eur": 143.53, "current_weight": 0.17, "target_weight": 0.10,
        "conviction": 0.0, "cooldown_until": None,
    }]
    tiers = pd.DataFrame([{
        "symbol": "5J50.DE", "tier": "FORTRESS",
        "last_updated": "2026-10-01", "notes": "Auto-balanced from ALPHA",
    }])
    advice, _rejected = build_advice(
        holdings=holdings, tiers=tiers, as_of=date(2026, 10, 2))

    sell = [a for a in advice if a["kind"] == "sell_part" and a["symbol"] == "5J50.DE"]
    assert len(sell) == 0, "FORTRESS assets must never receive sell_part advice"
    assert len(advice) == 1
    assert advice[0]["kind"] in {"change_savings_plan", "keep"}
    assert advice[0]["tier_word"] == "Long-term (never sell)"


def test_build_actions_resolves_tier_from_tiers_csv(monkeypatch):
    """The audit's legacy Tier must not decide a FORTRESS sell in the real run."""
    from quant.reporting import actions as actions_mod

    monkeypatch.setattr(actions_mod, "_tiers_map", lambda: {"5J50.DE": "FORTRESS"})
    audit = pd.DataFrame([{
        "Symbol": "5J50.DE", "Tier": "SECTOR", "Target_Weight": "10.0%",
        "Drift": "7.0%", "Value_EUR": 143.53,
        "Recommendation": "SELL 20 EUR (SECTOR drift 7.0% exceeds 6.0% threshold)",
    }])
    acts = actions_mod.build_actions(audit)
    assert not [a for a in acts if a["kind"] == "sell_part"]
