"""
test_v10_7_0_phase1.py — Money model and naming dictionary (v10.7.0, Phase 1).

Intent: lock the root fix. The system keeps exactly three money pools; the old
cash floor is gone; risk profiles describe the INVESTED pool only; and the
Section 11 naming dictionary is present and plain.

Invariants:
  - RISK_PROFILES values are 3-tuples (long_term_min, active_max, max_position).
  - No cash-floor constant or key remains in quant.config.
  - The invested-only bucket limits exist and are fractions.
  - The naming dictionary maps the old tier words to the new plain words.
  - The forbidden-token list is defined for the copy test.
"""
from __future__ import annotations

import quant.config as config
from quant.ui import copy as C


def test_risk_profiles_are_invested_only_three_tuples():
    for name, limits in config.RISK_PROFILES.items():
        assert len(limits) == 3, f"{name} must be (long_term_min, active_max, max_position)"
        long_term_min, active_max, max_position = limits
        assert 0.0 < long_term_min <= 1.0
        assert 0.0 < active_max <= 1.0
        assert 0.0 < max_position <= 1.0


def test_no_cash_floor_remains():
    assert not hasattr(config, "SAFETY_BUCKET_MIN")
    assert not hasattr(config, "cash_floor")
    # The old 5-tuple shape is gone: no profile carries a fifth element.
    for limits in config.RISK_PROFILES.values():
        assert len(limits) == 3


def test_invested_only_bucket_limits():
    assert config.LONG_TERM_MIN == 0.40
    assert config.ACTIVE_MAX == 0.50
    assert config.BETS_MAX == 0.02


def test_monthly_allocator_and_sizing_constants():
    assert config.MONTHLY_LONG_TERM_SHARE == 0.70
    assert config.SELL_MIN_EUR == 25.0
    assert config.SELL_ROUND_STEP_EUR == 5.0
    assert config.BUY_ROUND_STEP_EUR == 10.0
    assert config.UNTOUCHABLE_POSITION_EUR == 100.0
    assert config.SYNC_REMINDER_DAYS == 35


def test_risk_profile_descriptions_are_plain_and_invested():
    for name in config.RISK_PROFILES:
        desc = config.RISK_PROFILE_DESCRIPTIONS[name]
        assert "invested" in desc
        assert "cash" not in desc.lower()


def test_tier_dictionary_words():
    assert C.TIER_FORTRESS == "Long-term (never sell)"
    assert C.TIER_ALPHA == "Active (may sell)"
    assert C.TIER_SPECULATIVE == "Small bets (high risk)"
    assert C.tier_word("FORTRESS") == C.TIER_FORTRESS
    assert C.tier_word("ALPHA") == C.TIER_ALPHA
    assert C.tier_word("SPECULATIVE") == C.TIER_SPECULATIVE


def test_status_dictionary_has_no_forbidden_tokens():
    assert C.STATUS_ON_TRACK == "OK, do nothing"
    assert C.STATUS_TRIM == "Sell part"
    assert C.STATUS_WAITING.startswith("Cooldown until")
    for token in C.FORBIDDEN_TOKENS:
        assert token not in C.STATUS_ON_TRACK
        assert token not in C.STATUS_TRIM
        assert token not in C.STATUS_WAITING


def test_section_titles_use_new_vocabulary():
    assert C.SEC_TIERS == "How your money is split"
    assert C.SEC_EMERGENCY == "If you need cash now"
    assert C.SEC_TAX_LOSS == "Losses you can use to lower tax"
    assert C.SEC_STEPS == "Your steps this week"


def test_forbidden_tokens_cover_section_11():
    required = {
        "Trim", "On track", "Waiting until", "Current_Value_EUR",
        "Broker_PnL_EUR", "Tier balance", "Emergency liquidity",
        "Tax-loss harvesting", "SECTOR", "SATELLITE",
    }
    assert required.issubset(set(C.FORBIDDEN_TOKENS))
