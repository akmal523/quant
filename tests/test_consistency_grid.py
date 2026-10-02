"""test_consistency_grid.py — cross-category consistency (v10.7.4, Part 4).

Intent: two screens, one truth. The same fact must read the same everywhere.

Invariants:
  - TARGET_WEIGHTS_INVESTED is the sole per-symbol target map.
  - tiers.csv is the sole tier source; the audit's legacy column never decides.
  - revalue_holdings is the sole estimated-value source.
"""
from __future__ import annotations

from datetime import date

from quant import config
from quant.data.database import get_connection
from quant.engine import advice as advice_mod
from quant.engine import allocator, plans, valuation
from quant.engine.advice import build_advice
from quant.ui.copy import BROKER_STATEMENT_AS_OF
from quant.ui.render import _verdict_word

AS_OF = date(2026, 10, 2)


# ── Part 4.1: single target map ───────────────────────────────────────────────

def test_build_advice_reads_target_map():
    """A holding without target_weight falls back to TARGET_WEIGHTS_INVESTED."""
    holding = {"symbol": "EUNL.DE", "name": "iShares Core MSCI World",
               "tier": "FORTRESS", "value_eur": 1000.0, "current_weight": 0.10,
               "conviction": 0.0}
    advice, _ = build_advice([holding], tiers={"EUNL.DE": "FORTRESS"}, as_of=AS_OF)
    # target 0.50, current 0.10 -> gap 0.40 > 0.10 -> change_savings_plan.
    assert any(a["kind"] == "change_savings_plan" for a in advice)


def test_allocator_reads_target_map():
    holding = {"symbol": "EUNL.DE", "name": "iShares Core MSCI World",
               "tier": "FORTRESS", "value_eur": 1000.0, "current_weight": 0.10}
    legs = allocator.allocate(200.0, [holding], regime="bull")
    long_legs = [leg for leg in legs if leg["kind"] == "long_term"]
    assert long_legs and long_legs[0]["symbol"] == "EUNL.DE"


def test_single_target_map_instance():
    """No parallel maps: both modules reference the ONE config dictionary."""
    assert advice_mod.TARGET_WEIGHTS_INVESTED is config.TARGET_WEIGHTS_INVESTED
    assert allocator.TARGET_WEIGHTS_INVESTED is config.TARGET_WEIGHTS_INVESTED


def test_split_reason_uses_target_map_numbers():
    holding = {"symbol": "EUNL.DE", "name": "iShares Core MSCI World",
               "tier": "FORTRESS", "value_eur": 1000.0, "current_weight": 0.10}
    legs = allocator.allocate(200.0, [holding], regime="bull")
    long_leg = next(leg for leg in legs if leg["kind"] == "long_term")
    target_pct = config.TARGET_WEIGHTS_INVESTED["EUNL.DE"] * 100
    assert f"{target_pct:.0f} percent target" in long_leg["reason"]


# ── Part 4.2: single tier source ──────────────────────────────────────────────

def test_advice_resolves_tier_from_tiers_not_holding():
    """The holding's legacy tier says ALPHA; tiers.csv says FORTRESS."""
    holding = {"symbol": "5J50.DE", "name": "Global Aero & Defense",
               "tier": "ALPHA", "value_eur": 1000.0, "current_weight": 0.40,
               "target_weight": 0.10, "conviction": 0.0}
    advice, _ = build_advice([holding], tiers={"5J50.DE": "FORTRESS"}, as_of=AS_OF)
    assert not any(a["kind"] == "sell_part" for a in advice)


def test_verdict_renders_from_build_advice():
    holding = {"symbol": "5J50.DE", "name": "Global Aero & Defense",
               "tier": "ALPHA", "value_eur": 1000.0, "current_weight": 0.40,
               "target_weight": 0.10, "conviction": 0.0}
    advice, _ = build_advice([holding], tiers={"5J50.DE": "FORTRESS"}, as_of=AS_OF)
    record = next(a for a in advice if a.get("symbol") == "5J50.DE")
    assert "Sell part" not in _verdict_word(record)


def test_tier_change_flips_advice_within_one_run():
    holding = {"symbol": "AMZN", "name": "Amazon.com", "tier": "ALPHA",
               "value_eur": 1000.0, "current_weight": 0.40, "target_weight": 0.20,
               "conviction": 0.0}
    advice_alpha, _ = build_advice([holding], tiers={"AMZN": "ALPHA"}, as_of=AS_OF)
    assert any(a["kind"] == "sell_part" for a in advice_alpha)
    advice_fortress, _ = build_advice([holding], tiers={"AMZN": "FORTRESS"}, as_of=AS_OF)
    assert not any(a["kind"] == "sell_part" for a in advice_fortress)


# ── Part 4.3: single value source ─────────────────────────────────────────────

def test_revalue_holdings_is_the_value_source():
    conn = get_connection()
    conn.execute("DELETE FROM holdings_meta")
    conn.execute(
        "INSERT INTO holdings_meta (symbol, shares, sync_date, invested_at_sync) "
        "VALUES ('EUNL.DE', 10.0, ?, 1000.0)", [AS_OF])
    values = valuation.revalue_holdings(conn, {"EUNL.DE": 110.0})
    assert values["EUNL.DE"] == 1100.0


def test_estimated_label_carries_estimated():
    assert "estimated" in valuation.estimated_label(AS_OF)


def test_broker_statement_label():
    assert "broker statement" in BROKER_STATEMENT_AS_OF.format(date="1 Oct 2026")


def test_enter_actuals_updates_estimated_value():
    conn = get_connection()
    conn.execute("DELETE FROM holdings_meta")
    conn.execute("DELETE FROM flows")
    conn.execute(
        "INSERT INTO holdings_meta (symbol, shares, sync_date, invested_at_sync) "
        "VALUES ('EUNL.DE', 10.0, ?, 1000.0)", [AS_OF])
    before = valuation.revalue_holdings(conn, {"EUNL.DE": 100.0})["EUNL.DE"]
    plans.enter_actuals(
        conn, "2026-12",
        [{"symbol": "EUNL.DE", "amount_eur": 100.0, "date": AS_OF}],
        price_lookup={"EUNL.DE": 100.0})
    after = valuation.revalue_holdings(conn, {"EUNL.DE": 100.0})["EUNL.DE"]
    assert after > before
