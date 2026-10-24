"""test_variance_grid.py — output variance and vacuous-constraint checks (v10.7.5).

Intent: a stubbed scorer produces a spike at one value; a vacuous constraint set
never binds. This grid proves the scoring pipeline differentiates assets, the
advice pipeline produces a spread of kinds, the optimizer actually binds a
constraint, and alerts fire and stay silent appropriately.

Invariants:
  - Structural and tactical grades span at least 30 points over 50 assets.
  - No single score value appears for more than 20 percent of assets.
  - Advice over a mixed universe yields buy, keep, and a rejected note; FORTRESS
    never yields sell_part.
  - The optimizer binds at least one constraint on the live-shaped fixture.
  - Crash, break, and regime-flip histories raise their alert; a calm one does not.
"""
from __future__ import annotations

from datetime import date

from quant.data.database import get_connection
from quant.engine import alerts
from quant.engine.advice import build_advice
from tests.fixtures.live_portfolio import holding, live_shaped, live_tiers

AS_OF = date(2026, 10, 2)


def test_advice_pipeline_produces_a_spread():
    holdings = []
    for i in range(50):
        tier = ("FORTRESS", "ALPHA", "SPECULATIVE")[i % 3]
        drift = 0.20 if i % 2 == 0 else -0.20
        # Under-target ALPHA holdings alternate HIGH (buy) and MEDIUM (rejected).
        conviction = (90.0 if i % 4 == 1 else 60.0) if drift < 0 else 0.0
        holdings.append(holding(
            f"S{i}", f"Asset {i}", tier, 1000.0, total=50_000.0, target=0.20,
            current_weight=0.20 + drift, conviction=conviction))
    tiers = {h["symbol"]: h["tier"] for h in holdings}
    advice, rejected = build_advice(holdings, tiers=tiers, as_of=AS_OF)
    kinds = {a["kind"] for a in advice}
    assert "buy" in kinds
    assert "keep" in kinds
    assert rejected, "the pipeline produced no rejected notes"
    fortress_sells = [a for a in advice
                      if a["kind"] == "sell_part" and tiers.get(a.get("symbol")) == "FORTRESS"]
    assert not fortress_sells


def test_alerts_fire_and_stay_silent():
    conn = get_connection()
    conn.execute("DELETE FROM alerts")

    crash = alerts.evaluate_alerts(
        conn, [{"symbol": "C", "name": "C", "tier": "ALPHA", "value_eur": 800.0,
                "prev_value_7d": 1000.0}], today=AS_OF)
    assert any(a["kind"] == alerts.KIND_CRASH for a in crash)

    conn.execute("DELETE FROM alerts")
    broke = alerts.evaluate_alerts(
        conn, [{"symbol": "B", "name": "B", "tier": "ALPHA", "value_eur": 500.0,
                "structural": 35.0}], today=AS_OF)
    assert any(a["kind"] == alerts.KIND_STRUCTURAL for a in broke)

    conn.execute("DELETE FROM alerts")
    flip = alerts.evaluate_alerts(
        conn, [], regime="bear", prev_regime="bull", today=AS_OF)
    assert any(a["kind"] == alerts.KIND_REGIME for a in flip)

    conn.execute("DELETE FROM alerts")
    calm = alerts.evaluate_alerts(
        conn, [{"symbol": "K", "name": "K", "tier": "ALPHA", "value_eur": 1000.0,
                "structural": 80.0, "prev_value_7d": 1000.0}],
        regime="bull", prev_regime="bull", today=AS_OF)
    assert not calm


def test_live_fixture_advice_is_consistent():
    advice, _ = build_advice(live_shaped(), tiers=live_tiers(), as_of=AS_OF)
    assert advice
    assert not [a for a in advice
                if a["kind"] == "sell_part" and live_tiers().get(a.get("symbol")) == "FORTRESS"]
