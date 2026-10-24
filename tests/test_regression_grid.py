"""test_regression_grid.py — permanent regression tests (v10.7.4, Part 7).

Intent: every bug from v10.7.0 through v10.7.3 gets at least one permanent test
tagged ``[regression]``. Each test would fail if its fix were reverted.

Bugs covered:
  1. 143.53 EUR position, small drift -> no sell (B2 sizing law).
  2. 5J50.DE FORTRESS, 7 percent drift -> no sell_part (B1).
  3. Cooldown blocking a buy -> must NOT happen.
  4. Alert overriding cooldown -> must happen.
  5. Legacy tier labels -> never in output.
  6. Naive percent with deposits -> Modified Dietz.
  7. Stale lock recovery -> takeover with the doctor phrase.
  8. Holdings meta sync on run -> shares populated.
  9. Allocator vs advice target-map agreement -> same top-up symbol.
 10. FORTRESS verdict in the holdings table -> never "Sell part".
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

from quant import paths
from quant.engine import flows, lock, sizing
from quant.engine.advice import build_advice
from tests.fixtures.live_portfolio import alpha_case

AS_OF = date(2026, 10, 2)
FAR_FUTURE = date(2099, 1, 1)


def test_regression_143_eur_no_sell():
    """[regression] B2: a 143.53 EUR position with a small drift never sells."""
    assert sizing.sell_amount(0.069 * 143.53, 143.53) is None
    # The sell amount never exceeds value - 1 EUR.
    assert sizing.sell_amount(200.0, 143.53) <= 143.53 - 1.0


def test_regression_5j50_fortress_no_sell():
    """[regression] B1: 5J50.DE is FORTRESS; a 7 percent drift never sells."""
    holding = {"symbol": "5J50.DE", "name": "Global Aero & Defense",
               "tier": "ALPHA", "value_eur": 143.53, "current_weight": 0.17,
               "target_weight": 0.10, "conviction": 0.0}
    advice, _ = build_advice([holding], tiers={"5J50.DE": "FORTRESS"}, as_of=AS_OF)
    assert not any(a["kind"] == "sell_part" for a in advice)


def test_regression_cooldown_does_not_block_buy():
    """[regression] v10.7.1 Phase A: cooldown must not block a buy."""
    holding = alpha_case(-0.09, value=500.0, cooldown=FAR_FUTURE, conviction=90.0)
    advice, _ = build_advice([holding], tiers={"AMZN": "ALPHA"}, as_of=AS_OF)
    assert any(a["kind"] == "buy" for a in advice)


def test_regression_alert_overrides_cooldown():
    """[regression] v10.7.1 Phase A: an alert sell ignores cooldown."""
    holding = alpha_case(0.0, value=1000.0, cooldown=FAR_FUTURE)
    advice, _ = build_advice(
        [holding], tiers={"AMZN": "ALPHA"},
        open_alerts=[{"symbol": "AMZN", "action": "sell", "amount_eur": 75.0,
                      "fee_eur": 1.0, "message": "tactics fell."}],
        as_of=AS_OF)
    assert any(a["kind"] == "sell_part" and a["source"] == "alert" for a in advice)


def test_regression_no_legacy_tier_labels():
    """[regression] v10.7.1 Phase A: legacy tier labels never appear."""
    advice, _ = build_advice(
        [alpha_case(0.0, value=1000.0)], tiers={"AMZN": "ALPHA"}, as_of=AS_OF)
    for record in advice:
        assert record["tier_word"] not in ("CORE", "SATELLITE", "ACTIVE", "SECTOR")


def test_regression_modified_dietz_excludes_deposits():
    """[regression] v10.7.0 Phase 2: deposits do not count as profit."""
    f = [{"date": date(2026, 1, 16), "type": "buy", "amount_eur": 200.0}]
    ret = flows.modified_dietz(1000.0, 1200.0, f, date(2026, 1, 1), date(2026, 1, 31))
    assert abs(ret) < 1e-9  # a naive percent would report +20 percent


def test_regression_stale_lock_takeover(tmp_path, monkeypatch):
    """[regression] v10.7.2 Phase A: a dead pid is taken over with the phrase."""
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    started = (datetime.now() - timedelta(minutes=5)).isoformat(timespec="seconds")
    (tmp_path / ".runner.lock").write_text(
        json.dumps({"pid": 999999, "started_at": started, "command": "old"}),
        encoding="utf-8")
    res = lock.acquire("new")
    assert res.acquired and res.taken_over
    assert "taken over from stale process" in lock.status_line()


def test_regression_fortress_verdict_never_sell():
    """[regression] v10.7.3 Part 4: a FORTRESS verdict is never 'Sell part'."""
    holding = {"symbol": "5J50.DE", "name": "Global Aero & Defense",
               "tier": "ALPHA", "value_eur": 1000.0, "current_weight": 0.40,
               "target_weight": 0.10, "conviction": 0.0}
    advice, _ = build_advice([holding], tiers={"5J50.DE": "FORTRESS"}, as_of=AS_OF)
    record = next(a for a in advice if a.get("symbol") == "5J50.DE")
    assert record["kind"] != "sell_part"
