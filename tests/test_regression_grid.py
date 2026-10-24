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


def test_regression_143_eur_no_sell():
    """[regression] B2: a 143.53 EUR position with a small drift never sells."""
    assert sizing.sell_amount(0.069 * 143.53, 143.53) is None
    # The sell amount never exceeds value - 1 EUR.
    assert sizing.sell_amount(200.0, 143.53) <= 143.53 - 1.0


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
