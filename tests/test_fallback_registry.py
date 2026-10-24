"""test_fallback_registry.py — every declared fallback is triggered and honest.

Intent (v10.7.5, Part 3): a LEGITIMATE FALLBACK is declared, marked, tested, and
surfaced. This test iterates ``FALLBACKS`` so a new entry without a triggering
test fails CI automatically.

Invariants:
  - Every registry entry has a trigger and a test id.
  - Each trigger returns the documented default.
"""
from __future__ import annotations

from datetime import date
from unittest import mock

import numpy as np
import pytest

from quant.engine.fallback_registry import FALLBACKS


class _BrokenConn:
    def execute(self, *_a, **_k):
        raise RuntimeError("db down")


def _raise(*_a, **_k):
    raise RuntimeError("boom")

def _triggers() -> dict[str, tuple]:
    """Map each registry key to (trigger callable, expected default)."""
    from quant.data import user_data
    from quant.engine import (
        advice_state,
        analysis,
        flows,
        mtm,
        score_model,
        sizing,
        today_list,
    )
    from quant.portfolio import account, portfolio, tier_manager
    from quant.reporting import artifacts

    def _persist_broken():
        with mock.patch("quant.engine.holdings_view.holdings_view", side_effect=_raise):
            return today_list.persist()

    def _snapshots_broken():
        # Unreadable snapshots fall back to the current table (empty here).
        with mock.patch("quant.engine.positions.positions_now", return_value=[]):
            return mtm._snapshots(_BrokenConn())

    def _fx_rate_broken():
        with mock.patch("quant.data.currency.get_fx_to_eur", side_effect=_raise):
            return mtm._fx_rate("AMZN")

    def _analysis_regime_broken():
        with mock.patch("quant.data.database.read_only_connection", side_effect=_raise):
            return artifacts._analysis_regime()

    return {
        "flows.load_flows": (lambda: flows.load_flows(_BrokenConn()), []),
        "sizing.is_untouchable": (lambda: sizing.is_untouchable("bad"), True),
        "sizing.sell_amount": (lambda: sizing.sell_amount("bad", 100.0), None),
        "score_model._num": (lambda: score_model._num("n/a"), None),
        "account.load_account": (
            lambda: account.load_account("/nonexistent/account.yaml").loaded, False),
        "portfolio.load_portfolio": (
            lambda: portfolio.load_portfolio("/nonexistent/portfolio.csv").empty, True),
        "portfolio.portfolio_issues": (
            lambda: portfolio.portfolio_issues("/nonexistent/portfolio.csv"), []),
        "tier_manager.load_tiers": (
            lambda: tier_manager.load_tiers("/nonexistent/tiers.csv").empty, True),
        "account.write_account_fields": (
            lambda: account.write_account_fields(
                {}, "/nonexistent_dir/x/account.yaml"), False),
        "user_data._copy_if_missing": (
            lambda: user_data._copy_if_missing(
                "/nonexistent/src", "/nonexistent/dst"), False),
        "today_list._rows": (
            lambda: today_list._rows("SELECT 1", conn=_BrokenConn()), []),
        "today_list.persist": (_persist_broken, 0),
        "advice_state._insert": (
            lambda: advice_state._insert(_BrokenConn(), {}, date.today()), None),
        "artifacts._analysis_regime": (_analysis_regime_broken, {}),
        "advice_state._as_date": (lambda: advice_state._as_date("not-a-date"), None),
        "advice_state.mark_done": (lambda: advice_state.mark_done(_BrokenConn(), 1), False),
        "advice_state._query": (lambda: advice_state._query(_BrokenConn(), "SELECT 1"), []),
        "analysis._to_date": (lambda: analysis._to_date("not-a-date"), None),
        "analysis._technical_from_prices": (
            lambda: analysis._technical_from_prices(_BrokenConn(), "X"), None),
        "analysis._fundamentals_row": (
            lambda: analysis._fundamentals_row(_BrokenConn(), "X"), None),
        "analysis._sentiment_component": (
            lambda: analysis._sentiment_component(_BrokenConn(), "X"), None),
        "mtm._as_date": (lambda: mtm._as_date("not-a-date"), None),
        "mtm._snapshots": (_snapshots_broken, []),
        "mtm._split_rows": (lambda: mtm._split_rows(_BrokenConn()), []),
        "mtm._fx_rate": (_fx_rate_broken, 1.0),
        "mtm._price_history": (lambda: mtm._price_history(_BrokenConn(), {"X"}), {}),
        "mtm._query": (lambda: mtm._query(_BrokenConn(), "SELECT 1"), []),
    }

def test_every_registry_entry_has_a_trigger():
    triggers = _triggers()
    for key, entry in FALLBACKS.items():
        assert key in triggers, f"{key} has no triggering test"
        assert entry.get("test"), f"{key} has no test id"
        assert entry.get("marker"), f"{key} has no marker"

@pytest.mark.parametrize("key", sorted(FALLBACKS))
def test_fallback_returns_its_default(key):
    callable_, expected = _triggers()[key]
    actual = callable_()
    if isinstance(expected, np.ndarray):
        assert np.allclose(actual, expected)
    else:
        assert actual == expected
