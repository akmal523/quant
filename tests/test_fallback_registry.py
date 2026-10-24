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


class _BrokenLookup:
    def get(self, _k):
        raise RuntimeError("lookup down")


def _raise(*_a, **_k):
    raise RuntimeError("boom")


def _triggers() -> dict[str, tuple]:
    """Map each registry key to (trigger callable, expected default)."""
    from quant.data import user_data
    from quant.engine import (
        advice,
        advice_state,
        alerts,
        flows,
        news_pillar,
        notify,
        sizing,
        today_list,
        valuation,
    )
    from quant.portfolio import account, portfolio, tier_manager
    from quant.reporting import artifacts

    def _persist_broken():
        with mock.patch("quant.engine.holdings_view.holdings_view", side_effect=_raise):
            return today_list.persist()

    def _analysis_regime_broken():
        with mock.patch("quant.data.database.read_only_connection", side_effect=_raise):
            return artifacts._analysis_regime()

    def _read_cache_broken():
        with mock.patch.object(news_pillar, "_cache_path", return_value="/nonexistent/x.json"):
            return news_pillar._read_cache()

    def _model_available_broken():
        with mock.patch("importlib.util.find_spec", side_effect=RuntimeError("boom")):
            return news_pillar.model_available()

    def _last_rebalance_broken():
        with mock.patch("quant.data.database.get_connection", side_effect=RuntimeError("boom")):
            return portfolio.get_last_rebalance("X")

    def _load_state_missing():
        with mock.patch.object(notify, "_state_path",
                               return_value="/nonexistent/notify_state.json"):
            return notify._load_state()

    return {
        "advice._spec_change": (lambda: advice._spec_change({"pnl_pct": "bad"}), None),
        "advice._tier_lookup": (lambda: advice._tier_lookup("bad"), {}),
        "advice._as_date": (lambda: advice._as_date("not-a-date"), None),
        "alerts._num": (lambda: alerts._num("bad"), None),
        "alerts.has_open_alert": (lambda: alerts.has_open_alert(_BrokenConn(), "X", "k"), False),
        "alerts._has_unrearmed_dismissal": (
            lambda: alerts._has_unrearmed_dismissal(_BrokenConn(), "X", "k"), False),
        "alerts.open_alerts": (lambda: alerts.open_alerts(_BrokenConn()), []),
        "alerts.resolve_alert": (lambda: alerts.resolve_alert(_BrokenConn(), 1, "done"), False),
        "flows.load_flows": (lambda: flows.load_flows(_BrokenConn()), []),
        "news_pillar._read_cache": (_read_cache_broken, {}),
        "news_pillar.model_available": (_model_available_broken, False),
        "news_pillar._parse_published": (lambda: news_pillar._parse_published("bad"), None),
        "sizing.is_untouchable": (lambda: sizing.is_untouchable("bad"), True),
        "sizing.sell_amount": (lambda: sizing.sell_amount("bad", 100.0), None),
        "sizing.passes_fee_hurdle": (lambda: sizing.passes_fee_hurdle("bad", "bad", 1.0), True),
        "valuation.compute_shares": (lambda: valuation.compute_shares("bad", 100.0), None),
        "valuation._lookup": (lambda: valuation._lookup(_BrokenLookup(), "X"), None),
        "valuation._last_sync_date": (lambda: valuation._last_sync_date(_BrokenConn()), None),
        "valuation._meta_get": (lambda: valuation._meta_get(_BrokenConn(), "k"), None),
        "valuation.record_snapshot": (
            lambda: valuation.record_snapshot(_BrokenConn()), 0),
        "account.load_account": (
            lambda: account.load_account("/nonexistent/account.yaml").loaded, False),
        "portfolio.load_portfolio": (
            lambda: portfolio.load_portfolio("/nonexistent/portfolio.csv").empty, True),
        "portfolio.portfolio_issues": (
            lambda: portfolio.portfolio_issues("/nonexistent/portfolio.csv"), []),
        "portfolio.load_broker_data": (
            lambda: portfolio.load_broker_data("/nonexistent/portfolio.csv"), {}),
        "portfolio.get_last_rebalance": (_last_rebalance_broken, None),
        "tier_manager.load_tiers": (
            lambda: tier_manager.load_tiers("/nonexistent/tiers.csv").empty, True),
        "account.write_account_fields": (
            lambda: account.write_account_fields(
                {}, "/nonexistent_dir/x/account.yaml"), False),
        "user_data._copy_if_missing": (
            lambda: user_data._copy_if_missing(
                "/nonexistent/src", "/nonexistent/dst"), False),
        "notify._load_state": (
            lambda: _load_state_missing(), {}),
        "today_list._rows": (
            lambda: today_list._rows("SELECT 1", conn=_BrokenConn()), []),
        "today_list.persist": (_persist_broken, 0),
        "advice_state._insert": (
            lambda: advice_state._insert(_BrokenConn(), {}, date.today()), None),
        "artifacts._analysis_regime": (_analysis_regime_broken, {}),
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
