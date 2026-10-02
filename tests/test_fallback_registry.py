"""test_fallback_registry.py — every declared fallback is triggered and honest.

Intent (v10.7.5, Part 3): a LEGITIMATE FALLBACK is declared, marked, tested, and
surfaced. This test iterates ``FALLBACKS`` so a new entry without a triggering
test fails CI automatically. It also proves cache correctness (a cached result
equals a cold computation) and the optimizer fallback marker.

Invariants:
  - Every registry entry has a trigger and a test id.
  - Each trigger returns the documented default.
  - The optimizer fallback sets a marker the weekly report renders.
  - disk_cache: cold and warm results are identical on fresh inputs.
"""
from __future__ import annotations

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
    from quant.analytics.scoring import calculate_conviction
    from quant.engine import advice, alerts, flows, news_pillar, sizing, valuation
    from quant.portfolio import account, cash_rate, optimizer, portfolio, risk, tier_manager

    def _read_cache_broken():
        with mock.patch.object(news_pillar, "_cache_path", return_value="/nonexistent/x.json"):
            return news_pillar._read_cache()

    def _model_available_broken():
        with mock.patch("importlib.util.find_spec", side_effect=RuntimeError("boom")):
            return news_pillar.model_available()

    def _optimize_broken():
        with mock.patch("cvxpy.Problem.solve", side_effect=RuntimeError("boom")):
            return optimizer.optimize_portfolio(np.array([0.1, 0.2]), np.eye(2))

    def _optimize_cvar_broken():
        with mock.patch("cvxpy.Problem.solve", side_effect=RuntimeError("boom")):
            return optimizer.optimize_portfolio_cvar(np.array([0.1, 0.2]), np.eye(2))

    def _last_rebalance_broken():
        with mock.patch("quant.data.database.get_connection", side_effect=RuntimeError("boom")):
            return portfolio.get_last_rebalance("X")

    return {
        "scoring._clamp": (lambda: calculate_conviction(None, None, None), "LOW"),
        "advice._spec_change": (lambda: advice._spec_change({"pnl_pct": "bad"}), None),
        "advice._tier_lookup": (lambda: advice._tier_lookup("bad"), {}),
        "advice._as_date": (lambda: advice._as_date("not-a-date"), None),
        "alerts._num": (lambda: alerts._num("bad"), None),
        "alerts.has_open_alert": (lambda: alerts.has_open_alert(_BrokenConn(), "X", "k"), False),
        "alerts._has_unrearmed_dismissal": (
            lambda: alerts._has_unrearmed_dismissal(_BrokenConn(), "X", "k"), False),
        "alerts.open_alerts": (lambda: alerts.open_alerts(_BrokenConn()), []),
        "alerts.resolve_alert": (lambda: alerts.resolve_alert(_BrokenConn(), 1, "done"), False),
        "alerts.score_resolved_alerts": (
            lambda: alerts.score_resolved_alerts(_BrokenConn(), {}), 0),
        "flows.load_flows": (lambda: flows.load_flows(_BrokenConn()), []),
        "news_pillar._read_cache": (_read_cache_broken, {}),
        "news_pillar.model_available": (_model_available_broken, False),
        "news_pillar._parse_published": (lambda: news_pillar._parse_published("bad"), None),
        "sizing.is_untouchable": (lambda: sizing.is_untouchable("bad"), True),
        "sizing.sell_amount": (lambda: sizing.sell_amount("bad", 100.0), None),
        "sizing.passes_fee_hurdle": (lambda: sizing.passes_fee_hurdle("bad", "bad", 1.0), True),
        "valuation.compute_shares": (lambda: valuation.compute_shares("bad", 100.0), None),
        "valuation._lookup": (lambda: valuation._lookup(_BrokenLookup(), "X"), None),
        "valuation.sync_from_portfolio_csv": (
            lambda: valuation.sync_from_portfolio_csv(_BrokenConn(), {}), 0),
        "valuation._last_sync_date": (lambda: valuation._last_sync_date(_BrokenConn()), None),
        "valuation._meta_get": (lambda: valuation._meta_get(_BrokenConn(), "k"), None),
        "account.load_account": (
            lambda: account.load_account("/nonexistent/account.yaml").loaded, False),
        "cash_rate.fetch_live_cash_apy": (
            lambda: cash_rate.fetch_live_cash_apy(fetcher=_raise, parser=lambda _h: None), None),
        "optimizer.optimize_portfolio": (
            _optimize_broken, np.full(2, min(0.5, 0.10))),
        "optimizer.optimize_portfolio_cvar": (
            _optimize_cvar_broken, np.full(2, min(0.5, 0.10))),
        "portfolio.load_portfolio": (
            lambda: portfolio.load_portfolio("/nonexistent/portfolio.csv").empty, True),
        "portfolio.portfolio_issues": (
            lambda: portfolio.portfolio_issues("/nonexistent/portfolio.csv"), []),
        "portfolio.load_broker_data": (
            lambda: portfolio.load_broker_data("/nonexistent/portfolio.csv"), {}),
        "portfolio.get_last_rebalance": (_last_rebalance_broken, None),
        "risk.estimate_spread_bps": (lambda: risk.estimate_spread_bps(None), 50.0),
        "risk.calculate_liquidity_score": (
            lambda: risk.calculate_liquidity_score("X", None), 0.0),
        "tier_manager.load_tiers": (
            lambda: tier_manager.load_tiers("/nonexistent/tiers.csv").empty, True),
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


def test_optimizer_fallback_marker_and_report_line():
    from quant.portfolio import optimizer

    optimizer.clear_optimizer_fallback()
    assert optimizer.optimizer_fallback_line() is None
    with mock.patch("cvxpy.Problem.solve", side_effect=RuntimeError("boom")):
        optimizer.optimize_portfolio(np.array([0.1, 0.2]), np.eye(2))
    assert optimizer.optimizer_fallback_used()
    assert optimizer.optimizer_fallback_line() == "optimizer fallback: equal weight"
    optimizer.clear_optimizer_fallback()


def test_disk_cache_cold_equals_warm(tmp_path, monkeypatch):
    from quant.analytics import cache

    monkeypatch.setattr(cache, "cache_dir", lambda: tmp_path)
    calls = {"n": 0}

    @cache.disk_cache(max_age_days=7)
    def _square(x: int) -> int:
        calls["n"] += 1
        return x * x

    cache.clear_cache()
    cold = _square(7)
    warm = _square(7)
    assert cold == warm == 49
    assert calls["n"] == 1  # the warm call was served from disk
