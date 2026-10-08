"""fallback_registry.py — the declared fallback registry (v10.7.5, Part 3).

A LEGITIMATE FALLBACK is declared here, marked, tested, and surfaced to the user.
A STUB is undeclared, silent, untested, or returns success while computing
nothing. The static auditor (``scripts/audit_stubs.py``) cross-checks AST
except-paths in Tier S modules against this map: an except-path that returns a
default and is not listed here is a blocking finding.

Each entry: ``failure`` (the failure mode), ``default`` (the value returned),
``marker`` (the data-quality value or warning string, or "none" for a pure input
coercion), and ``test`` (the id of the test that triggers it).

The registry test (``tests/test_fallback_registry.py``) iterates this map, so a
new entry without a triggering test fails CI automatically.
"""
from __future__ import annotations

from typing import Any


def _e(failure: str, default: Any, marker: str, test: str) -> dict:
    return {"failure": failure, "default": default, "marker": marker, "test": test}


FALLBACKS: dict[str, dict] = {
    # ── scoring ───────────────────────────────────────────────────────────────
    "scoring._clamp": _e("non-numeric conviction input", 0.0, "none",
                         "test_fallback_scoring_clamp"),
    # ── advice ────────────────────────────────────────────────────────────────
    "advice._spec_change": _e("non-numeric pnl_pct", None, "none",
                              "test_fallback_advice_spec_change"),
    "advice._tier_lookup": _e("unreadable tiers frame", {}, "empty",
                              "test_fallback_advice_tier_lookup"),
    "advice._as_date": _e("unparseable date", None, "none",
                          "test_fallback_advice_as_date"),
    # ── alerts ────────────────────────────────────────────────────────────────
    "alerts._num": _e("non-numeric value", None, "none",
                      "test_fallback_alerts_num"),
    "alerts.has_open_alert": _e("unreadable alerts table", False, "none",
                                "test_fallback_alerts_has_open_alert"),
    "alerts._has_unrearmed_dismissal": _e("unreadable alerts table", False, "none",
                                          "test_fallback_alerts_unrearmed"),
    "alerts.open_alerts": _e("unreadable alerts table", [], "empty",
                             "test_fallback_alerts_open_alerts"),
    "alerts.resolve_alert": _e("unwritable alerts table", False, "none",
                               "test_fallback_alerts_resolve_alert"),
    "alerts.score_resolved_alerts": _e("unreadable alerts table", 0, "none",
                                       "test_fallback_alerts_score_resolved"),
    # ── flows ─────────────────────────────────────────────────────────────────
    "flows.load_flows": _e("unreadable flows table", [], "empty",
                           "test_fallback_flows_load_flows"),
    # ── news_pillar ───────────────────────────────────────────────────────────
    "news_pillar._read_cache": _e("unreadable news cache", {}, "empty",
                                  "test_fallback_news_pillar_read_cache"),
    "news_pillar.model_available": _e("torch unavailable", False, "none",
                                      "test_fallback_news_pillar_model_available"),
    "news_pillar._parse_published": _e("unparseable published date", None, "none",
                                       "test_fallback_news_pillar_parse_published"),
    # ── sizing ────────────────────────────────────────────────────────────────
    "sizing.is_untouchable": _e("non-numeric position value", True, "none",
                                "test_fallback_sizing_is_untouchable"),
    "sizing.sell_amount": _e("non-numeric drift or value", None, "none",
                             "test_fallback_sizing_sell_amount"),
    "sizing.passes_fee_hurdle": _e("non-numeric alpha or amount", True, "none",
                                   "test_fallback_sizing_passes_fee_hurdle"),
    # ── valuation ─────────────────────────────────────────────────────────────
    "valuation.compute_shares": _e("non-numeric value or price", None, "none",
                                   "test_fallback_valuation_compute_shares"),
    "valuation._lookup": _e("price lookup raised", None, "none",
                            "test_fallback_valuation_lookup"),
    "valuation.sync_from_portfolio_csv": _e("portfolio CSV unreadable", 0, "none",
                                            "test_fallback_valuation_sync_csv"),
    "valuation._last_sync_date": _e("unreadable holdings_meta", None, "none",
                                    "test_fallback_valuation_last_sync_date"),
    "valuation._meta_get": _e("unreadable meta table", None, "none",
                              "test_fallback_valuation_meta_get"),
    "valuation.record_snapshot": _e("unreadable holdings_meta", 0, "none",
                                    "test_fallback_valuation_record_snapshot"),
    # ── account ───────────────────────────────────────────────────────────────
    "account.load_account": _e("missing or invalid account.yaml",
                               "AccountState(EUR, None, balanced, loaded=False)",
                               "fallback", "test_fallback_account_load_account"),
    "account.write_account_fields": _e("unwritable account.yaml", False, "none",
                                       "test_fallback_account_write_fields"),
    # ── cash_rate ─────────────────────────────────────────────────────────────
    "cash_rate.fetch_live_cash_apy": _e("live fetch failed", None, "fallback",
                                        "test_fallback_cash_rate_fetch"),
    # ── optimizer ─────────────────────────────────────────────────────────────
    "optimizer.optimize_portfolio": _e("solver failed", "equal weight within caps",
                                       "optimizer fallback: equal weight",
                                       "test_fallback_optimizer_portfolio"),
    "optimizer.optimize_portfolio_cvar": _e("solver failed",
                                            "equal weight within caps",
                                            "optimizer fallback: equal weight",
                                            "test_fallback_optimizer_cvar"),
    # ── portfolio ─────────────────────────────────────────────────────────────
    "portfolio.load_portfolio": _e("portfolio CSV unreadable", "empty frame",
                                   "empty", "test_fallback_portfolio_load"),
    "portfolio.portfolio_issues": _e("portfolio CSV unreadable", [], "empty",
                                     "test_fallback_portfolio_issues"),
    "portfolio.load_broker_data": _e("portfolio CSV unreadable", {}, "empty",
                                     "test_fallback_portfolio_broker_data"),
    "portfolio.get_last_rebalance": _e("unreadable rebalance_log", None, "none",
                                       "test_fallback_portfolio_last_rebalance"),
    # ── risk ──────────────────────────────────────────────────────────────────
    "risk.estimate_spread_bps": _e("unusable price history", 50.0, "fallback",
                                   "test_fallback_risk_estimate_spread"),
    "risk.calculate_liquidity_score": _e("unusable price history", 0.0, "fallback",
                                         "test_fallback_risk_liquidity_score"),
    # ── tier_manager ──────────────────────────────────────────────────────────
    "tier_manager.load_tiers": _e("tiers.csv unreadable", "empty frame", "empty",
                                  "test_fallback_tier_manager_load_tiers"),
    # ── persistence (v10.8.1) ─────────────────────────────────────────────────
    "user_data._copy_if_missing": _e("missing or unreadable source file", False,
                                     "none", "test_fallback_user_data_copy_if_missing"),
    "recovery._meta_rows": _e("unreadable holdings_meta", [], "empty",
                              "test_fallback_recovery_meta_rows"),
    "render._rows_differ": _e("uncomparable table frames", False, "none",
                              "test_fallback_render_rows_differ"),
    "notify._load_state": _e("missing or invalid notify state", {}, "empty",
                             "test_fallback_notify_load_state"),
}


def register_fallback(name: str, failure: str, default: Any, marker: str) -> None:
    """Record a fallback at runtime (idempotent). Never raises.

    New code should call this from its except-path so the fallback is declared
    even before the static registry is updated.
    """
    FALLBACKS.setdefault(name, {
        "failure": failure, "default": default, "marker": marker, "test": "",
    })


def is_registered(name: str) -> bool:
    """True when a qualified ``module.function`` name is in the registry."""
    return name in FALLBACKS
