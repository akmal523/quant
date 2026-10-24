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
    # ── flows ─────────────────────────────────────────────────────────────────
    "flows.load_flows": _e("unreadable flows table", [], "empty",
                           "test_fallback_flows_load_flows"),
    # ── sizing ────────────────────────────────────────────────────────────────
    "sizing.is_untouchable": _e("non-numeric position value", True, "none",
                                "test_fallback_sizing_is_untouchable"),
    "sizing.sell_amount": _e("non-numeric drift or value", None, "none",
                             "test_fallback_sizing_sell_amount"),
    # ── account ───────────────────────────────────────────────────────────────
    "account.load_account": _e("missing or invalid account.yaml",
                               "AccountState(EUR, None, balanced, loaded=False)",
                               "fallback", "test_fallback_account_load_account"),
    "account.write_account_fields": _e("unwritable account.yaml", False, "none",
                                       "test_fallback_account_write_fields"),
    # ── portfolio ─────────────────────────────────────────────────────────────
    "portfolio.load_portfolio": _e("portfolio CSV unreadable", "empty frame",
                                   "empty", "test_fallback_portfolio_load"),
    "portfolio.portfolio_issues": _e("portfolio CSV unreadable", [], "empty",
                                     "test_fallback_portfolio_issues"),
    # ── tier_manager ──────────────────────────────────────────────────────────
    "tier_manager.load_tiers": _e("tiers.csv unreadable", "empty frame", "empty",
                                  "test_fallback_tier_manager_load_tiers"),
    # ── persistence (v10.8.1) ─────────────────────────────────────────────────
    "user_data._copy_if_missing": _e("missing or unreadable source file", False,
                                     "none", "test_fallback_user_data_copy_if_missing"),
    # ── the ONE list / regime (v10.8.10) ──────────────────────────────────────
    "today_list._rows": _e("unreadable store", [], "empty",
                           "test_fallback_today_list_rows"),
    "today_list.persist": _e("advice log not writable / inputs unreadable", 0,
                             "none", "test_fallback_today_list_persist"),
    "advice_state._insert": _e("unwritable advice_log", None, "none",
                               "test_fallback_advice_state_insert"),
    "artifacts._analysis_regime": _e("unreadable meta table", {}, "empty",
                                     "test_fallback_artifacts_analysis_regime"),
    "advice_state._as_date": _e("unparseable date", None, "none",
                                "test_fallback_advice_state_as_date"),
    "advice_state.mark_done": _e("unwritable advice_log", False, "none",
                                 "test_fallback_advice_state_mark_done"),
    "advice_state._query": _e("unreadable advice_log", [], "empty",
                              "test_fallback_advice_state_query"),
    # ── analysis chain: a missing component scores SCORE_MISSING_PENALTY and
    #    lowers the confidence tier (the surfaced marker) ──────────────────────
    "analysis._to_date": _e("unparseable date", None, "none",
                            "test_fallback_analysis_to_date"),
    "analysis._technical_from_prices": _e("unreadable market_history", None,
                                          "confidence",
                                          "test_fallback_analysis_technical"),
    "analysis._fundamentals_row": _e("unreadable fundamentals", None, "confidence",
                                     "test_fallback_analysis_fundamentals"),
    "analysis._sentiment_component": _e("unreadable nlp_evidence", None, "confidence",
                                        "test_fallback_analysis_sentiment"),
    # ── valuation store (mtm) ─────────────────────────────────────────────────
    "mtm._as_date": _e("unparseable date", None, "none", "test_fallback_mtm_as_date"),
    "mtm._snapshots": _e("unreadable position_snapshots", "the current table as today's",
                         "none", "test_fallback_mtm_snapshots"),
    "mtm._split_rows": _e("unreadable splits", [], "empty", "test_fallback_mtm_split_rows"),
    "mtm._fx_rate": _e("FX lookup raised", 1.0, "none (currency keeps the last live "
                       "rate, so this only fires on an unexpected error)",
                       "test_fallback_mtm_fx_rate"),
    "mtm._price_history": _e("unreadable market_history", {}, "empty",
                             "test_fallback_mtm_price_history"),
    "mtm._query": _e("unreadable valuation store", [], "empty", "test_fallback_mtm_query"),
}


def register_fallback(name: str, failure: str, default: Any, marker: str) -> None:
    """Record a fallback at runtime (idempotent). Never raises.

    New code should call this from its except-path so the fallback is declared
    even before the static registry is updated.
    """
    FALLBACKS.setdefault(name, {
        "failure": failure, "default": default, "marker": marker, "test": "",
    })
