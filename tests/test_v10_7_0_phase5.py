"""
test_v10_7_0_phase5.py — Steps generator, forbidden-token copy, briefing (v10.7.0).

Intent: lock the plain-English UI. The steps generator produces only the five
allowed sources; the forbidden-token test renders every page and the briefing
and fails if any old vocabulary appears.

Invariants: pure steps tests; AppTest renders on the isolated fixture DB.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from quant.engine import steps
from quant.ui import copy as C

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")

_TEXTLIKE = ("markdown", "info", "warning", "error", "caption", "text",
             "title", "header", "subheader")


def _all_text(at: AppTest) -> str:
    chunks: list[str] = []
    for attr in _TEXTLIKE:
        for el in getattr(at, attr, []):
            chunks.append(str(getattr(el, "value", "")))
    return " ".join(chunks)


_PAGE_FILE = {
    C.PAGE_TODAY: "pages/today.py",
    C.PAGE_MONTHLY: "pages/monthly.py",
    C.PAGE_PORTFOLIO: "pages/portfolio.py",
    C.PAGE_EXPLORE: "pages/explore.py",
    C.PAGE_SETTINGS: "pages/settings.py",
}


def _run_page(page: str) -> AppTest:
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page(_PAGE_FILE[page]).run()
    return at


# ── Steps generator (Section 10.5) ────────────────────────────────────────────

def test_open_alert_becomes_a_step():
    built = steps.build_steps(
        date(2026, 10, 1),
        open_alerts=[{"message": "Amazon.com: sell about 75 EUR.",
                      "amount_eur": 75.0, "fee_eur": 1.0}])
    assert built and built[0]["source"] == "alert"


def test_unapproved_after_25th():
    built = steps.build_steps(date(2026, 10, 26), plan=None)
    assert any(s["source"] == "monthly" for s in built)
    # Before the 25th, no monthly step.
    early = steps.build_steps(date(2026, 10, 10), plan=None)
    assert not any(s["source"] == "monthly" for s in early)


def test_approved_plan_not_executed():
    plan = {"budget_eur": 200.0, "execution_date": date(2026, 10, 1)}
    built = steps.build_steps(date(2026, 10, 2), plan=plan, actuals_entered=False)
    assert any(s["source"] == "plan" for s in built)


def test_actuals_missing_after_seven_days():
    plan = {"budget_eur": 200.0, "execution_date": date(2026, 10, 1)}
    built = steps.build_steps(date(2026, 10, 10), plan=plan, actuals_entered=False)
    assert any(s["source"] == "actuals" for s in built)


def test_fortress_gap_suggestion_only_over_ten_points():
    holdings = [{"symbol": "EUNL.DE", "name": "MSCI World", "tier": "FORTRESS",
                 "current_weight": 0.34, "target_weight": 0.50}]
    built = steps.build_steps(date(2026, 10, 1), holdings=holdings)
    assert any(s["source"] == "fortress_gap" for s in built)
    small = [{"symbol": "EUNL.DE", "name": "MSCI World", "tier": "FORTRESS",
              "current_weight": 0.45, "target_weight": 0.50}]
    assert not any(s["source"] == "fortress_gap"
                   for s in steps.build_steps(date(2026, 10, 1), holdings=small))


def test_rejected_actions_explain_small_positions():
    holdings = [{"symbol": "5J50.DE", "name": "Global Aero & Defense",
                 "tier": "ALPHA", "value_eur": 143.0}]
    rejected = steps.rejected_actions(holdings)
    assert rejected and "Do not sell" in rejected[0]


# ── Forbidden-token copy test (Section 11) ────────────────────────────────────

@pytest.mark.parametrize("page", [C.PAGE_TODAY, C.PAGE_MONTHLY, C.PAGE_PORTFOLIO,
                                  C.PAGE_EXPLORE, C.PAGE_SETTINGS])
def test_no_forbidden_tokens_on_any_page(page):
    at = _run_page(page)
    assert not at.exception, f"{page} raised: {at.exception}"
    text = _all_text(at)
    for token in C.FORBIDDEN_TOKENS:
        assert token not in text, f"forbidden token {token!r} rendered on {page}"


def test_no_emoji_in_rendered_pages():
    emoji_re = re.compile(
        "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF\U0000FE0F]")
    for page in _PAGE_FILE:
        text = _all_text(_run_page(page))
        assert not emoji_re.search(text), f"emoji rendered on {page}"


def test_briefing_has_no_forbidden_tokens():
    import pandas as pd

    from quant.portfolio.account import AccountState
    from quant.reporting.briefing import build_briefing_md

    md = build_briefing_md(
        as_of="2026-10-01", version="10.7.0", regime_label="rising",
        regime_prob=0.8, regime_source="HMM", audit_df=pd.DataFrame(),
        account=AccountState("EUR", 0.0, "balanced", True),
        total_value=0.0, pnl_eur=0.0, pnl_pct=0.0, with_news=0, without_news=0,
        latest_bar="2026-10-01", alerts=[], steps=[], money=None)
    for token in C.FORBIDDEN_TOKENS:
        assert token not in md, f"forbidden token {token!r} in the briefing"
    assert "## Alerts" in md
    assert "## Your steps this week" in md
