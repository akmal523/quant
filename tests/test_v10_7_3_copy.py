"""
test_v10_7_3_copy.py — UI truth pass copy test (v10.7.3, Part 9.1).

Intent: the surviving old strings from Part 1 are forbidden on every rendered
page, and the NEW strings are present. The exact new strings are also asserted
on the copy module so a future edit cannot silently regress them.

Invariants: tests never touch the network or the real systemd.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from quant.ui import copy as C

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")
_TEXTLIKE = ("markdown", "info", "warning", "error", "caption", "text",
             "title", "header", "subheader", "button")
_PAGE_FILE = {
    C.PAGE_TODAY: "pages/today.py",
    C.PAGE_MONTHLY: "pages/monthly.py",
    C.PAGE_HOLDINGS: "pages/portfolio.py",
    C.PAGE_FIND: "pages/explore.py",
}


def _all_text(at: AppTest) -> str:
    chunks: list[str] = []
    for attr in _TEXTLIKE:
        for el in getattr(at, attr, []):
            # Buttons expose .label (and a boolean .value); text elements expose
            # .value. Prefer the label when present.
            val = getattr(el, "label", None)
            if val is None:
                val = getattr(el, "value", "")
            chunks.append(str(val))
    return " ".join(chunks)


def _render(page: str) -> str:
    at = AppTest.from_file(DASHBOARD, default_timeout=90)
    at.run()
    at.switch_page(_PAGE_FILE[page]).run()
    assert not at.exception, f"{page} raised: {at.exception}"
    return _all_text(at)


# ── Forbidden old strings (Part 1) ────────────────────────────────────────────

@pytest.mark.parametrize("page", [C.PAGE_TODAY, C.PAGE_MONTHLY,
                                  C.PAGE_HOLDINGS, C.PAGE_FIND])
def test_forbidden_old_strings_absent(page):
    text = _render(page)
    for token in C.FORBIDDEN_TOKENS:
        assert token not in text, f"forbidden token {token!r} on {page}"


# ── Required new strings (exact constants) ────────────────────────────────────

def test_required_new_strings_exact():
    assert C.HEADER_REVIEW_PREPARED.format(prepared="Friday 2 Oct 2026") == \
        "Report as of Friday 2 Oct 2026."
    assert C.MARKET_REGIME_LINE.format(label="rising", confidence="high") == \
        ("Market is rising, high confidence. This affects only the Active part; "
         "the Long-term part is untouched.")
    assert C.HELP_BROKER_VALUES == (
        "Values come from your broker. Between syncs we show estimates from known "
        "shares and the latest price, labeled estimated.")
    assert C.SEC_BROKER_REFERENCE == "Broker reference"
    assert C.MONTHLY_LEG_LINE.format(
        amount="140", name="Global Aero & Defense", symbol="5J50.DE",
        kind=C.monthly_leg_kind("long_term"), fee="0") == (
        "140 EUR to Global Aero & Defense (5J50.DE), Long-term (never sell), "
        "via savings plan. Fee 0 EUR.")
    assert C.MONTHLY_CASH_LEG_LINE.format(amount="60") == "Keep 60 EUR in cash."
    assert C.NOTHING_REJECTED == "No considered actions were rejected this week."
    assert C.BTN_SAVE_AND_REVIEW == "Save and run review"
    assert C.BTN_SAVE_AND_REVIEW_HELP == \
        "Saves, then recomputes scores and advice now."
    assert C.BTN_SAVE_ONLY_HELP == \
        "Saves without recomputing; the evening run will pick it up."
    assert C.EMERGENCY_HINT == \
        "Enter an amount to see the order in which positions would be sold."
    assert C.SCORES_CAPTION == (
        "Structure and Tactics are scores from 0 to 100. Structure is fundamental "
        "quality; Tactics is timing.")
    assert C.SEC_BROKER_STATEMENT == "Broker statement (editable)"
    assert C.BROKER_STATEMENT_CAPTION == (
        "This is what your broker reported at the last sync. Edit only after "
        "exporting fresh values from Trade Republic.")


# ── Required new strings present on rendered pages ────────────────────────────

def test_overview_renders_new_strings():
    text = _render(C.PAGE_TODAY)
    assert C.NOTHING_REJECTED in text
    assert C.SEC_STEPS in text
    assert C.SEC_NOT_THIS_WEEK in text


def test_holdings_renders_new_strings():
    text = _render(C.PAGE_HOLDINGS)
    assert C.HELP_BROKER_VALUES in text
    # The expander label is not part of the text harness; its caption is.
    assert C.BROKER_STATEMENT_CAPTION in text
    assert C.BTN_SAVE_AND_REVIEW in text
    assert C.BTN_SAVE_AND_REVIEW_HELP in text
    assert C.BTN_SAVE_ONLY_HELP in text
    assert C.EMERGENCY_HINT in text


def test_monthly_renders_new_strings():
    text = _render(C.PAGE_MONTHLY)
    assert C.MONTHLY_STATUS_NOT_APPROVED in text
    assert C.MONTHLY_SPLIT_HEADER in text
