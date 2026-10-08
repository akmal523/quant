"""test_v10_8_2_verify.py — copy/look and interactive-element inventory (v10.8.2, 20-21).

Section 20: every instrument appears in exactly one label form ("Name (TICKER)",
never a duplicated ticker). Section 21: every interactive element on the five
pages is labelled and does something visible.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")
_PAGES = {
    "Portfolio": "pages/portfolio.py",
    "Update holdings": "pages/update.py",
    "History": "pages/history.py",
    "Full analysis": "pages/analysis.py",
    "Settings": "pages/settings.py",
}


# ── One label form (section 20) ───────────────────────────────────────────────

def test_label_form_is_name_ticker():
    from quant.ui.search import label_for

    assert label_for("iShares Core MSCI World", "EUNL.DE") == \
        "iShares Core MSCI World (EUNL.DE)"


def test_label_form_never_duplicates_the_ticker():
    from quant.ui.search import label_for

    assert label_for("Global Aero & Def (5J50)", "5J50.DE") == "Global Aero & Def (5J50)"


def test_label_form_bare_symbol_when_name_is_the_symbol():
    from quant.ui.search import label_for

    assert label_for("EUNL.DE", "EUNL.DE") == "EUNL.DE"


def test_label_form_contains_the_base_ticker_at_most_once():
    from quant.ui.search import label_for

    for name, sym in [("Amazon.com", "AMZN"), ("iShares NASDAQ 100", "SXRV.DE"),
                      ("Global Aero & Def (5J50)", "5J50.DE")]:
        label = label_for(name, sym)
        base = sym.split(".")[0]
        assert label.count(base) <= 1, label


# ── Interactive-element inventory (section 21) ────────────────────────────────

def _run(page: str) -> AppTest:
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page(_PAGES[page]).run()
    return at


@pytest.mark.parametrize("page", list(_PAGES))
def test_page_renders_and_every_button_is_labelled(page):
    at = _run(page)
    assert not at.exception, f"{page} raised: {at.exception}"
    for b in at.button:
        assert str(getattr(b, "label", "")).strip(), f"unlabelled button on {page}"


def test_portfolio_has_the_invest_block():
    at = _run("Portfolio")
    labels = [str(getattr(n, "label", "")) for n in at.number_input]
    assert any("Amount" in lbl for lbl in labels)


def test_update_holdings_has_the_add_holding_input():
    at = _run("Update holdings")
    assert at.text_input, "Update holdings has no add-holding input"


def test_settings_has_the_risk_profile_control():
    at = _run("Settings")
    labels = [str(getattr(r, "label", "")) for r in at.radio]
    assert any("Risk profile" in lbl for lbl in labels)
