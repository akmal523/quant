"""
test_ui_copy.py — Banned-token contract (v10.5.1, spec 9).

Renders all four pages via AppTest on the isolated fixture DB and asserts the
rendered text contains none of the banned tokens, and that catalog empty states
appear in their scenarios.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from quant.ui import copy as C  # noqa: E402

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")

# Internal identifiers that must never reach the UI (P2/P3).
BANNED = [
    r"run_\d", r"run_latest", r"\.duckdb", r"/home/", r"exit(ed)? \d",
    r"\(source:", r"INFO\]", r"\bn/a\b", r"\bunknown\b",
    r"quant_update", r"\bquant run\b", r"WATCHLIST", r"\bPLAIN\b",
    r"factor_scores", r"advice below", r"Fix in Portfolio",
    r"\(\w+\) \(\w+\)", r"\u00b7\s*$", r"·\s*$",
    # H3.5 (F4): the dotted-ticker double-paren shape ("Global Aero & Def
    # (5J50) (5J50.DE)") the original pattern missed.
    r"\(\w+\) \([\w.]+\)",
    # H3.8 (M9): the non-catalogue as-of preamble; the only as-of string is
    # "Scores as of {date}."
    r"From the review of",
]

_TEXTLIKE = ("markdown", "info", "warning", "error", "caption", "text",
             "title", "header", "subheader")


def _all_text(at: AppTest) -> str:
    chunks: list[str] = []
    for attr in _TEXTLIKE:
        for el in getattr(at, attr, []):
            chunks.append(str(getattr(el, "value", "")))
    return " ".join(chunks)


# v10.8.2 (section 4): the final five pages.
_PAGES = {
    "Portfolio": "pages/portfolio.py",
    "Update holdings": "pages/update.py",
    "History": "pages/history.py",
    "Full analysis": "pages/analysis.py",
    "Settings": "pages/settings.py",
}


def _run_page(page: str) -> AppTest:
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page(_PAGES[page]).run()
    return at


@pytest.mark.parametrize("page", list(_PAGES))
def test_no_banned_tokens_on_any_page(page):
    at = _run_page(page)
    assert not at.exception, f"{page} raised: {at.exception}"
    text = _all_text(at)
    for pat in BANNED:
        assert not re.search(pat, text, re.IGNORECASE), \
            f"banned token {pat!r} rendered on {page}"


def test_home_shows_value_and_what_to_do():
    at = _run_page("Portfolio")
    text = _all_text(at)
    assert C.SEC_WHAT_TO_DO in text
    assert "Nothing to do" in text or "Buy" in text or "Sell" in text


def test_settings_shows_data_folder():
    at = _run_page("Settings")
    text = _all_text(at)
    assert "Your data folder" in text


def test_navigation_order_and_labels():
    dash = (Path(__file__).resolve().parents[1] / "quant" / "dashboard.py").read_text()
    order = [dash.index(f'"{path}"') for path in _PAGES.values()]
    assert order == sorted(order)
    for title in _PAGES:
        assert title in dash


def test_update_page_links_to_portfolio():
    pages = (Path(__file__).resolve().parents[1] / "quant" / "ui" / "pages.py").read_text()
    assert 'st.switch_page("pages/portfolio.py")' in pages
    assert 'st.switch_page("pages/update.py")' in pages
