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
    r"factor_scores",
]

_TEXTLIKE = ("markdown", "info", "warning", "error", "caption", "text",
             "title", "header", "subheader")


def _all_text(at: AppTest) -> str:
    chunks: list[str] = []
    for attr in _TEXTLIKE:
        for el in getattr(at, attr, []):
            chunks.append(str(getattr(el, "value", "")))
    return " ".join(chunks)


def _run_page(page: str) -> AppTest:
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    return at


@pytest.mark.parametrize("page", [C.PAGE_TODAY, C.PAGE_PORTFOLIO,
                                  C.PAGE_EXPLORE, C.PAGE_SETTINGS])
def test_no_banned_tokens_on_any_page(page):
    at = _run_page(page)
    assert not at.exception, f"{page} raised: {at.exception}"
    text = _all_text(at)
    for pat in BANNED:
        assert not re.search(pat, text, re.IGNORECASE), \
            f"banned token {pat!r} rendered on {page}"


def test_today_shows_portfolio_value_section():
    at = _run_page(C.PAGE_TODAY)
    text = _all_text(at)
    # The Today page always renders the Portfolio value section (chart or the
    # catalog empty state / first-run guide, depending on history).
    assert (C.SEC_PORTFOLIO_VALUE in text) or (C.FIRST_RUN_STEPS[0] in text)


def test_settings_shows_data_status():
    at = _run_page(C.PAGE_SETTINGS)
    text = _all_text(at)
    assert C.SEC_DATA_STATUS in text
    # v10.5.2 A2: an empty fixture DB shows the missing-data empty state; a
    # populated DB shows the full sentence. A placeholder sentence is never shown.
    assert (C.EMPTY_NO_MARKET_DATA in text) or ("instruments, prices through" in text)
