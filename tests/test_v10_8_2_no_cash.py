"""test_v10_8_2_no_cash.py — no user-visible string contains "cash" (section 8).

The owner does not want cash in the product. This renders the five pages and
fails if the word "cash" appears in any user-visible string.
"""
from __future__ import annotations

import re
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
_TEXTLIKE = ("markdown", "info", "warning", "error", "caption", "text",
             "title", "header", "subheader")


def _text(at: AppTest) -> str:
    chunks = []
    for attr in _TEXTLIKE:
        for el in getattr(at, attr, []):
            chunks.append(str(getattr(el, "value", "")))
    return " ".join(chunks)


@pytest.mark.parametrize("page", list(_PAGES))
def test_no_cash_on_any_page(page):
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page(_PAGES[page]).run()
    assert not at.exception, f"{page} raised: {at.exception}"
    assert not re.search(r"\bcash\b", _text(at), re.IGNORECASE), \
        f"'cash' rendered on {page}"
