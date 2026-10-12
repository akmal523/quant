"""test_v10_8_2_no_cash.py — v10.8.4: cash is back (T3) + the UI smoke test (R2/T7).

The v10.8.2 "no cash in the product" doctrine is superseded (v10.8.4 ruling): cash
is a first-class row on Portfolio and a routing target. This file now enforces the
hard UI rule instead — no page renders a literal None / NaN — and confirms the
cash row is present where the spec requires it.
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


def _frames(at: AppTest) -> str:
    """The rendered dataframe cells, as text (T2/T3 tables)."""
    chunks = []
    for el in getattr(at, "dataframe", []):
        value = getattr(el, "value", None)
        if value is not None:
            try:
                chunks.append(str(value.to_string()))
            except Exception:  # noqa: BLE001
                chunks.append(str(value))
    return " ".join(chunks)


@pytest.mark.parametrize("page", list(_PAGES))
def test_no_literal_none_or_nan_on_any_page(page):
    """R2/T7: a computed field is a value or an explicit empty state, never None."""
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page(_PAGES[page]).run()
    assert not at.exception, f"{page} raised: {at.exception}"
    text = _text(at)
    tables = _frames(at)
    assert not re.search(r"\bNone\b", text + " " + tables), \
        f"literal None rendered on {page}"
    assert not re.search(r"\bNaN\b", text + " " + tables), \
        f"literal NaN rendered on {page}"


def test_cash_row_is_present_on_portfolio():
    """T3: cash is a first-class row on Portfolio (class 'Cash')."""
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page(_PAGES["Portfolio"]).run()
    assert not at.exception, f"Portfolio raised: {at.exception}"
    assert re.search(r"\bCash\b", _text(at) + " " + _frames(at)), \
        "cash row missing on Portfolio"
