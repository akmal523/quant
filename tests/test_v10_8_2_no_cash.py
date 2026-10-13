"""test_v10_8_2_no_cash.py — v10.8.7 (R-C1): cash is NOT an entity.

The v10.8.4 "cash is back" ruling is superseded. This file enforces the hard rule
again: no page renders a cash row, a cash target, a cash line or a cash
percentage; the Total value is invested only. It also keeps the v10.8.4 UI smoke
test (no literal None / NaN on any page) and adds the A2 grep-guard: no code reads
``portfolio_total.cash_eur`` (frozen audit data) outside the schema/migration.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")
_QUANT = Path(__file__).resolve().parents[1] / "quant"
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


def test_no_cash_row_or_target_on_portfolio():
    """R-C1: no cash row, no cash target, no cash percentage on Portfolio."""
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page(_PAGES["Portfolio"]).run()
    assert not at.exception, f"Portfolio raised: {at.exception}"
    combined = _text(at) + " " + _frames(at)
    assert not re.search(r"\bCash\b", combined), "cash row/line rendered on Portfolio"
    assert not re.search(r"cash target", combined, re.I), "cash target rendered"
    assert not re.search(r"cash is \d+%", combined, re.I), "cash percentage rendered"


def test_no_cash_eur_reads_outside_schema():
    """A2: no code reads ``portfolio_total.cash_eur`` (frozen audit data).

    The column is kept for audit but must never be read. ``account.py`` (the
    account.yaml field) and ``history.py`` (the legacy portfolio_history column)
    are different concepts and are allowlisted, as is the schema in database.py.
    """
    allow = {"database.py", "account.py", "history.py"}
    read_patterns = [
        re.compile(r"SELECT[^\"'\n]*cash_eur", re.I),
        re.compile(r"cash_eur[^\"'\n]*FROM", re.I),
        re.compile(r"\[\s*[\"']cash_eur[\"']\s*\]"),
        re.compile(r"\.get\(\s*[\"']cash_eur[\"']"),
        re.compile(r"cash_series|cash_from_flows|cash_as_of"),
    ]
    offenders: list[str] = []
    for path in _QUANT.rglob("*.py"):
        if path.name in allow:
            continue
        text = path.read_text(encoding="utf-8")
        for pat in read_patterns:
            if pat.search(text):
                offenders.append(f"{path.relative_to(_QUANT)}: {pat.pattern}")
    assert not offenders, f"cash_eur read outside schema: {offenders}"
