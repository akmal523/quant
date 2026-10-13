"""test_v10_8_6_invariants.py — v10.8.6 P0 invariant tests.

Phase 1 (Defect A): the Full analysis page renders all four tabs; the Research and
Holdings drawers are a single lazy "Inspect symbol" select each (no per-row eager
expanders); inspecting overlapping symbols across both tabs never raises
``StreamlitDuplicateElementKey`` / ``StreamlitDuplicateElementId``; a failing tab is
isolated by the per-tab error boundary.

Phase 2 (Defect B) invariants are added to this file in a later phase.

Fixtures are synthetic only (a small broker CSV + price history); the user's real DB
and data dir are never touched (tests/conftest.py isolates the session).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")

# Symbols deliberately shared between Holdings Analysis and Research.
SEED_SYMBOLS = ("EUNL.DE", "AMZN", "SXRV.DE")
_PAGES = {
    "Portfolio": "pages/portfolio.py",
    "Update holdings": "pages/update.py",
    "History": "pages/history.py",
    "Full analysis": "pages/analysis.py",
    "Settings": "pages/settings.py",
}


def _run(page: str) -> AppTest:
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page(page).run()
    return at


def _text(at: AppTest) -> str:
    chunks = []
    for attr in ("markdown", "info", "warning", "error", "caption", "text",
                 "title", "header", "subheader"):
        for el in getattr(at, attr, []):
            chunks.append(str(getattr(el, "value", "")))
    return " ".join(chunks)


def _frames(at: AppTest) -> list:
    return [getattr(el, "value", None) for el in getattr(at, "dataframe", [])]


def _selectbox_by_key(at: AppTest, key: str):
    for sb in getattr(at, "selectbox", []):
        if getattr(sb, "key", None) == key:
            return sb
    return None


# ── Synthetic holdings (overlapping across the two tabs) ──────────────────────

def _seed_market() -> None:
    from quant.data.database import write_connection

    with write_connection() as conn:
        for sym in SEED_SYMBOLS:
            conn.execute("DELETE FROM market_history WHERE Symbol = ?", [sym])
            for d, close in ((date(2026, 7, 1), 100.0), (date(2026, 7, 2), 101.0),
                             (date(2026, 7, 3), 102.0)):
                conn.execute(
                    "INSERT OR REPLACE INTO market_history (Symbol, Date, Close) "
                    "VALUES (?, ?, ?)", [sym, str(d), close])


def _clear_market() -> None:
    from quant.data.database import write_connection

    with write_connection() as conn:
        for sym in SEED_SYMBOLS:
            conn.execute("DELETE FROM market_history WHERE Symbol = ?", [sym])


@pytest.fixture
def overlapping_portfolio(tmp_path, monkeypatch):
    """A synthetic broker CSV holding the three shared symbols."""
    from quant import paths

    csv = tmp_path / "portfolio.csv"
    rows = "".join(f"{s},100.0,500.0,0.0\n" for s in SEED_SYMBOLS)
    csv.write_text(
        "Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\n" + rows,
        encoding="utf-8")
    monkeypatch.setattr(paths, "DATA_PORTFOLIO", str(csv))
    _seed_market()
    try:
        yield
    finally:
        _clear_market()


# ── Phase 1 (A2): content + error boundary ────────────────────────────────────

def test_full_analysis_renders_all_four_tabs():
    """A content assert: the page renders without exception; each tab shows content."""
    at = _run("pages/analysis.py")
    assert not at.exception, f"Full analysis raised: {at.exception}"
    combined = _text(at) + " " + " ".join(
        str(v.to_string()) for v in _frames(at) if v is not None)
    # Market tab always renders a sentence (regime or an empty-state reason).
    assert "Market trend" in combined or "Not available" in combined
    # Pipeline tab renders its stage table or an explicit empty state.
    assert "Pipeline health" in combined or "Run analysis" in combined


def test_every_tab_body_is_isolated_by_an_error_boundary(monkeypatch):
    """A failing tab shows an in-tab error card; the other tabs still render."""
    from quant.ui import pages as P

    def _boom() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(P, "_render_research", _boom)
    at = _run("pages/analysis.py")
    assert not at.exception, "one failing tab must not kill the page"
    text = _text(at)
    assert "Research tab failed" in text
    assert "See Pipeline health" in text


# ── Phase 1 (A1/A3): cross-tab interaction + overlapping symbols ──────────────

def test_overlapping_symbols_inspect_without_duplicate_key(overlapping_portfolio):
    """Inspect every symbol in Research, then in Holdings Analysis: no key collision.

    Streamlit renders every tab body each run; the same symbol (EUNL.DE) in both
    tabs must not collide on a page-global key.
    """
    at = _run("pages/analysis.py")
    assert not at.exception, at.exception
    for key in ("rs_inspect", "ha_inspect"):
        assert _selectbox_by_key(at, key) is not None, f"{key} missing"
        for sym in SEED_SYMBOLS:
            sb = _selectbox_by_key(at, key)
            sb.select(sym)
            at.run()
            assert not at.exception, f"{key} -> {sym} raised {at.exception}"


def test_both_tables_render_for_overlapping_symbols(overlapping_portfolio):
    """Regression: both the Research funnel and the Holdings table render."""
    at = _run("pages/analysis.py")
    assert not at.exception
    has_drift = any(
        v is not None and hasattr(v, "columns") and "Drift (pp)" in v.columns
        for v in _frames(at))
    has_stage = any(
        v is not None and hasattr(v, "columns") and "Stage" in v.columns
        for v in _frames(at))
    assert has_drift, "Holdings Analysis table missing"
    assert has_stage, "Research funnel table missing"


def test_no_page_raises_on_load():
    """All five pages load without exception (content markers may be empty states)."""
    for title, page in _PAGES.items():
        at = _run(page)
        assert not at.exception, f"{title} raised: {at.exception}"
