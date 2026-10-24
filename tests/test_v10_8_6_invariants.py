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

from tests import store_readers

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
    assert "The Research tab could not load" in text
    assert "see Pipeline health" in text


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
    from quant.ui import columns as COLS

    has_stage = any(
        v is not None and hasattr(v, "columns")
        and list(v.columns) == COLS.table_column_order("research")
        for v in _frames(at))
    assert has_drift, "Holdings Analysis table missing"
    assert has_stage, "Research funnel table missing"


def test_no_page_raises_on_load():
    """All five pages load without exception (content markers may be empty states)."""
    for title, page in _PAGES.items():
        at = _run(page)
        assert not at.exception, f"{title} raised: {at.exception}"


# ── Phase 2 (Defect B): valuation-store invariants on a synthetic ledger ──────
# A small ledger with an opening lot, a BUY, a SELL and a SPLIT, plus a price for
# every synthetic trading day. Never the user's real data.

V_DAYS = [date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 5),
          date(2026, 10, 6), date(2026, 10, 7), date(2026, 10, 8),
          date(2026, 10, 9), date(2026, 10, 12), date(2026, 10, 13)]
_V_SYMBOLS = ("EUNL.DE", "AMZN")


@pytest.fixture
def valuation_store():
    """Seed holdings_meta + trades + splits + prices, then backfill the store."""
    from quant.data.database import write_connection
    from quant.engine import mtm

    with write_connection() as conn:
        # Hermetic: clear the ledger inputs so backfill sees ONLY this fixture's
        # symbols (a leftover holdings_meta row from another test would otherwise
        # be valued here too).
        conn.execute("DELETE FROM valuation")
        conn.execute("DELETE FROM portfolio_total")
        conn.execute("DELETE FROM holdings_meta")
        conn.execute("DELETE FROM trades")
        conn.execute("DELETE FROM splits")
        conn.execute("DELETE FROM flows")
        for sym in _V_SYMBOLS:
            conn.execute("DELETE FROM market_history WHERE Symbol = ?", [sym])
        conn.execute(
            "INSERT INTO holdings_meta (symbol, shares, sync_date, invested_at_sync) "
            "VALUES ('EUNL.DE', 10.0, ?, 1000.0)", [date(2026, 10, 1)])
        conn.execute(
            "INSERT INTO holdings_meta (symbol, shares, sync_date, invested_at_sync) "
            "VALUES ('AMZN', 4.0, ?, 200.0)", [date(2026, 10, 1)])
        conn.execute(
            "INSERT INTO trades (date, symbol, action, shares) "
            "VALUES (?, 'EUNL.DE', 'buy', 5.0)", [date(2026, 10, 5)])
        conn.execute(
            "INSERT INTO trades (date, symbol, action, shares) "
            "VALUES (?, 'EUNL.DE', 'sell', 3.0)", [date(2026, 10, 8)])
        conn.execute(
            "INSERT INTO splits (symbol, effective_date, factor) "
            "VALUES ('EUNL.DE', ?, 2.0)", [date(2026, 10, 10)])
        for i, d in enumerate(V_DAYS):
            conn.execute(
                "INSERT OR REPLACE INTO market_history (Symbol, Date, Close) "
                "VALUES ('EUNL.DE', ?, ?)", [str(d), 100.0 + i])
            conn.execute(
                "INSERT OR REPLACE INTO market_history (Symbol, Date, Close) "
                "VALUES ('AMZN', ?, ?)", [str(d), 50.0 + i])
        mtm.backfill(conn, fx_lookup=lambda _s: 1.0, source="test")


def test_invariant_tail_matches_latest_close(valuation_store):
    """Invariant 1: the valuation tail equals the latest close date."""
    from quant.engine import mtm

    assert mtm.latest_price_date() == store_readers.latest_market_date()


def test_invariant_one_point_per_trading_day(valuation_store):
    """Invariant 2: one valuation point per trading day, no gaps, no flat fill."""
    from quant.engine import mtm

    series = mtm.read_series(symbol="EUNL.DE", start=date(2026, 10, 1),
                             end=date(2026, 10, 13))
    got = {str(r["date"])[:10] for r in series}
    expected = {str(d) for d in V_DAYS}
    assert got == expected, f"missing/extra days: {expected ^ got}"
    assert len(series) == len(V_DAYS)


def test_invariant_day_delta_equals_valuation_difference(valuation_store):
    """Invariant 3: the header delta == total(D) - total(D-1); one source."""
    from quant.engine import mtm
    from quant.engine.mtm import read_total_series
    from quant.reporting.artifacts import compute_day_delta

    d_last, d_prev = V_DAYS[-1], V_DAYS[-2]
    expected = store_readers.portfolio_total(d_last) - store_readers.portfolio_total(d_prev)
    eur, _pct, when = mtm.day_delta()
    assert abs(eur - expected) < 1e-9, f"delta {eur} != {expected}"
    assert str(when)[:10] == str(d_last)
    # The pure helper the header is routed through agrees to the cent.
    pure_eur, _pure_pct = compute_day_delta(read_total_series())
    assert abs(pure_eur - eur) < 1e-9


def test_invariant_chart_tail_equals_portfolio_total(valuation_store):
    """Invariant 4: the chart's last plotted value matches the store total.

    v10.8.7 (R-C1): the total is invested only; there is no cash term.
    """
    from quant.engine import mtm
    from quant.reporting.artifacts import read_portfolio_total

    d = mtm.latest_price_date()
    chart = read_portfolio_total()
    chart_tail = float(chart["value_eur"].iloc[-1])
    assert abs(chart_tail - store_readers.portfolio_total(d)) < 1e-9


def test_invariant_growth_view_has_portfolio_and_per_asset(valuation_store):
    """Invariant 5: Growth has the portfolio + one line per held asset at 100."""
    from quant.engine import mtm
    from quant.reporting.artifacts import compute_growth_view

    d = mtm.latest_price_date()
    symbols = mtm.held_symbols(as_of=d)
    assert set(symbols) == set(_V_SYMBOLS)
    growth = compute_growth_view(mtm.read_total_series(), mtm.read_symbol_series(),
                                 symbols)
    assert "portfolio" in growth
    for sym in symbols:
        assert sym in growth, f"missing per-asset line for {sym}"
    for key, series in growth.items():
        assert series, f"{key} has no points"
        assert abs(series[0].value - 100.0) < 1e-9, f"{key} not normalized to 100"


def test_invariant_value_view_hover_lists_per_asset(valuation_store):
    """Invariant 6: the Value hover lists total + each held asset, summing.

    v10.8.7 (R-C1): there is no cash key; the total is the invested sum.
    """
    from quant.engine import mtm
    from quant.reporting.artifacts import compute_value_view_hover

    d = mtm.latest_price_date()
    symbols = mtm.held_symbols(as_of=d)
    hover = compute_value_view_hover(d, mtm.read_total_series(),
                                     mtm.read_symbol_series(), symbols)
    assert "total" in hover and "cash" not in hover
    for sym in symbols:
        assert sym in hover, f"missing per-asset value for {sym} in hover"
    expected = sum(hover[sym] for sym in symbols)
    assert abs(expected - hover["total"]) < 1e-9


def test_header_delta_smoke_matches_pure_helper(valuation_store):
    """Clarification 3: the header number equals compute_day_delta output."""
    from quant.engine import mtm
    from quant.reporting.artifacts import compute_day_delta

    eur, pct, _when = mtm.day_delta()
    pure_eur, pure_pct = compute_day_delta(mtm.read_total_series())
    assert abs(eur - pure_eur) < 1e-9 and abs(pct - pure_pct) < 1e-12


# ── Catch-up (Clarification 2): tail-equal + lock-held retry ──────────────────

def test_catch_up_aligns_tail_to_latest_price(valuation_store):
    """A valuation tail behind the price tail is caught up to the latest close."""
    from datetime import date as _d

    from quant.data.database import write_connection
    from quant.engine import mtm

    with write_connection() as conn:
        conn.execute("DELETE FROM valuation WHERE date > ?", [_d(2026, 10, 8)])
        conn.execute("DELETE FROM portfolio_total WHERE date > ?", [_d(2026, 10, 8)])
    assert mtm.latest_price_date() == _d(2026, 10, 8)
    res = mtm.catch_up(retries=1, base_delay=0.0)
    assert res["ok"] is True
    assert mtm.latest_price_date() == store_readers.latest_market_date()


def test_catch_up_retries_then_marks_stale(valuation_store, monkeypatch):
    """On a held lock: bounded retry, one stale mark, no raise, no spam."""
    from datetime import date as _d

    from quant.data.database import write_connection
    from quant.engine import mtm

    with write_connection() as conn:
        conn.execute("DELETE FROM valuation WHERE date > ?", [_d(2026, 10, 8)])
        conn.execute("DELETE FROM portfolio_total WHERE date > ?", [_d(2026, 10, 8)])
    calls = {"backfill": 0, "stale": 0}

    def _locked(*_a, **_k):
        calls["backfill"] += 1
        raise RuntimeError("database is locked")

    monkeypatch.setattr(mtm, "backfill", _locked)
    monkeypatch.setattr(mtm, "_mark_valuation_stale",
                        lambda _e: calls.__setitem__("stale", calls["stale"] + 1))
    res = mtm.catch_up(retries=3, base_delay=0.0)
    assert res["ok"] is False
    assert calls["backfill"] == 3, "must retry the bounded number of times"
    assert calls["stale"] == 1, "must mark stale exactly once"
