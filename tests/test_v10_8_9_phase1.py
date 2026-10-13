"""test_v10_8_9_phase1.py — v10.8.9 Phase 1: T1 (column contract product-wide).

Ruling R-F1: the column-spec contract in ``quant/ui/columns.py`` governs EVERY
table: non-empty headers, currency in the header for monetary columns, a declared
column budget per table (<= 8), and low-priority content in the drawer/tooltip.
No table may overflow the default container.

The global AppTest table audit below is the PERMANENT guard against overflow
regressions: for every page it collects all rendered dataframes and asserts the
header/currency/budget invariants.

Synthetic fixtures only; the user's real DB and data dir are never touched
(tests/conftest.py isolates the session in an ephemeral DuckDB file).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant.ui import columns as COLS

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")

# The Research funnel target set (owner-confirmed, v10.8.9).
RESEARCH_TARGET = ["Symbol", "Class", "Stage", "Score", "Confidence", "News",
                   "Now / Target"]

# The audit's monetary heuristic: a header naming a money quantity must carry a
# unit token (EUR / ccy / %). Percent headers are unit-bearing, not money.
_MONEY_WORDS = ("amount", "value", "profit", "suggested", "cost", "live",
                "typed", "realized", "price", "plan", "delta")
_UNIT_TOKENS = ("EUR", "ccy", "%", "$", "GBP")

_PAGES = {
    "Portfolio": "pages/portfolio.py",
    "Update holdings": "pages/update.py",
    "History": "pages/history.py",
    "Full analysis": "pages/analysis.py",
    "Settings": "pages/settings.py",
}

# 4 held + 11 watched = the 15-row acceptance universe. Unique symbols so the
# test never collides with (or depends on) another test's registry rows.
HELD = ("Z9H0.DE", "Z9H1.DE", "Z9H2.DE", "Z9H3.DE")
WATCHED = tuple(f"Z9W{i:02d}.DE" for i in range(11))


def _is_monetary(header: str) -> bool:
    """True when a header names a money quantity (the audit's heuristic)."""
    h = header.lower()
    return any(w in h for w in _MONEY_WORDS)


def _carries_unit(header: str) -> bool:
    """True when a header carries a currency/unit note."""
    return any(t in header for t in _UNIT_TOKENS)


# ── Pure: the product-wide spec invariants (R-F1) ─────────────────────────────

def test_every_spec_header_is_non_empty():
    """R-F1: no table may declare an empty header."""
    for spec in COLS.TABLE_SPECS:
        assert spec.columns, f"{spec.name} has no columns"
        for c in spec.columns:
            assert c.header.strip(), f"{spec.name}.{c.name} has an empty header"


def test_every_monetary_column_carries_its_currency():
    """R-F1: a monetary column carries its currency note in the header."""
    for spec in COLS.TABLE_SPECS:
        for c in spec.columns:
            if c.currency:
                assert c.currency in c.header, \
                    (f"{spec.name}.{c.name}: currency {c.currency!r} not in "
                     f"header {c.header!r}")


def test_every_table_budget_is_at_most_eight():
    """R-F1: no table declares more columns than its budget, and budget <= 8."""
    for spec in COLS.TABLE_SPECS:
        assert len(spec.columns) <= spec.budget <= 8, \
            f"{spec.name}: {len(spec.columns)} columns vs budget {spec.budget}"


def test_registry_covers_the_five_pages_tables():
    """R-F1: every product table is registered (a missing one is a defect)."""
    ids = {t.name for t in COLS.TABLE_SPECS}
    assert {"editor", "entry_check", "research", "holdings_analysis",
            "invest_class", "invest_instrument", "home_holdings"} <= ids


def test_audit_heuristic_sees_every_money_column():
    """Self-test: the audit detects EUR/ccy money columns and their units."""
    for spec in COLS.TABLE_SPECS:
        for c in spec.columns:
            if c.currency in ("EUR", "paid ccy"):
                assert _is_monetary(c.header), f"{c.header!r} not seen as monetary"
                assert _carries_unit(c.header), f"{c.header!r} lacks a unit"


# ── Pure: the Research funnel spec (F1) ───────────────────────────────────────

def test_research_spec_headers_equal_the_target_set():
    """F1: the Research table is exactly the owner target set, in order."""
    assert COLS.table_headers("research") == RESEARCH_TARGET


def test_research_table_has_no_component_detail_columns():
    """F1: component detail lives in the drawer, never as a table column."""
    headers = COLS.table_headers("research")
    assert "Components" not in headers
    assert "Reason" not in headers


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _seed_universe() -> None:
    from quant.data.database import write_connection

    with write_connection() as conn:
        for i, sym in enumerate(HELD + WATCHED):
            comp = 72.0 if i < len(HELD) else 5.0
            conf = "high" if i < len(HELD) else "low"
            conn.execute("DELETE FROM asset_registry WHERE symbol = ?", [sym])
            conn.execute(
                "INSERT INTO asset_registry (symbol, instrument_class, "
                "universe_status) VALUES (?, ?, ?)", [sym, "EQUITY", "WATCHLIST"])
            conn.execute("DELETE FROM scores WHERE symbol = ?", [sym])
            conn.execute(
                "INSERT INTO scores (symbol, composite, confidence, "
                "structural_grade, tactical_grade, present_components, "
                "total_components) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [sym, comp, conf, 60.0, 50.0, 2, 4])


def _clear_universe() -> None:
    from quant.data.database import write_connection

    with write_connection() as conn:
        for sym in HELD + WATCHED:
            conn.execute("DELETE FROM asset_registry WHERE symbol = ?", [sym])
            conn.execute("DELETE FROM scores WHERE symbol = ?", [sym])


@pytest.fixture
def seeded_universe():
    """A 15-symbol universe (4 would-be held + 11 watched), removed after."""
    _seed_universe()
    try:
        yield
    finally:
        _clear_universe()


# ── Acceptance: the 15-row Research universe (F1) ─────────────────────────────

def _held_holdings() -> list[dict]:
    return [{"symbol": s, "name": s, "tier": "ALPHA", "current_weight": 0.1,
             "target_weight": 0.2} for s in HELD]


def test_research_15_row_universe_contract(seeded_universe):
    """Acceptance: a 15-row universe (4 held + 11 watched) fits the spec."""
    from quant.ui import pages

    rows = pages._research_rows(_held_holdings())
    by_sym = {r["_sym"]: r for r in rows}
    # All 15 seeded symbols are present (the DB may hold unrelated rows too).
    assert set(HELD + WATCHED) <= set(by_sym)
    for sym in HELD:
        assert by_sym[sym]["Stage"] == "Held"
        assert " / " in by_sym[sym]["Now / Target"]
    for sym in WATCHED:
        assert by_sym[sym]["Stage"] != "Held"
        assert by_sym[sym]["Now / Target"] == "-"
    df = pd.DataFrame(rows, columns=COLS.table_column_order("research"))
    assert list(df.columns) == RESEARCH_TARGET
    assert len(df.columns) == 7 <= COLS.column_budget("research")
    assert len(df) >= 15
    # Every cell of every declared column has a value (no truncation source).
    for col in RESEARCH_TARGET:
        assert all(str(v).strip() for v in df[col]), f"empty cell in {col}"


# ── The global AppTest table audit (the permanent overflow guard) ─────────────

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402


def _rendered_labels(el) -> list[str]:
    """The visible column labels of a rendered dataframe (index excluded)."""
    try:
        cfg = json.loads(el.proto.columns)
        labels = [str(v.get("label", "")) for k, v in cfg.items() if k != "_index"]
        if labels:
            return labels
    except Exception:  # noqa: BLE001
        pass
    val = getattr(el, "value", None)
    if val is None:
        return []
    return [str(c) for c in val.columns]


def _run(page: str) -> AppTest:
    at = AppTest.from_file(DASHBOARD, default_timeout=90)
    at.run()
    at.switch_page(page).run()
    return at


@pytest.mark.parametrize("page", list(_PAGES))
def test_rendered_tables_respect_the_contract(page):
    """Acceptance: on every page, every rendered table satisfies R-F1."""
    at = _run(_PAGES[page])
    assert not at.exception, f"{page} raised: {at.exception}"
    for el in at.dataframe:
        labels = _rendered_labels(el)
        assert labels, f"a table on {page} rendered no headers"
        for lbl in labels:
            assert lbl.strip(), f"an empty header rendered on {page}"
        assert len(labels) <= COLS.max_budget(), \
            f"{page}: {len(labels)} columns exceeds the budget"
        for lbl in labels:
            if _is_monetary(lbl):
                assert _carries_unit(lbl), \
                    f"{page}: monetary header {lbl!r} lacks a currency note"


def test_research_table_renders_the_contract_headers(seeded_universe, monkeypatch):
    """Acceptance: the Research table renders the target headers at width."""
    from quant.ui import pages

    # A deterministic 15 watched rows (no held dependency on the holdings view).
    monkeypatch.setattr(pages, "_holdings", lambda: [])
    at = AppTest.from_file(DASHBOARD, default_timeout=90)
    at.run()
    at.switch_page("pages/analysis.py").run()
    assert not at.exception, f"Full analysis raised: {at.exception}"
    research = [el for el in at.dataframe if _rendered_labels(el) == RESEARCH_TARGET]
    assert research, "the Research table did not render its contract headers"
    labels = _rendered_labels(research[0])
    assert len(labels) == len(RESEARCH_TARGET) <= COLS.column_budget("research")
    rendered = research[0].value
    assert len(rendered) >= 15
    assert rendered.columns.tolist() == RESEARCH_TARGET
