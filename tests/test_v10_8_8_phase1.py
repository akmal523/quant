"""test_v10_8_8_phase1.py — v10.8.8 Phase 1: T1 (editor column contract + Entry check).

Ruling R-P1: the broker editor grid contains ONLY the two read-only identity
columns (Name, Instrument) and the four typed inputs; derived and live data live
OUTSIDE the grid, in the read-only Entry-check table. Every header is non-empty
and money columns carry their currency; no table exceeds its column budget.

Addendum A1: the Name cell renders the display name WITHOUT any ticker in
parentheses (never the row's Instrument value).
Addendum A2: the Entry-check identity column is "Instrument" (the same broker
ticker as the grid).

Synthetic fixtures only; the user's real DB and data dir are never touched.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant.ui import columns as COLS

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")

# The exact contract (owner-confirmed, v10.8.8).
EDITOR_HEADERS = ["Name", "Instrument", "Avg cost / share (paid ccy)",
                  "Value (EUR)", "Profit (EUR)", "Plan (EUR/m)"]
CHECK_HEADERS = ["Instrument", "Shares (derived)", "Typed value (EUR)",
                 "Live value (EUR)", "Delta %"]


# ── The contract (pure) ───────────────────────────────────────────────────────

def test_editor_contract_headers_exact():
    """R-P1: the editor header set equals the contract, every header non-empty."""
    assert COLS.editor_headers() == EDITOR_HEADERS
    assert all(h.strip() for h in COLS.editor_headers())


def test_check_contract_headers_exact():
    """R-P1/A2: the Entry-check header set equals the contract."""
    assert COLS.check_headers() == CHECK_HEADERS
    assert all(h.strip() for h in COLS.check_headers())


def test_column_budgets():
    """R-P1: editor <= 7 columns, check table <= 5 columns."""
    assert len(COLS.EDITOR_COLUMNS) <= COLS.EDITOR_MAX_COLUMNS <= 7
    assert len(COLS.CHECK_COLUMNS) <= COLS.CHECK_MAX_COLUMNS <= 5


def test_cost_header_carries_its_currency():
    """D3: the cost column header carries its currency note (paid ccy)."""
    spec = next(c for c in COLS.EDITOR_COLUMNS if c.name == "Avg_Entry_Price")
    assert spec.currency == "paid ccy"
    assert "paid ccy" in spec.header


def test_editor_identity_columns_are_read_only():
    """R-P1: Name and Instrument are read-only; the four inputs are editable."""
    disabled = COLS.editor_disabled()
    assert "Name" in disabled and "Symbol" in disabled
    editable = [c.name for c in COLS.EDITOR_COLUMNS if c.editable]
    assert editable == ["Avg_Entry_Price", "Current_Value_EUR", "Broker_PnL_EUR",
                        "Plan_EUR_month"]


def test_no_remove_column():
    """Owner decision: the grid is exactly the 6 contract columns (no Remove)."""
    assert "Remove" not in COLS.editor_column_order()
    assert len(COLS.editor_column_order()) == 6


# ── A1: clean Name ────────────────────────────────────────────────────────────

def test_clean_name_never_embeds_the_ticker():
    """A1: an unknown symbol yields "" (never the ticker)."""
    from quant.ui import render as R

    for sym in ("ZZZZ.XX", "EUNL.DE", "SXRV.DE", "AMZN", "5J50.DE"):
        name = R._clean_name(sym)
        assert sym.upper() not in name.upper()
        assert f"({sym})" not in name
        assert f"({sym.split('.')[0]})" not in name


# ── Entry-check amber case (pure) ─────────────────────────────────────────────

def test_entry_check_amber_on_10pct_mistype(monkeypatch):
    """Acceptance: a 10% mistyped Value yields the delta, shares and amber flag."""
    from quant.data import currency
    from quant.ui import render as R

    # 5 shares (500 invested / 100 cost) at a live 18 EUR = 90; typed 100 -> +10%.
    monkeypatch.setattr(currency, "price_in_eur", lambda _sym, *_a, **_k: 18.0)
    portfolio = pd.DataFrame([{"Symbol": "EUNL.DE", "Avg_Entry_Price": 100.0,
                               "Current_Value_EUR": 100.0, "Broker_PnL_EUR": -400.0}])
    rows = R._entry_check_rows(portfolio)
    assert len(rows) == 1
    row = rows[0]
    assert row["Shares"] == 5.0
    assert row["Typed"] == 100.0
    assert row["Live"] == 90.0
    assert row["Delta"] == pytest.approx(10.0)
    assert R._is_amber(row["Delta"]) is True


def test_entry_check_not_amber_within_one_percent():
    """R-P1: only a |Delta| > 1% disagreement flags the row amber."""
    from quant.ui import render as R

    assert R._is_amber(0.5) is False
    assert R._is_amber(-1.0) is False
    assert R._is_amber(1.01) is True
    assert R._is_amber(None) is False


# ── AppTest: the rendered headers equal the contract ──────────────────────────

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture
def seeded_portfolio(tmp_path, monkeypatch):
    """A one-row portfolio whose live value differs from the typed value.

    The CSV holds the typed value (110); a saved table + a market bar make the
    live value 100, so the Entry-check row shows a ~9% delta (amber).
    """
    from quant import paths
    from quant.data.database import write_connection

    csv = tmp_path / "portfolio.csv"
    csv.write_text(
        "Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\n"
        "EUNL.DE,100.0,110.0,10.0\n", encoding="utf-8")
    monkeypatch.setattr(paths, "DATA_PORTFOLIO", str(csv))
    with write_connection() as conn:
        conn.execute("DELETE FROM market_history WHERE Symbol = ?", ["EUNL.DE"])
        conn.execute(
            "INSERT OR REPLACE INTO position_snapshots (snapshot_date, symbol, shares) "
            "VALUES (?, ?, ?)", ["2099-01-01", "EUNL.DE", 1.0])
        conn.execute(
            "INSERT OR REPLACE INTO market_history (Symbol, Date, Close) "
            "VALUES (?, ?, ?)", ["EUNL.DE", "2026-10-13", 100.0])
    return csv


def _update_page() -> AppTest:
    at = AppTest.from_file(DASHBOARD, default_timeout=90)
    at.run()
    at.switch_page("pages/update.py").run()
    return at


def _rendered_labels(el) -> list[str]:
    """The visible column labels of a rendered dataframe (index excluded)."""
    cfg = json.loads(el.proto.columns)
    return [v.get("label", "") for k, v in cfg.items() if k != "_index"]


def test_rendered_editor_headers_equal_contract(seeded_portfolio):
    """R-P1: every rendered editor header is non-empty and equals the contract."""
    at = _update_page()
    assert not at.exception, f"Update holdings raised: {at.exception}"
    labels = _rendered_labels(at.dataframe[0])
    assert labels == EDITOR_HEADERS
    assert all(lbl.strip() for lbl in labels)


def test_rendered_check_headers_equal_contract(seeded_portfolio):
    """R-P1/A2: the rendered Entry-check headers equal the contract."""
    at = _update_page()
    labels = _rendered_labels(at.dataframe[1])
    assert labels == CHECK_HEADERS


def test_no_table_exceeds_its_column_budget(seeded_portfolio):
    """R-P1: the editor <= 7 columns, the Entry-check table <= 5 columns."""
    at = _update_page()
    assert len(_rendered_labels(at.dataframe[0])) <= COLS.EDITOR_MAX_COLUMNS
    assert len(_rendered_labels(at.dataframe[1])) <= COLS.CHECK_MAX_COLUMNS


def test_rendered_name_cells_never_contain_the_instrument(seeded_portfolio):
    """A1: no rendered Name cell contains the row's Instrument value."""
    at = _update_page()
    df = at.dataframe[0].value
    assert "Name" in df.columns and "Symbol" in df.columns
    for _, r in df.iterrows():
        name = str(r.get("Name") or "")
        sym = str(r.get("Symbol") or "")
        if not name or not sym:
            continue
        assert sym.upper() not in name.upper(), f"Name {name!r} embeds {sym!r}"
        assert f"({sym})" not in name
        assert f"({sym.split('.')[0]})" not in name, f"Name {name!r} embeds a ticker"


def test_entry_check_caption_renders(seeded_portfolio):
    """R-P1: the Entry-check caption names the live date and the derived shares."""
    at = _update_page()
    caps = " ".join(c.value for c in at.caption)
    assert "shares derived from your entries" in caps


def test_entry_check_renders_the_delta(seeded_portfolio):
    """Acceptance: the mistyped row renders a non-zero Delta % in the check table."""
    at = _update_page()
    df = at.dataframe[1].value
    assert "Delta" in df.columns
    deltas = [abs(float(v)) for v in df["Delta"].tolist() if v == v and v is not None]
    assert deltas and max(deltas) > 1.0, f"no amber-worthy delta rendered: {deltas}"
