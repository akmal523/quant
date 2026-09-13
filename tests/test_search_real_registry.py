"""
test_search_real_registry.py — Registry migration + backfill + search (R5).

Fixture registry starts with EMPTY name columns; the backfill fills them through an
injected metadata source; search then resolves "amazon" and "gold"; the label
formatter removes duplicate parens (the banned-token guard meets reality).
"""
from __future__ import annotations

import duckdb

from quant.data.database import migrate_registry_display_name
from quant.data.names import backfill_display_names
from quant.ui.search import TAG_KEYWORDS, _record, label_for, search


def _v1052_shaped_registry() -> duckdb.DuckDBPyConnection:
    """A pre-R5 asset_registry: no display_name column."""
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE asset_registry (symbol VARCHAR PRIMARY KEY, name VARCHAR, "
        "instrument_class VARCHAR, isin VARCHAR, currency VARCHAR, universe_status VARCHAR)"
    )
    conn.execute("INSERT INTO asset_registry (symbol, name, isin, currency, universe_status) "
                 "VALUES ('AMZN','','US0231351067','','ACTIVE')")
    conn.execute("INSERT INTO asset_registry (symbol, name, isin, currency, universe_status) "
                 "VALUES ('GLD','','US78463V1070','','ACTIVE')")
    return conn


def test_migration_adds_column_and_backfill_preserves(tmp_path):
    conn = _v1052_shaped_registry()
    migrate_registry_display_name(conn)
    cols = {r[0] for r in conn.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = 'asset_registry'").fetchall()}
    assert "display_name" in cols

    cur = tmp_path / "names_curated.csv"
    cur.write_text("symbol,display_name\nAMZN,Amazon\n", encoding="utf-8")
    _meta = {"AMZN": ("Amazon.com Inc", "USD"), "GLD": ("SPDR Gold Shares", "USD")}
    backfill_display_names(conn, metadata_source=lambda s: _meta[s],
                           curated_path=str(cur), fill_currency=True)

    got = dict(conn.execute(
        "SELECT symbol, display_name FROM asset_registry").fetchall())
    assert got["AMZN"] == "Amazon"                    # curated preserved
    assert got["GLD"] == "SPDR Gold Shares"
    cur_vals = dict(conn.execute("SELECT symbol, currency FROM asset_registry").fetchall())
    assert cur_vals["GLD"] == "USD"


def test_search_resolves_from_backfilled_registry(tmp_path):
    conn = _v1052_shaped_registry()
    migrate_registry_display_name(conn)
    _meta = {"AMZN": ("Amazon.com Inc", "USD"), "GLD": ("SPDR Gold Shares", "USD")}
    backfill_display_names(conn, metadata_source=lambda s: _meta[s],
                           curated_path=str(tmp_path / "none.csv"))
    rows = conn.execute(
        "SELECT symbol, name, isin, display_name FROM asset_registry").fetchall()
    records = [_record(sym, nm, isin, TAG_KEYWORDS.get(sym), dn)
               for sym, nm, isin, dn in rows]

    res = search(records, "amazon")
    assert res and res[0]["symbol"] == "AMZN"
    res = search(records, "gold")
    assert any(r["symbol"] == "GLD" for r in res)


def test_label_removes_duplicate_parens():
    assert label_for("MSCI World (IWDA.AS)", "IWDA.AS") == "MSCI World (IWDA.AS)"
    assert label_for("MSCI World", "IWDA.AS") == "MSCI World (IWDA.AS)"
    assert label_for("AMZN", "AMZN") == "AMZN"


def test_label_for_cleans_ucits_etf_input():
    # B2: the raw Yahoo longName must be cleaned even if the registry value is not.
    label = label_for(
        "iShares S&P 500 Information Technology Sector UCITS ETF USD (Acc)",
        "QDVE.DE",
    )
    assert label == "iShares S&P 500 Information Technology Sector (QDVE.DE)"


def test_startup_backfill_surfaces_friendly_name(monkeypatch):
    """B1 regression: a pre-existing DB shows friendly names WITHOUT running update.

    The dashboard entry runs the startup backfill on first render; Explore then
    shows the cleaned friendly name for a fresh registry row.
    """
    import pytest as _pytest

    _pytest.importorskip("streamlit")
    from pathlib import Path as _Path

    from streamlit.testing.v1 import AppTest

    from quant.data import names as names_mod
    from quant.data.database import get_connection

    conn = get_connection()
    conn.execute("DELETE FROM asset_registry WHERE symbol = 'AAPL'")
    conn.execute(
        "INSERT INTO asset_registry (symbol, name, display_name, instrument_class, "
        "isin, currency, universe_status) VALUES ('AAPL','','','EQUITY',"
        "'US0378331005','','ACTIVE')"
    )
    monkeypatch.setattr(names_mod, "_yahoo_identity", lambda _s: ("Apple Inc", "USD"))

    dashboard = str(_Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")
    at = AppTest.from_file(dashboard, default_timeout=60)
    at.run()  # entry runs the startup backfill for the pre-existing row
    at.switch_page("pages/explore.py").run()
    at.text_input[0].set_value("AAPL").run()
    text = " ".join(
        str(getattr(el, "value", ""))
        for attr in ("subheader", "title", "caption", "markdown")
        for el in getattr(at, attr, [])
    )
    assert "Apple" in text
