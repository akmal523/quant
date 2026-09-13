"""
test_display_names.py — Clean rules + curated precedence (v10.5.3, R5).
"""
from __future__ import annotations

import duckdb

from quant.data.names import backfill_display_names, clean_display_name, load_curated_names


def test_cleans_real_shaped_longname():
    assert clean_display_name(
        "iShares Core MSCI World UCITS ETF USD (Acc)", "EUNL.DE"
    ) == "iShares Core MSCI World"


def test_degenerate_cleans_to_symbol():
    assert clean_display_name("UCITS ETF USD (Acc)", "ZZZZ") == "ZZZZ"


def test_empty_and_none_fall_back_to_symbol():
    assert clean_display_name("", "X") == "X"
    assert clean_display_name(None, "X") == "X"


def test_curated_precedence_and_missing_only(tmp_path):
    cur = tmp_path / "names_curated.csv"
    cur.write_text("symbol,display_name\nAMZN,Amazon\n", encoding="utf-8")
    assert load_curated_names(str(cur)) == {"AMZN": "Amazon"}

    conn = duckdb.connect()
    conn.execute("CREATE TABLE asset_registry (symbol VARCHAR, name VARCHAR, "
                 "display_name VARCHAR, currency VARCHAR)")
    conn.execute("INSERT INTO asset_registry VALUES ('AMZN','','','')")
    conn.execute("INSERT INTO asset_registry VALUES ('MSFT','','Microsoft','')")
    backfill_display_names(conn, metadata_source=lambda s: ("Yahoo", "USD"),
                           curated_path=str(cur))
    got = dict(conn.execute("SELECT symbol, display_name FROM asset_registry").fetchall())
    assert got["AMZN"] == "Amazon"          # curated wins
    assert got["MSFT"] == "Microsoft"       # existing cell never overwritten


def test_ensure_display_names_survives_write_lock(monkeypatch):
    """App startup must not crash when the DB write lock is held."""
    from quant.data import database, names

    def _locked(*_a, **_k):
        raise TimeoutError("database locked")

    monkeypatch.setattr(database, "connect_with_retry", _locked)
    monkeypatch.setattr(database, "get_connection", _locked)
    result = names.ensure_display_names()  # must not raise
    assert result.get("skipped_reason") == "lock"


def test_ensure_display_names_writes_state_file(tmp_path, monkeypatch):
    from quant import paths
    from quant.data import database, names

    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    conn = database.get_connection()
    conn.execute("DELETE FROM asset_registry WHERE symbol = 'ZZTEST'")
    conn.execute(
        "INSERT INTO asset_registry (symbol, name, display_name, instrument_class, "
        "isin, currency, universe_status) VALUES ('ZZTEST','','','EQUITY','','','ACTIVE')")
    monkeypatch.setattr(names, "_yahoo_identity", lambda _s: ("Test Name", "USD"))
    # Force the connect_with_retry path to fall back to the in-process connection.
    monkeypatch.setattr(database, "connect_with_retry",
                        lambda **_k: (_ for _ in ()).throw(TimeoutError("x")))

    names.ensure_display_names()
    state = names.read_names_state()
    assert state
    assert state["rows_total"] >= 1
    assert state["filled"] >= 1
    assert state["skipped_reason"] in (None, "metadata_unreachable", "lock")
