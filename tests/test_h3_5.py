"""
test_h3_5.py — Explore card + discovery loop (H3.5).

F1-F4: one card helper (title/class/subtitle), the label rule, CSV class
precedence, and no empty-string registry cells. F5: the discovery loop
(universe_master candidates + the zero-match sentence).
"""
from __future__ import annotations

import duckdb

from quant.data.database import get_connection
from quant.execution.taxonomy import classify_instrument
from quant.ui.cards import explore_card_fields
from quant.ui.copy import DISCOVERY_NOT_TRACKED, NOT_TRACKED_LABEL
from quant.ui.search import discovery_candidates, label_for


def test_label_rule_no_double_paren():
    assert label_for("Global Aero & Def (5J50)", "5J50.DE") == "Global Aero & Def (5J50)"
    assert label_for("Silver (SGLN.L)", "SGLN.L") == "Silver (SGLN.L)"
    assert label_for("Amazon.com", "AMZN") == "Amazon.com (AMZN)"


def test_registry_csv_class_precedence():
    # SLV is COMMODITY in data/broker_registry.csv; the CSV wins over heuristics.
    assert classify_instrument("SLV") == "COMMODITY"


def _local_db():
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE asset_registry (symbol VARCHAR PRIMARY KEY, name VARCHAR, "
        "instrument_class VARCHAR, isin VARCHAR, currency VARCHAR, "
        "universe_status VARCHAR, display_name VARCHAR, updated_at DOUBLE)")
    conn.execute(
        "CREATE TABLE funnel_survivors (symbol VARCHAR PRIMARY KEY, score DOUBLE, "
        "updated_at DOUBLE)")
    return conn


def test_sync_classifies_broad_etf_by_name_and_writes_nulls():
    from quant.data.registry_repair import sync_registry_to_working_universe

    conn = _local_db()
    conn.execute("INSERT INTO asset_registry (symbol, universe_status) "
                 "VALUES ('AAPL','ACTIVE')")
    sync_registry_to_working_universe(conn)
    cls = conn.execute(
        "SELECT instrument_class FROM asset_registry WHERE symbol='SGLN.L'").fetchone()
    assert cls and cls[0] == "COMMODITY"          # F3: classified by name
    blanks = conn.execute(
        "SELECT COUNT(*) FROM asset_registry WHERE display_name='' OR name='' "
        "OR currency='' OR isin=''").fetchone()[0]
    assert blanks == 0                            # F2: NULL, never ''


def test_explore_card_prefers_registry_over_csv():
    conn = get_connection()
    seeded = [
        ("ZZ5J", "Global Aero & Def (5J50)", "Global Aero & Def (5J50)", "ETF", "EUR",
         "IE000U9ODG19"),
        ("ZZSGLN", "Silver (SGLN.L)", "Silver ETF (SGLN.L)", "COMMODITY", "GBX", None),
    ]
    for sym, dn, nm, cls, cur, isin in seeded:
        conn.execute("DELETE FROM asset_registry WHERE symbol=?", [sym])
        conn.execute(
            "INSERT INTO asset_registry (symbol, display_name, name, instrument_class, "
            "currency, isin, universe_status) VALUES (?,?,?,?,?,?,'WATCHLIST')",
            [sym, dn, nm, cls, cur, isin])
    try:
        a = explore_card_fields("ZZ5J")
        assert a["name"] == "Global Aero & Def (5J50)"
        assert "Fund" in a["subtitle"] and "EUR" in a["subtitle"]
        b = explore_card_fields("ZZSGLN")
        assert b["name"] == "Silver (SGLN.L)"
        assert "Commodity" in b["subtitle"] and "GBX" in b["subtitle"]
    finally:
        for sym, *_ in seeded:
            conn.execute("DELETE FROM asset_registry WHERE symbol=?", [sym])


def test_discovery_candidates_and_sentence():
    conn = get_connection()
    conn.execute("DELETE FROM universe_master WHERE symbol='ZZAAPL'")
    conn.execute("INSERT INTO universe_master (symbol, name) VALUES ('ZZAAPL','ZZAAPL')")
    try:
        cands = discovery_candidates("zzaapl")
        assert any(c["symbol"] == "ZZAAPL" for c in cands)
        assert DISCOVERY_NOT_TRACKED.format(label="ZZAAPL") == (
            "ZZAAPL is in the discovery universe but not tracked. "
            "Add it in Portfolio to track it.")
        assert NOT_TRACKED_LABEL.format(label="X (X)") == "X (X) - not tracked yet"
    finally:
        conn.execute("DELETE FROM universe_master WHERE symbol='ZZAAPL'")
