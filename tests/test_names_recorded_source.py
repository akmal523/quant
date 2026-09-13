"""
test_names_recorded_source.py — D1 recorded-source test (H2 commit 2).

Runs the PRODUCTION metadata path (names._yahoo_identity -> yfinance.Ticker)
against a recorded info fixture, with no injection into our own functions. Seeds
symbol-valued (poisoned) registry rows, runs the real startup backfill, and
asserts AMZN resolves by "amazon" and the Portfolio label renders "Amazon (AMZN)".
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant import paths
from quant.data import database, names

FIXTURE = Path(__file__).parent / "fixtures" / "yahoo_identity.json"


class _FakeTicker:
    def __init__(self, symbol):
        self._info = json.loads(FIXTURE.read_text(encoding="utf-8")).get(symbol, {}).get("info", {})

    def get_info(self):
        return self._info


def test_production_metadata_path_fills_and_resolves(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    import yfinance as yf
    monkeypatch.setattr(yf, "Ticker", _FakeTicker)  # stub the SOURCE, not our code

    conn = database.get_connection()
    conn.execute("DELETE FROM asset_registry WHERE symbol IN ('AMZN','AAPL','MSFT')")
    for sym in ("AMZN", "AAPL", "MSFT"):
        conn.execute(
            "INSERT INTO asset_registry (symbol, name, display_name, instrument_class, "
            "isin, currency, universe_status) VALUES (?, ?, ?, 'EQUITY', '', '', 'ACTIVE')",
            [sym, sym, sym],
        )

    # Real connect_with_retry fails in-process; the guard falls back to the session
    # connection (documented), so the production backfill path still runs.
    names.ensure_display_names()

    row = conn.execute(
        "SELECT display_name FROM asset_registry WHERE symbol = 'AMZN'").fetchone()
    assert row[0] == "Amazon.com"

    from quant.ui.search import _record, search
    rows = conn.execute(
        "SELECT symbol, name, isin, display_name FROM asset_registry "
        "WHERE symbol = 'AMZN'").fetchall()
    records = [_record(s, n, i, None, d) for s, n, i, d in rows]
    res = search(records, "amazon")
    assert res and res[0]["symbol"] == "AMZN"

    from quant.ui.search import label_for
    assert label_for(row[0], "AMZN") == "Amazon.com (AMZN)"
