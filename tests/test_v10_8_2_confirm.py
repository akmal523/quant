"""test_v10_8_2_confirm.py — Confirm and save (v10.8.2, section 3).

One action writes the table, appends a position snapshot, and records the diff
in the ledger. Proven recoverable across a restart (the table is on disk).
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from quant.engine.confirm import confirm_save
from quant.engine.diff import diff_tables


def _df(*rows):
    return pd.DataFrame(list(rows), columns=[
        "Symbol", "Avg_Entry_Price", "Current_Value_EUR", "Broker_PnL_EUR"])


def _setup(tmp_path, monkeypatch):
    from quant import paths

    monkeypatch.setattr(paths, "DATA_PORTFOLIO", str(tmp_path / "portfolio.csv"))
    monkeypatch.setattr(paths, "DATA_TIERS", str(tmp_path / "tiers.csv"))
    monkeypatch.setattr(paths, "DATA_ACCOUNT", str(tmp_path / "account.yaml"))
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path / "outputs")
    return paths


def test_confirm_writes_table_snapshot_and_ledger(tmp_path, monkeypatch):
    paths = _setup(tmp_path, monkeypatch)
    new = _df({"Symbol": "AMZN", "Avg_Entry_Price": 150.0,
               "Current_Value_EUR": 180.0, "Broker_PnL_EUR": 30.0})
    diff = diff_tables([], new.to_dict("records"), prices={"AMZN": 180.0})
    from quant.data import database

    conn = database.get_connection()
    conn.execute("DELETE FROM position_snapshots WHERE symbol = 'AMZN'")
    conn.execute("DELETE FROM flows WHERE symbol = 'AMZN'")
    conn.execute("DELETE FROM trades WHERE symbol = 'AMZN'")

    res = confirm_save(new, diff, when=date(2026, 10, 8), path=paths.DATA_PORTFOLIO)
    assert res["ok"] is True
    # Table on disk (survives a restart).
    assert "AMZN" in (tmp_path / "portfolio.csv").read_text(encoding="utf-8")
    # Position snapshot.
    snaps = conn.execute(
        "SELECT shares FROM position_snapshots WHERE symbol = 'AMZN'").fetchall()
    assert snaps and abs(float(snaps[0][0]) - 1.0) < 1e-9   # 150 invested / 150
    # Ledger: one flow + one trade.
    assert conn.execute(
        "SELECT COUNT(*) FROM flows WHERE symbol = 'AMZN'").fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM trades WHERE symbol = 'AMZN'").fetchone()[0] == 1

    # Cleanup the shared session DB.
    conn.execute("DELETE FROM position_snapshots WHERE symbol = 'AMZN'")
    conn.execute("DELETE FROM flows WHERE symbol = 'AMZN'")
    conn.execute("DELETE FROM trades WHERE symbol = 'AMZN'")


def test_no_changes_still_saves_the_table(tmp_path, monkeypatch):
    paths = _setup(tmp_path, monkeypatch)
    rows = _df({"Symbol": "AMZN", "Avg_Entry_Price": 150.0,
                "Current_Value_EUR": 180.0, "Broker_PnL_EUR": 30.0})
    diff = diff_tables(rows.to_dict("records"), rows.to_dict("records"))
    assert diff["summary"]["changed"] is False
    res = confirm_save(rows, diff, when=date(2026, 10, 8),
                       path=paths.DATA_PORTFOLIO)
    assert res["ok"] is True
    assert (tmp_path / "portfolio.csv").exists()
