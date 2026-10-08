"""
test_v10_8_1_persistence.py — changes survive a restart (v10.8.1, Part A).

Guards the owner's report ("reopen the site and it is back to default"): the
settings store autosaves, a confirmed table survives a process restart, the
recovery button rebuilds from the newest snapshot, and a failed write reports a
specific reason instead of a generic message.
"""
from __future__ import annotations

from quant import paths
from quant.engine import backup as backup_mod
from quant.portfolio.account import load_account, write_account_fields
from quant.ui.render import _plain_save_error


def test_autosave_settings_persist_across_sessions(tmp_path, monkeypatch):
    """A2 (A4#3): a changed risk profile / savings day is read back."""
    p = tmp_path / "account.yaml"
    monkeypatch.setattr(paths, "DATA_ACCOUNT", str(p))
    assert write_account_fields({"risk_profile": "aggressive"}, str(p)) is True
    assert write_account_fields({"savings_plan_day": 15}, str(p)) is True
    reloaded = load_account(str(p))
    assert reloaded.risk_profile == "aggressive"
    assert reloaded.savings_plan_day == 15


def test_confirmed_table_survives_restart(tmp_path, monkeypatch):
    """A2 (A4#4): a saved table is still there after the process ends."""
    import pandas as pd

    from quant.portfolio.editor import save_portfolio
    from quant.portfolio.portfolio import load_portfolio

    p = tmp_path / "portfolio.csv"
    df = pd.DataFrame([{"Symbol": "AMZN", "Avg_Entry_Price": 150.0,
                        "Current_Value_EUR": 160.0, "Broker_PnL_EUR": 10.0}])
    save_portfolio(df, str(p))
    again = load_portfolio(str(p))
    assert not again.empty
    assert str(again.iloc[0]["Symbol"]) == "AMZN"


def test_recovery_rebuilds_from_holdings_meta(tmp_path, monkeypatch):
    """A3: rebuild the five columns from the newest saved snapshot."""
    from quant.data import database
    from quant.engine.recovery import restore_last_saved_holdings

    conn = database.get_connection()
    conn.execute("DELETE FROM holdings_meta WHERE symbol = 'RECTEST'")
    conn.execute("INSERT INTO holdings_meta (symbol, shares, sync_date, "
                 "invested_at_sync) VALUES ('RECTEST', 2.0, '2026-10-01', 100.0)")

    def _price(_sym, conn=None):
        return 60.0

    monkeypatch.setattr("quant.data.currency.price_in_eur", _price)
    rows = restore_last_saved_holdings(conn=conn)
    row = next(r for r in rows if r["Symbol"] == "RECTEST")
    assert row["Avg_Entry_Price"] == 50.0       # 100 invested / 2 shares
    assert row["Current_Value_EUR"] == 120.0    # 2 shares * 60
    assert row["Broker_PnL_EUR"] == 20.0        # 120 - 100


def test_snapshot_user_files_writes_and_prunes(tmp_path, monkeypatch):
    """A2: the light auto-backup writes an archive and keeps only five."""
    import glob

    data = tmp_path / "data"
    data.mkdir(parents=True)
    (data / "portfolio.csv").write_text("Symbol\nAMZN\n", encoding="utf-8")
    (data / "account.yaml").write_text("base_currency: EUR\n", encoding="utf-8")
    monkeypatch.setattr(paths, "DATA_PORTFOLIO", str(data / "portfolio.csv"))
    monkeypatch.setattr(paths, "DATA_TIERS", str(data / "tiers.csv"))
    monkeypatch.setattr(paths, "DATA_ACCOUNT", str(data / "account.yaml"))
    out = tmp_path / "backups"
    for i in range(7):
        from datetime import datetime
        res = backup_mod.snapshot_user_files(
            str(out), now=datetime(2026, 10, 1, 0, 0, i))
        assert res["ok"] is True
    kept = glob.glob(str(out / "quant-backup-*.tar.gz"))
    assert len(kept) == 5


def test_write_failure_reports_a_specific_reason():
    """A2: a lock conflict names the cause; never a generic message."""
    lock_msg = _plain_save_error(RuntimeError("database is locked by another writer"))
    assert "Try again in a minute" in lock_msg
    other = _plain_save_error(RuntimeError("disk full"))
    assert "Nothing was written" in other


def test_plain_save_error_never_says_success():
    for exc in (RuntimeError("locked"), RuntimeError("boom"), OSError("io")):
        msg = _plain_save_error(exc).lower()
        assert "saved" not in msg
        assert "success" not in msg
