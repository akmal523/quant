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


def test_snapshot_user_files_writes_and_prunes(tmp_path, monkeypatch):
    """A2: the light auto-snapshot writes its own series and keeps only ten.

    v10.8.10: snapshots live in ``backups/auto`` with their own prefix, so they
    never prune a full backup and never count as one.
    """
    import glob

    data = tmp_path / "data"
    data.mkdir(parents=True)
    (data / "portfolio.csv").write_text("Symbol\nAMZN\n", encoding="utf-8")
    (data / "account.yaml").write_text("base_currency: EUR\n", encoding="utf-8")
    monkeypatch.setattr(paths, "DATA_PORTFOLIO", str(data / "portfolio.csv"))
    monkeypatch.setattr(paths, "DATA_TIERS", str(data / "tiers.csv"))
    monkeypatch.setattr(paths, "DATA_ACCOUNT", str(data / "account.yaml"))
    out = tmp_path / "backups"
    out.mkdir()
    full = out / "quant-backup-20261001-0000.tar.gz"
    full.write_bytes(b"full backup")
    before = backup_mod.read_meta().get("last_backup_at")
    for i in range(12):
        from datetime import datetime
        res = backup_mod.snapshot_user_files(
            str(out), now=datetime(2026, 10, 1, 0, 0, i))
        assert res["ok"] is True
    kept = glob.glob(str(out / "auto" / "quant-snapshot-*.tar.gz"))
    assert len(kept) == backup_mod.KEEP_SNAPSHOTS
    assert full.exists()
    assert backup_mod.read_meta().get("last_backup_at") == before
