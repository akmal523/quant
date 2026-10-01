"""
test_v10_7_2_phase_c.py — Backup: one command, honest hygiene.

Covers (v10.7.2, Part 3):
  - the archive contains the expected members and excludes notify.toml by default;
  - it includes notify.toml with the flag plus a warning;
  - pruning keeps exactly 5;
  - the doctor/Settings lines use the exact copy;
  - a backup while a writer holds the lock waits via the lock helper.
"""
from __future__ import annotations

import json
import tarfile
from datetime import datetime, timedelta

from quant import paths
from quant.engine import backup, lock
from quant.ui import copy as C


def _env(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(paths, "DATA_PORTFOLIO", str(tmp_path / "data" / "portfolio.csv"))
    monkeypatch.setattr(paths, "DATA_TIERS", str(tmp_path / "data" / "tiers.csv"))
    monkeypatch.setattr(paths, "DATA_ACCOUNT", str(tmp_path / "data" / "account.yaml"))
    monkeypatch.setattr(paths, "DB_FILE", str(tmp_path / "quant_cache.duckdb"))
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "outputs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "portfolio.csv").write_text("Symbol\nAMZN\n", encoding="utf-8")
    (tmp_path / "data" / "tiers.csv").write_text("symbol,tier\nAMZN,ALPHA\n", encoding="utf-8")
    (tmp_path / "data" / "account.yaml").write_text("cash_eur: 0\n", encoding="utf-8")
    (tmp_path / "quant_cache.duckdb").write_text("db", encoding="utf-8")


def test_archive_members_exclude_notify(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    (tmp_path / "data" / "notify.toml").write_text("channel = 'telegram'\n", encoding="utf-8")
    res = backup.create_backup(custom_dir=str(tmp_path / "backups"))
    assert res["ok"] is True
    with tarfile.open(res["path"]) as tar:
        names = tar.getnames()
    assert "data/portfolio.csv" in names
    assert "data/tiers.csv" in names
    assert "data/account.yaml" in names
    assert "quant_cache.duckdb" in names
    assert "data/notify.toml" not in names


def test_include_secrets_adds_notify(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    (tmp_path / "data" / "notify.toml").write_text("channel = 'telegram'\n", encoding="utf-8")
    res = backup.create_backup(custom_dir=str(tmp_path / "backups"), include_secrets=True)
    assert res["ok"] is True
    assert res["warning"] is True
    with tarfile.open(res["path"]) as tar:
        assert "data/notify.toml" in tar.getnames()


def test_prune_keeps_five(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    out = tmp_path / "backups"
    out.mkdir()
    for i in range(7):
        (out / f"quant-backup-2026010{i}-1200.tar.gz").write_text("x", encoding="utf-8")
    removed = backup.prune_backups(str(out))
    assert removed == 2
    assert len(list(out.iterdir())) == 5


def test_backup_lines_exact_copy(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    assert backup.backup_line() == C.BACKUP_NEVER

    backup.create_backup(custom_dir=str(tmp_path / "backups"))
    assert backup.backup_line().startswith("Last backup: ")

    meta = backup.read_meta()
    meta["last_backup_at"] = (datetime.now() - timedelta(days=40)).isoformat()
    backup.write_meta(meta)
    assert backup.backup_line() == C.BACKUP_OLD.format(n=40)


def test_ui_backup_line_nags_at_most_weekly(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    from datetime import date

    today = date(2026, 10, 2)
    first = backup.ui_backup_line(today)
    assert first == C.BACKUP_NEVER
    # Same week: suppressed.
    assert backup.ui_backup_line(date(2026, 10, 4)) == ""
    # A week later: shown again.
    assert backup.ui_backup_line(date(2026, 10, 10)) == C.BACKUP_NEVER


def test_backup_uses_lock_helper(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    calls: list[tuple] = []
    real = lock.acquire

    def spy(command, **kw):
        calls.append((command, kw.get("blocking")))
        return real(command, **kw)

    monkeypatch.setattr(lock, "acquire", spy)
    res = backup.create_backup(custom_dir=str(tmp_path / "backups"))
    assert res["ok"] is True
    assert calls and calls[0][0] == "backup" and calls[0][1] is True


def test_backup_blocked_by_live_lock(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    (tmp_path / "outputs" / ".runner.lock").write_text(
        json.dumps({"pid": 1, "started_at": datetime.now().isoformat(),
                    "command": "other"}), encoding="utf-8")
    monkeypatch.setattr(lock, "acquire",
                        lambda *a, **k: lock.LockResult(False, "held by live process"))
    res = backup.create_backup(custom_dir=str(tmp_path / "backups"))
    assert res["ok"] is False
