"""
backup.py — one command, honest hygiene (v10.7.2, Part 3).

Intent: one command produces a restorable archive of the user-owned state, and
the system reminds about backups at most weekly, never nagging. The archive is a
stdlib tar.gz holding the broker CSV, the tier file, the DuckDB database (plus
its ``.wal`` sibling when present), and the account settings. ``data/notify.toml``
is included ONLY with ``--include-secrets`` (it holds the bot token).

Invariants:
  - ``create_backup`` acquires the shared runner lock (all writers respect it).
  - The archive is pruned to the 5 most recent backups.
  - ``last_backup_at`` is persisted; the doctor/Settings render one line.
  - Never raises; returns a result dict.
"""
from __future__ import annotations

import json
import os
import tarfile
from datetime import date, datetime

from quant import paths
from quant.engine import lock as lock_mod

BACKUP_DIR_NAME = "backups"
KEEP_BACKUPS = 5
STALE_BACKUP_DAYS = 30
NAG_INTERVAL_DAYS = 7
STATE_NAME = "backup_state.json"
PREFIX = "quant-backup-"


def backup_dir(custom: str | None = None) -> str:
    """The backup output directory (default ``data/backups``)."""
    return custom or os.path.join(str(paths.DATA_DIR), BACKUP_DIR_NAME)


def _state_path() -> str:
    return os.path.join(str(paths.OUTPUTS_DIR), STATE_NAME)


def read_meta() -> dict:
    """The persisted backup metadata ({} when absent)."""
    try:
        with open(_state_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def write_meta(meta: dict) -> None:
    """Persist the backup metadata. Never raises."""
    try:
        os.makedirs(str(paths.OUTPUTS_DIR), exist_ok=True)
        with open(_state_path(), "w", encoding="utf-8") as f:
            json.dump(meta, f)
    except Exception:  # noqa: BLE001
        pass


def last_backup_at() -> datetime | None:
    """The timestamp of the last backup, or None when never."""
    raw = read_meta().get("last_backup_at")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw))
    except ValueError:
        return None


def _members(include_secrets: bool) -> list[tuple[str, str]]:
    """(arcname, source_path) for existing files, project-relative arcnames."""
    candidates = [
        ("data/portfolio.csv", paths.DATA_PORTFOLIO),
        ("data/tiers.csv", paths.DATA_TIERS),
        ("data/account.yaml", paths.DATA_ACCOUNT),
        (os.path.basename(paths.DB_FILE), paths.DB_FILE),
        (os.path.basename(paths.DB_FILE) + ".wal", paths.DB_FILE + ".wal"),
    ]
    if include_secrets:
        candidates.append(
            ("data/notify.toml", os.path.join(str(paths.DATA_DIR), "notify.toml")))
    return [(arc, src) for arc, src in candidates if os.path.exists(src)]


def prune_backups(out_dir: str | None = None, keep: int = KEEP_BACKUPS) -> int:
    """Keep the ``keep`` most recent backups. Returns the number removed."""
    out_dir = out_dir or backup_dir()
    try:
        files = sorted(
            (f for f in os.listdir(out_dir)
             if f.startswith(PREFIX) and f.endswith(".tar.gz")),
            reverse=True)
    except Exception:  # noqa: BLE001
        return 0
    removed = 0
    for name in files[keep:]:
        try:
            os.remove(os.path.join(out_dir, name))
            removed += 1
        except Exception:  # noqa: BLE001
            pass
    return removed


def snapshot_user_files(custom_dir: str | None = None,
                        now: datetime | None = None) -> dict:
    """Lightweight auto-backup of the user input files (v10.8.1, A2).

    Copies the table, tiers and account settings (NOT the large database) into a
    timestamped archive, pruned to the 5 most recent. Used before every confirmed
    table save and before every update. Never raises.
    """
    now = now or datetime.now()
    try:
        out_dir = backup_dir(custom_dir)
        os.makedirs(out_dir, exist_ok=True)
        members = [
            (arc, src) for arc, src in (
                ("data/portfolio.csv", paths.DATA_PORTFOLIO),
                ("data/tiers.csv", paths.DATA_TIERS),
                ("data/account.yaml", paths.DATA_ACCOUNT),
            ) if os.path.exists(src)
        ]
        if not members:
            return {"ok": False, "path": None, "members": [], "error": "nothing to back up"}
        name = f"{PREFIX}{now.strftime('%Y%m%d-%H%M%S')}.tar.gz"
        path = os.path.join(out_dir, name)
        with tarfile.open(path, "w:gz") as tar:
            for arc, src in members:
                tar.add(src, arcname=arc)
        prune_backups(out_dir)
        meta = read_meta()
        meta["last_backup_at"] = now.isoformat(timespec="seconds")
        write_meta(meta)
        return {"ok": True, "path": path, "members": [a for a, _ in members],
                "error": None}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "path": None, "members": [], "error": str(e)}


def create_backup(custom_dir: str | None = None, include_secrets: bool = False,
                  now: datetime | None = None) -> dict:
    """Create a backup archive under the runner lock. Never raises.

    Returns ``{ok, path, size_mb, members, warning, error}``.
    """
    now = now or datetime.now()
    res = lock_mod.acquire("backup", blocking=True)
    if not res.acquired:
        return {"ok": False, "path": None, "size_mb": 0.0, "members": [],
                "warning": False, "error": res.message}
    try:
        out_dir = backup_dir(custom_dir)
        os.makedirs(out_dir, exist_ok=True)
        members = _members(include_secrets)
        name = f"{PREFIX}{now.strftime('%Y%m%d-%H%M')}.tar.gz"
        path = os.path.join(out_dir, name)
        with tarfile.open(path, "w:gz") as tar:
            for arc, src in members:
                tar.add(src, arcname=arc)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        prune_backups(out_dir)
        meta = read_meta()
        meta["last_backup_at"] = now.isoformat(timespec="seconds")
        write_meta(meta)
        return {"ok": True, "path": path, "size_mb": size_mb,
                "members": [arc for arc, _ in members],
                "warning": include_secrets, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "path": None, "size_mb": 0.0, "members": [],
                "warning": include_secrets, "error": str(e)}
    finally:
        lock_mod.release()


def backup_line(today: date | None = None) -> str:
    """The doctor/Settings one-line backup status (always shown)."""
    from quant.ui import copy as C

    last = last_backup_at()
    if last is None:
        return C.BACKUP_NEVER
    today = today or date.today()
    days = (today - last.date()).days
    if days > STALE_BACKUP_DAYS:
        return C.BACKUP_OLD.format(n=days)
    return C.BACKUP_LAST.format(date=C.fmt_date(last.date()))


def ui_backup_line(today: date | None = None) -> str:
    """The UI line: the gentle reminder at most once per week, never nagging."""
    from quant.ui import copy as C

    today = today or date.today()
    last = last_backup_at()
    if last is not None and (today - last.date()).days <= STALE_BACKUP_DAYS:
        return C.BACKUP_LAST.format(date=C.fmt_date(last.date()))
    meta = read_meta()
    last_nag = meta.get("last_nag_at")
    if last_nag:
        try:
            if (today - date.fromisoformat(str(last_nag)[:10])).days < NAG_INTERVAL_DAYS:
                return ""
        except ValueError:
            pass
    meta["last_nag_at"] = today.isoformat()
    write_meta(meta)
    return backup_line(today)
