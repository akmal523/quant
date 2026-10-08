"""
user_data.py — one-time legacy migration to the per-user data dir (v10.8.1, A2).

Intent: before v10.8.1 user state lived inside the repository (``data/`` and
``quant_cache.duckdb``). A ``git pull`` that untracked those files, a
``git clean -fdx``, or a re-clone could wipe or fork the owner's data. This
module brings that legacy state forward into the per-user data dir on first
start, and never touches the originals.

Invariants:
  - Copy only: never moves and never deletes the legacy files.
  - Never overwrites a file the user data dir already owns.
  - Runs at most once, guarded by a marker file, so deleting a curated input in
    the user data dir is not undone on the next start.
  - Never raises; returns the list of copied relative paths.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

from quant import paths

# Files and directories brought forward from the repository ``data/`` dir. This
# includes the user's own state (positions, account, tiers, database, backups)
# AND the shipped curated inputs, so a fresh checkout keeps working.
LEGACY_FILES: tuple[str, ...] = (
    "portfolio.csv",
    "account.yaml",
    "tiers.csv",
    "notify.toml",
    "broker_registry.csv",
    "isin_curated.csv",
    "names_curated.csv",
    "themes.csv",
    "watchlist.csv",
)
LEGACY_TOP_FILES: tuple[str, ...] = (
    "quant_cache.duckdb",
    "quant_cache.duckdb.wal",
)
LEGACY_BACKUP_DIRNAME = "backups"
MARKER_NAME = ".migrated_to_user_data"
# Shipped read-only package assets that are safe to seed into the user data dir
# (never user state; bundled themes ship inside the wheel).
_BUNDLED_SEEDS: tuple[str, ...] = ("themes.csv",)


def migration_marker() -> str:
    """Absolute path of the one-time migration marker."""
    return os.path.join(str(paths.DATA_DIR), MARKER_NAME)


def migration_done() -> bool:
    """True when the one-time migration has already run."""
    return os.path.exists(migration_marker())


def _copy_if_missing(src: str, dst: str) -> bool:
    """Copy ``src`` to ``dst`` only when ``dst`` is absent. Returns True if copied."""
    try:
        if os.path.exists(dst) or not os.path.exists(src):
            return False
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copy2(src, dst)
        return True
    except Exception:  # noqa: BLE001
        return False


def _copy_backups_forward() -> list[str]:
    """Copy legacy ``data/backups/*.tar.gz`` into the user data dir. Copy only."""
    copied: list[str] = []
    src_dir = Path(paths.LEGACY_DATA_DIR) / LEGACY_BACKUP_DIRNAME
    if not src_dir.is_dir():
        return copied
    dst_dir = Path(paths.DATA_DIR) / LEGACY_BACKUP_DIRNAME
    for src in sorted(src_dir.glob("*.tar.gz")):
        dst = dst_dir / src.name
        if _copy_if_missing(str(src), str(dst)):
            copied.append(f"{LEGACY_BACKUP_DIRNAME}/{src.name}")
    return copied


def migrate_legacy_user_data() -> list[str]:
    """Bring legacy repo user state into the user data dir. Once; copy only.

    Writes the marker after the single pass. Returns the copied relative paths
    (for one Settings line). Never raises.
    """
    if migration_done():
        return []
    paths.ensure_dirs()
    copied: list[str] = []
    legacy = Path(paths.LEGACY_DATA_DIR)
    # A wheel install has no legacy repo data dir; seeding still runs elsewhere.
    if legacy.is_dir():
        for name in LEGACY_FILES:
            dst = os.path.join(str(paths.DATA_DIR), name)
            if _copy_if_missing(str(legacy / name), dst):
                copied.append(name)
        copied.extend(_copy_backups_forward())
    for name in LEGACY_TOP_FILES:
        dst = os.path.join(str(paths.PROJECT_ROOT), name)
        if _copy_if_missing(str(paths.CODE_ROOT / name), dst):
            copied.append(name)
    # Seed bundled read-only assets that live in the package (never user state).
    for name in _BUNDLED_SEEDS:
        dst = os.path.join(str(paths.DATA_DIR), name)
        if _copy_if_missing(str(Path(paths.BUNDLED_DATA_DIR) / name), dst):
            copied.append(name)
    _write_marker()
    return copied


def _write_marker() -> None:
    """Record that the one-time migration ran. Never raises."""
    try:
        os.makedirs(str(paths.DATA_DIR), exist_ok=True)
        with open(migration_marker(), "w", encoding="utf-8") as f:
            f.write("migrated\n")
    except Exception:  # noqa: BLE001
        pass
