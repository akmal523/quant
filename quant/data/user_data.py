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


# The owner's own state (not the shipped curated inputs) that the migration copied.
_USER_STATE_FILES: tuple[str, ...] = ("portfolio.csv", "account.yaml", "tiers.csv",
                                      "notify.toml")


def leftover_legacy_state() -> list[str]:
    """User-state files still in the code checkout after the migration ran.

    The migration copies and never deletes, so these stale copies stay behind
    (and drift from the live data). The doctor lists them; the owner decides.
    """
    if not migration_done():
        return []
    legacy = Path(paths.LEGACY_DATA_DIR)
    found = [f"data/{n}" for n in _USER_STATE_FILES if (legacy / n).exists()]
    if (legacy / LEGACY_BACKUP_DIRNAME).is_dir():
        found.append(f"data/{LEGACY_BACKUP_DIRNAME}/")
    found += [n for n in LEGACY_TOP_FILES if (Path(paths.CODE_ROOT) / n).exists()]
    return found


def _write_marker() -> None:
    """Record that the one-time migration ran. Never raises."""
    try:
        os.makedirs(str(paths.DATA_DIR), exist_ok=True)
        with open(migration_marker(), "w", encoding="utf-8") as f:
            f.write("migrated\n")
    except Exception:  # noqa: BLE001
        pass


# v10.8.10: per-symbol targets moved from source code (config) into the user's
# tiers.csv. These are the values the code used to hardcode; they seed the new
# target_pct column once, only where no target is set yet.
LEGACY_TARGETS_PCT = {"EUNL.DE": 50.0, "SXRV.DE": 20.0, "AMZN": 20.0, "5J50.DE": 10.0}


def migrate_targets_into_tiers() -> list[str]:
    """Seed ``target_pct`` in tiers.csv from the old hardcoded map. Once.

    Only symbols in the holdings table, only when the file has no target at all
    (so a user who cleared a target is never overridden). Returns the symbols
    seeded. Never raises.
    """
    try:
        import pandas as pd

        from quant.portfolio.portfolio import load_portfolio
        from quant.portfolio.tier_manager import load_tiers, save_tiers

        tiers = load_tiers()
        if not tiers.empty and tiers["target_pct"].notna().any():
            return []
        held = set(load_portfolio(paths.DATA_PORTFOLIO).get("Symbol", []))
        seed = {s: t for s, t in LEGACY_TARGETS_PCT.items() if s in held}
        if not seed:
            return []
        known = set(tiers["symbol"]) if not tiers.empty else set()
        new = [{"symbol": s, "tier": "ALPHA", "target_pct": None, "last_updated": "",
                "notes": ""} for s in seed if s not in known]
        tiers = pd.concat([tiers, pd.DataFrame(new)], ignore_index=True) if new else tiers
        tiers["target_pct"] = [seed.get(s, t) for s, t in
                               zip(tiers["symbol"], tiers["target_pct"], strict=True)]
        save_tiers(tiers)
        return sorted(seed)
    except Exception:  # noqa: BLE001
        return []

