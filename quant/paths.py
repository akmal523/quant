"""
paths.py — Central filesystem paths.

Intent: the app must run from any CWD AND from an installed wheel. A source
checkout resolves writable paths to the repo root (so tests and the repo ``data/``
are used); an installed package resolves them to the OS per-user data dir, because
a wheel must NEVER write into site-packages.

State Transition (F-series):
  source checkout (manifest/VCS next to the package) -> writable root = repo root
  installed wheel                                   -> writable root = user_data_dir("quant-ai")

Invariants:
  - ``PACKAGE_DIR`` is always the installed package dir (shipped assets live here).
  - ``PROJECT_ROOT`` is the writable root (repo root in dev, user data dir in prod).
  - Importing this module performs NO filesystem writes.
  - ``ensure_dirs()``/bootstrap create the directories, never import time.
"""
from __future__ import annotations

import os
from pathlib import Path

# quant/paths.py -> PACKAGE_DIR == the quant package dir; _DEV_ROOT == its parent.
PACKAGE_DIR: Path = Path(__file__).resolve().parent
_DEV_ROOT: Path = PACKAGE_DIR.parent


def _is_development(root: Path) -> bool:
    """True for a source checkout: the project manifest or VCS sits next to it."""
    return (root / "pyproject.toml").exists() or (root / ".git").exists()


def _user_data_dir() -> Path:
    """Per-user writable dir for an installed package (never site-packages)."""
    try:
        from platformdirs import user_data_dir

        return Path(user_data_dir("quant-ai"))
    except Exception:  # noqa: BLE001
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
        return Path(base) / "quant-ai"


def _resolve_root() -> Path:
    """Writable root: explicit override > repo checkout > per-user data dir.

    ``QUANT_DATA_DIR`` (env) is an explicit escape hatch for tests, CI, and
    portable installs; it takes precedence over auto-detection.
    """
    override = os.environ.get("QUANT_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return _DEV_ROOT if _is_development(_DEV_ROOT) else _user_data_dir()


IS_DEVELOPMENT: bool = _is_development(_DEV_ROOT)

# Writable root: repo root in a checkout, user data dir when installed, or the
# QUANT_DATA_DIR override when set.
PROJECT_ROOT: Path = _resolve_root()

DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
SEC_FILINGS_DIR: Path = PROJECT_ROOT / "sec_filings"

# Shipped read-only package assets (seed inputs). Present inside the wheel.
BUNDLED_DATA_DIR: Path = PACKAGE_DIR / "_data"

# DuckDB analytical store.
DB_FILE: str = str(PROJECT_ROOT / "quant_cache.duckdb")

# Static inputs (data/ under the writable root; seeded on first run when absent).
DATA_PORTFOLIO: str = str(DATA_DIR / "portfolio.csv")
DATA_BROKER_REGISTRY: str = str(DATA_DIR / "broker_registry.csv")
DATA_WATCHLIST: str = str(DATA_DIR / "watchlist.csv")
DATA_CONFIG: str = str(DATA_DIR / "config.yaml")
# v10.5.0: account-level inputs (cash, risk profile, base currency). Single
# source of truth for cash; replaces the account_state DuckDB table.
DATA_ACCOUNT: str = str(DATA_DIR / "account.yaml")
# v10.5.2 (F1): user-verified ISINs. Highest-trust source for registry repair;
# overrides live metadata forever. Ships with a header and zero data rows.
DATA_ISIN_CURATED: str = str(DATA_DIR / "isin_curated.csv")
# v10.5.3: user-curated friendly display names (symbol, display_name).
DATA_NAMES_CURATED: str = str(DATA_DIR / "names_curated.csv")
# v10.6.0 (H3.4): prose theme tags for Explore search (theme,symbols). Themes are
# human labels, NOT financial identifiers (F1 does not apply). User-editable.
DATA_THEMES: str = str(DATA_DIR / "themes.csv")

# Runtime dirs that must exist before any write.
RUNTIME_DIRS: tuple[Path, ...] = (PROJECT_ROOT, DATA_DIR, OUTPUTS_DIR, SEC_FILINGS_DIR)


def ensure_dirs() -> None:
    """Create the writable runtime directories. Idempotent; never raises.

    Kept out of import time so importing ``quant.paths`` has no side effects.
    """
    for d in RUNTIME_DIRS:
        try:
            Path(d).mkdir(parents=True, exist_ok=True)
        except Exception:  # noqa: BLE001
            pass
