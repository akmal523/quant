"""
paths.py — Central filesystem paths, resolved relative to the project root.

Intent: the app must run regardless of the current working directory. All data
files, the DuckDB store, and generated directories are resolved from the repo
root (the parent of this ``quant`` package), never from CWD.
"""
from __future__ import annotations

from pathlib import Path

# quant/paths.py -> parents[1] == repository root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
SEC_FILINGS_DIR: Path = PROJECT_ROOT / "sec_filings"

# DuckDB analytical store (kept at the repo root).
DB_FILE: str = str(PROJECT_ROOT / "quant_cache.duckdb")

# Static inputs (moved into data/).
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
