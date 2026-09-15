"""bootstrap.py — first-run data-dir seeding (F-series).

Intent: an installed wheel ships no ``data/`` directory, so a fresh install has no
input files. Seed the empty input templates (headers only) and the bundled themes
into the writable data dir on first run.

Invariants:
  - NEVER overwrites a file the user already owns (F1: inputs are user-owned).
  - Templates ship in code, so no user portfolio/registry leaks into the wheel.
  - Idempotent and non-raising; safe to call on every startup.

State Transition: first run -> empty data dir -> templates seeded -> user edits
the seeded files from then on.

Dependencies: quant.paths only.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from quant import paths

# Header-only templates (no user data). Exact column names the readers expect.
_TEMPLATES: dict[str, str] = {
    "portfolio.csv": "Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\n",
    "broker_registry.csv": (
        "yahoo_ticker,isin,tr_ticker,exchange,currency,instrument_class,isin_source\n"
    ),
    "isin_curated.csv": "symbol,isin,verified_by,verified_at\n",
    "names_curated.csv": "symbol,display_name\n",
    "watchlist.csv": "Symbol\n",
    # Account default: currency + profile only; cash stays unset (honest empty state).
    "account.yaml": "base_currency: EUR\nrisk_profile: balanced\n",
}

# Shipped assets copied verbatim when missing (prose themes; not identifiers).
_BUNDLED: tuple[str, ...] = ("themes.csv",)


def seed_user_data() -> list[str]:
    """Create the writable dirs and seed missing inputs. Returns created filenames.

    Existing files are left untouched; a missing bundled asset is skipped.
    """
    paths.ensure_dirs()
    data = Path(paths.DATA_DIR)
    try:
        data.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: BLE001
        pass
    created: list[str] = []
    for name, text in _TEMPLATES.items():
        target = data / name
        if target.exists():
            continue
        try:
            target.write_text(text, encoding="utf-8")
            created.append(name)
        except Exception:  # noqa: BLE001
            pass
    for name in _BUNDLED:
        target = data / name
        src = Path(paths.BUNDLED_DATA_DIR) / name
        if target.exists() or not src.exists():
            continue
        try:
            shutil.copyfile(src, target)
            created.append(name)
        except Exception:  # noqa: BLE001
            pass
    return created
