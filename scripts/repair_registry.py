"""
scripts/repair_registry.py — Idempotent registry repair (v10.5.2, A6).

Intent: two independent, idempotent repairs.
  1. State machine (v10.2): set CORE_ETFS to CORE, mark inverse/leveraged
     products WATCHLIST+INVERSE, clear graduated_at for WATCHLIST, delist ZNWD.L.
  2. ISIN backfill (v10.5.2, F1): fill MISSING ISIN cells only, from a source
     hierarchy — data/isin_curated.csv (user-verified) > existing non-empty cells
     (never overwritten) > live Yahoo instrument metadata. Every written value
     passes is_valid_isin; invalid or absent metadata yields "not found", never a
     guess. Provenance is recorded in the isin_source column (user/curated/yahoo).

Invariants:
  - Idempotent: a second run changes nothing.
  - Missing cells only; existing ISINs are never overwritten.
  - No synthesized identifiers (F1): a value is written only if checksum-valid.

Run:  python3 scripts/repair_registry.py
Dependencies: config, database, taxonomy, quant.data.identifiers, pandas.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import os
import sys
import time

# Allow running from the scripts/ dir: add the project root to sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant.config import CORE_ETFS
from quant.data.database import get_connection, init_db
# ISIN backfill lives in quant.data.registry_repair so the UI can call it
# in-process (A6). Re-exported here for the documented script entry point.
from quant.data.registry_repair import load_curated, repair_isins  # noqa: F401
from quant.execution.taxonomy import (
    set_core, set_structure, mark_delisted, INVERSE_STRUCTURE,
    sync_broker_registry,
)

# Inverse/leveraged products that decay over time. Never accumulation vehicles.
INVERSE_LEVERAGED = {
    "SDS": INVERSE_STRUCTURE,   # 2x inverse S&P 500
    "SH": INVERSE_STRUCTURE,    # inverse S&P 500
}

# Delisted symbols (consecutive fetch failures).
DELISTED = {"ZNWD.L"}


def repair_state_machine() -> dict:
    """Idempotent v10.2 state-machine repair. Returns a summary dict."""
    init_db()
    conn = get_connection()

    # 0. Sync broker_registry.csv ISINs into asset_registry.
    synced = sync_broker_registry()

    # 1. CORE sleeve: immutable, never graduated, never demoted.
    for sym in CORE_ETFS:
        set_core(sym)

    # 2. Inverse/leveraged products -> WATCHLIST with structure = INVERSE.
    for sym, structure in INVERSE_LEVERAGED.items():
        set_structure(sym, structure)
        conn.execute(
            """INSERT INTO asset_registry (symbol, universe_status, updated_at)
               VALUES (?, 'WATCHLIST', ?)
               ON CONFLICT (symbol) DO UPDATE SET
                 universe_status = 'WATCHLIST',
                 updated_at = excluded.updated_at""",
            [sym, time.time()],
        )

    # 3. Clear graduated_at for WATCHLIST symbols (status/history contradiction).
    conn.execute(
        "UPDATE asset_registry SET graduated_at = NULL WHERE universe_status = 'WATCHLIST'"
    )

    # 4. Delisted symbols.
    for sym in DELISTED:
        mark_delisted(sym, "possibly delisted (consecutive fetch failures)")

    return {
        "core_set": len(CORE_ETFS),
        "inverse_marked": len(INVERSE_LEVERAGED),
        "delisted": len(DELISTED),
        "broker_synced": synced,
    }


def repair() -> dict:
    """Run both idempotent repairs. Returns a merged summary dict."""
    state = repair_state_machine()
    isins = repair_isins()
    return {**state, **isins}


if __name__ == "__main__":
    summary = repair()
    print(f"Repair complete: core={summary['core_set']}, "
          f"inverse={summary['inverse_marked']}, delisted={summary['delisted']}, "
          f"broker_synced={summary['broker_synced']}.")
    print(f"filled {summary['filled']} ISINs from market data, "
          f"{summary['not_found']} not found.")
