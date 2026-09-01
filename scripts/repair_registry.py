"""
scripts/repair_registry.py — One-off registry repair for v10.2.

Intent: fix the contradictory registry state left by the v10.1 state machine.
  - Set the configured CORE_ETFS list to status CORE (immutable sleeve).
  - Move SDS, SH (and any future inverse/leveraged names) to WATCHLIST with
    structure = INVERSE.
  - Clear graduated_at for symbols whose status is WATCHLIST (fixes the
    WATCHLIST-with-graduated_at contradiction).
  - Set ZNWD.L to DELISTED (consecutive fetch failures).

Run once:  python3 scripts/repair_registry.py
Dependencies: config, database, taxonomy.
"""
from __future__ import annotations

import os
import sys
import time

# Allow running from the scripts/ dir: add the project root to sys.path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CORE_ETFS
from database import get_connection, init_db
from taxonomy import (
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


def repair() -> dict:
    """Run the one-off registry repair. Returns a summary dict."""
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


if __name__ == "__main__":
    summary = repair()
    print(f"Repair complete: core={summary['core_set']}, "
          f"inverse={summary['inverse_marked']}, delisted={summary['delisted']}, "
          f"broker_synced={summary['broker_synced']}.")