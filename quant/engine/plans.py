"""
plans.py — the pending-sync marker (v10.8.2, one workflow).

Intent: the monthly plan storage and actuals reconciliation were removed with the
Monthly decision page (v10.8.2). What remains is the pending-sync marker: a small
file that says the estimated positions are newer than the broker CSV export, so
the UI can show the estimate and the sync reminder.

Invariants:
  - The marker helpers never raise.
"""
from __future__ import annotations

import os
from datetime import date

from quant import paths

PENDING_SYNC_MARKER = ".pending_sync"


def mark_pending_sync() -> None:
    """Write the pending-sync marker (portfolio changed, awaiting a CSV sync)."""
    try:
        os.makedirs(str(paths.OUTPUTS_DIR), exist_ok=True)
        with open(os.path.join(str(paths.OUTPUTS_DIR), PENDING_SYNC_MARKER),
                  "w", encoding="utf-8") as f:
            f.write(date.today().isoformat())
    except Exception:  # noqa: BLE001
        pass


def is_pending_sync() -> bool:
    """True when the pending-sync marker exists."""
    return os.path.exists(os.path.join(str(paths.OUTPUTS_DIR), PENDING_SYNC_MARKER))


def clear_pending_sync() -> None:
    """Remove the pending-sync marker (after a fresh CSV sync)."""
    try:
        os.remove(os.path.join(str(paths.OUTPUTS_DIR), PENDING_SYNC_MARKER))
    except Exception:  # noqa: BLE001
        pass
    # R7: the pending-position list is cleared with the marker.
    try:
        from quant.data.database import get_connection

        get_connection().execute("DELETE FROM meta WHERE key = 'pending_symbols'")
    except Exception:  # noqa: BLE001
        pass
