"""
engine — the local engine that lets the system run by itself (v10.7.0).

Intent: today nothing runs unless the user types a command. This package holds
the daily job, the scheduler, the Telegram digest, flows, the ONE list
(rules, pairing, weights), the invest plan and the valuation store. Modules
are pure where possible; I/O lives at the edges (explicit connections, explicit
paths).

Modules:
  - positions.py  the holdings, marked to the valuation store
  - flows.py      money flows and Modified Dietz performance math
  - retention.py  data retention (downsample, prune)
  - daily.py      the daily job body and last-run marker
  - scheduler.py  systemd user timer install/status (Phase 3)
  - notify.py     the daily Telegram digest
  - today_list.py the ONE list: built read-only, persisted by writers
"""
from __future__ import annotations

__all__ = [
    "positions",
    "flows",
    "retention",
    "daily",
]
