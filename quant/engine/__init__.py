"""
engine — the local engine that lets the system run by itself (v10.7.0).

Intent: today nothing runs unless the user types a command. This package holds
the daily job, the scheduler, alerts, notifications, flows, the ONE list
(advice, taxonomy, pairing, weights), the invest plan and valuation. Modules are pure where possible; I/O lives at the
edges (explicit connections, explicit paths).

Modules:
  - valuation.py  holdings metadata and estimated revaluation
  - flows.py      money flows and Modified Dietz performance math
  - retention.py  data retention (downsample, prune)
  - daily.py      the daily job body and last-run marker
  - scheduler.py  systemd user timer install/status (Phase 3)
  - alerts.py     level-triggered alert conditions (Phase 3)
  - notify.py     Telegram dispatch (Phase 3)
  - today_list.py the ONE list: built read-only, persisted by writers
"""
from __future__ import annotations

__all__ = [
    "valuation",
    "flows",
    "retention",
    "daily",
]
