"""
engine — the local engine that lets the system run by itself (v10.7.0).

Intent: today nothing runs unless the user types a command. This package holds
the daily job, the scheduler, alerts, notifications, flows, the monthly
allocator, and valuation. Modules are pure where possible; I/O lives at the
edges (explicit connections, explicit paths).

Modules:
  - valuation.py  holdings metadata and estimated revaluation
  - flows.py      money flows and Modified Dietz performance math
  - retention.py  data retention (downsample, prune)
  - daily.py      the daily job body and last-run marker
  - scheduler.py  systemd user timer install/status (Phase 3)
  - alerts.py     level-triggered alert conditions (Phase 3)
  - notify.py     Telegram/email dispatch (Phase 3)
  - allocator.py  the monthly savings-plan split (Phase 4)
"""
from __future__ import annotations

__all__ = [
    "valuation",
    "flows",
    "retention",
    "daily",
]
