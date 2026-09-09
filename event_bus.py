"""
event_bus.py — Pub/Sub Event Architecture (Part 2, Upgrade #7).

Intent: decouple modules so they react to events (regime change, drawdown,
tax opportunity, dip detected) instead of being called top-down. Modules
subscribe to event types; publishers emit events; the bus fans out to all
subscribers. This makes the system event-driven and extensible.

Invariants:
  - subscribe registers a handler for an event type.
  - publish calls every handler for the event type with the data dict.
  - A handler exception does not break other handlers.

Dependencies: none (stdlib).
"""
from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

# Canonical event types.
EVENTS = {
    "REGIME_CHANGED": "regime_changed",
    "DRAWDOWN_THRESHOLD": "drawdown_threshold",
    "TAX_OPPORTUNITY": "tax_opportunity",
    "DIP_DETECTED": "dip_detected",
    "CORRELATION_SPIKE": "correlation_spike",
    "COST_HURDLE_MISSED": "cost_hurdle_missed",
}


class EventBus:
    """Simple synchronous event bus for portfolio events."""

    def __init__(self):
        self.subscribers: dict[str, list[Callable]] = {}

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Register a handler for an event type."""
        self.subscribers.setdefault(event_type, []).append(handler)

    def publish(self, event_type: str, data: dict) -> None:
        """Call every handler subscribed to the event type with the data."""
        for handler in self.subscribers.get(event_type, []):
            try:
                handler(data)
            except Exception as e:  # noqa: BLE001 — isolate handler failures
                logger.exception("Event handler %s failed for %s: %s",
                                 getattr(handler, "__name__", handler),
                                 event_type, e)