"""test_advanced.py — event bus and behavioral guardrails (v10.8.2).

The Part 2 advanced modules (portfolio context, strategy engine, cash manager,
risk monitor, tax optimizer, attribution) were removed in v10.8.2 as
development-only code unreachable from the five pages.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


def test_event_bus_publish():
    """Publish calls all subscribers for the event type."""
    from quant.infra.event_bus import EventBus
    bus = EventBus()
    received = []
    bus.subscribe("DIP_DETECTED", lambda d: received.append(d["symbol"]))
    bus.publish("DIP_DETECTED", {"symbol": "AMZN"})
    assert received == ["AMZN"]

