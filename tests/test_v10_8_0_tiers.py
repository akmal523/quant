"""test_v10_8_0_tiers.py — one tier vocabulary and cooldown source (v10.8.0, 2.3).

The drift threshold and minimum days are keyed by the CURRENT tier names; legacy
names map through LEGACY_TIER_MAPPING. A cooldown starts from the last recorded
trade, never a first-run baseline.
"""
from __future__ import annotations

import pytest

from quant.config import (
    REBALANCE_DRIFT_TIERS,
    REBALANCE_FREQUENCY_DAYS,
    rebalance_min_days,
    rebalance_threshold,
)


@pytest.mark.parametrize("tier,threshold,days", [
    ("FORTRESS", 0.10, 90),
    ("ALPHA", 0.05, 7),
    ("SPECULATIVE", 0.05, 7),
])
def test_current_tier_threshold_and_days(tier, threshold, days):
    assert rebalance_threshold(tier) == threshold
    assert rebalance_min_days(tier) == days


@pytest.mark.parametrize("legacy,current", [
    ("CORE", "FORTRESS"),
    ("SATELLITE", "ALPHA"),
    ("ACTIVE", "ALPHA"),
    ("SECTOR", "ALPHA"),
])
def test_legacy_tier_maps_to_current(legacy, current):
    assert rebalance_threshold(legacy) == rebalance_threshold(current)
    assert rebalance_min_days(legacy) == rebalance_min_days(current)


def test_tables_use_only_current_tiers():
    for key in REBALANCE_DRIFT_TIERS:
        assert key in ("FORTRESS", "ALPHA", "SPECULATIVE")
    for key in REBALANCE_FREQUENCY_DAYS:
        assert key in ("FORTRESS", "ALPHA", "SPECULATIVE")


def test_unknown_tier_uses_generic_default():
    assert rebalance_threshold("NOPE") == 0.05
    assert rebalance_min_days("NOPE") == 7


def test_cooldown_starts_from_recorded_trade():
    from quant.data import database
    from quant.portfolio.portfolio import get_last_rebalance

    database.init_db()
    with database.write_connection() as conn:
        conn.execute("DELETE FROM flows")
        conn.execute("DELETE FROM rebalance_log")
        conn.execute(
            "INSERT INTO flows (date, type, amount_eur, symbol) VALUES (?, ?, ?, ?)",
            ["2026-09-09", "buy", 200.0, "EUNL.DE"])
    assert get_last_rebalance("EUNL.DE") == "2026-09-09"


def test_no_first_run_baseline():
    from quant.data import database
    from quant.portfolio.portfolio import get_last_rebalance

    database.init_db()
    with database.write_connection() as conn:
        conn.execute("DELETE FROM flows")
        conn.execute("DELETE FROM rebalance_log")
    assert get_last_rebalance("EUNL.DE") is None
