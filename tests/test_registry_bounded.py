"""
test_registry_bounded.py — registry bounded invariant (H3-fix).

Part 1 (b929ae9): the CSV ceiling (broker_registry.csv is user-owned routing
input, not the working universe).
Part 2 (H3-fix): asset_registry holds EXACTLY the working universe W, the
sync is idempotent, and a bulk universe_master insert is refused. Guards the
defect class (feeding the 1000+ discovery pool into asset_registry).
"""
from __future__ import annotations

from quant.data.registry_repair import (
    sync_registry_to_working_universe,
    working_universe,
)


def _local_db():
    """An isolated registry DB so the session DB is never polluted."""
    import duckdb

    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE asset_registry (symbol VARCHAR PRIMARY KEY, name VARCHAR, "
        "instrument_class VARCHAR, isin VARCHAR, currency VARCHAR, "
        "universe_status VARCHAR, display_name VARCHAR, updated_at DOUBLE)"
    )
    conn.execute(
        "CREATE TABLE funnel_survivors (symbol VARCHAR PRIMARY KEY, score DOUBLE, "
        "updated_at DOUBLE)"
    )
    return conn


def test_sync_makes_registry_exactly_working_universe():
    conn = _local_db()
    conn.execute("INSERT INTO asset_registry (symbol, universe_status) "
                 "VALUES ('__POLLUTED__', 'WATCHLIST')")
    conn.execute("INSERT INTO asset_registry (symbol, universe_status) "
                 "VALUES ('AAPL', 'ACTIVE')")
    conn.execute("INSERT INTO funnel_survivors (symbol) VALUES ('__SURV__')")
    sync_registry_to_working_universe(conn)
    reg = {r[0] for r in conn.execute("SELECT symbol FROM asset_registry").fetchall()}
    wu = working_universe(conn)
    assert reg == wu                      # exact, both directions
    assert reg <= wu and wu <= reg
    assert "__POLLUTED__" not in reg      # outside W -> pruned
    assert "AAPL" in reg and "__SURV__" in reg
    blanks = conn.execute(
        "SELECT COUNT(*) FROM asset_registry "
        "WHERE currency IS NULL OR trim(currency) = ''"
    ).fetchone()[0]
    assert blanks == 0                    # currency stamped for every W row


def test_sync_membership_is_idempotent():
    conn = _local_db()
    conn.execute("INSERT INTO asset_registry (symbol, universe_status) "
                 "VALUES ('AAPL', 'ACTIVE')")
    sync_registry_to_working_universe(conn)
    first = {r[0] for r in conn.execute("SELECT symbol FROM asset_registry").fetchall()}
    sync_registry_to_working_universe(conn)
    second = {r[0] for r in conn.execute("SELECT symbol FROM asset_registry").fetchall()}
    assert first == second
