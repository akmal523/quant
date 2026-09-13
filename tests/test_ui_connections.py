"""
test_ui_connections.py — UI connection hygiene (v10.5.1, spec 4.1).

Asserts: read_only_connection always closes; the dashboard never opens a
persistent read-write connection.
"""
from __future__ import annotations

from pathlib import Path


def test_read_only_connection_closes_on_exit():
    from quant.data import database

    database.init_db()
    with database.read_only_connection() as conn:
        conn.execute("SELECT 1")
    # After the context exits the connection must be closed.
    closed = False
    try:
        conn.execute("SELECT 1")
    except Exception:  # noqa: BLE001
        closed = True
    assert closed


def test_read_only_connection_closes_on_exception():
    from quant.data import database

    database.init_db()
    holder = {}
    try:
        with database.read_only_connection() as conn:
            holder["conn"] = conn
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    closed = False
    try:
        holder["conn"].execute("SELECT 1")
    except Exception:  # noqa: BLE001
        closed = True
    assert closed


def test_dashboard_uses_only_read_only_connections():
    src = (Path(__file__).resolve().parents[1] / "quant" / "dashboard.py").read_text()
    assert "read_only_connection" in src
    assert "get_connection" not in src
