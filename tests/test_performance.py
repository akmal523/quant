"""
test_performance.py — Performance and memory tests (v10.6.5).

Hermetic: synthetic data only, no network, no live store.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_context_managers_cleanup():
    """read_only_connection can be entered repeatedly without leaking."""
    from quant.data.database import read_only_connection

    with read_only_connection() as conn:
        assert conn.execute("SELECT 1").fetchone() == (1,)
    with read_only_connection() as conn:
        assert conn.execute("SELECT 1").fetchone() == (1,)
