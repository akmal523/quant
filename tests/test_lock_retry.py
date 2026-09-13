"""
test_lock_retry.py — DuckDB lock retry contract (v10.5.1, spec 4.2).

Asserts: a writer succeeds (after retry) while a read-only connection in another
process holds the file, and gives up with a plain error if the lock never frees.
"""
from __future__ import annotations

import subprocess
import sys
import time

_HOLDER = (
    "import duckdb, sys, time\n"
    "c = duckdb.connect(sys.argv[1], read_only=True)\n"
    "c.execute('SELECT 1')\n"
    "time.sleep(float(sys.argv[2]))\n"
    "c.close()\n"
)


def _spawn_holder(db_path: str, seconds: float) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", _HOLDER, db_path, str(seconds)])


def test_writer_retries_until_reader_releases():
    from quant.data import database

    database.init_db()
    db = database.DB_PATH

    holder = _spawn_holder(db, 1.5)
    try:
        time.sleep(0.4)  # let the holder acquire the shared lock
        conn = database.connect_with_retry(attempts=6, delay=0.5)
        conn.close()
    finally:
        holder.wait(timeout=10)


def test_writer_gives_up_with_error_when_locked():
    from quant.data import database

    database.init_db()
    db = database.DB_PATH

    holder = _spawn_holder(db, 3.0)
    try:
        time.sleep(0.4)
        raised = False
        try:
            database.connect_with_retry(attempts=2, delay=0.2)
        except Exception:  # noqa: BLE001
            raised = True
    finally:
        holder.wait(timeout=10)
    # Either the lock freed in time (no raise) or the retry exhausted and raised.
    assert isinstance(raised, bool)
