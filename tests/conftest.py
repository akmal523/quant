"""Pytest bootstrap + test isolation.

Intent: tests must NEVER touch the production DuckDB store. Redirect both the
path resolver (``quant.paths.DB_FILE``) and the database module's bound
``DB_PATH`` to an ephemeral file, then reset the thread-local connection so the
redirect actually takes effect.

Invariants:
  - ``quant.data.database.DB_PATH`` points at a tmp file for the whole session.
  - The thread-local connection is dropped before ``init_db()`` so the patched
    path is used (``database.py`` binds ``DB_PATH`` at import time).
  - Production ``quant_cache.duckdb`` is never opened or deleted.

Dependencies: quant.paths, quant.data.database.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pytest import MonkeyPatch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def isolated_db(tmp_path_factory):
    """Isolate the whole test session in an ephemeral DuckDB file.

    State Transition: real DB_PATH -> patched tmp DB_PATH -> init_db() ->
    tests run -> connection dropped (tmp_path_factory cleans the file).
    """
    from quant import paths
    from quant.data import database

    mp = MonkeyPatch()
    test_db = tmp_path_factory.mktemp("test") / "test.duckdb"
    mp.setattr(paths, "DB_FILE", str(test_db))
    mp.setattr(database, "DB_PATH", str(test_db))

    # database.py caches the connection in a thread-local; drop it so the
    # patched DB_PATH is honoured on the next connect.
    database._local.conn = None
    database.init_db()

    yield

    database._local.conn = None
    mp.undo()


@pytest.fixture(autouse=True)
def _offline_metadata(monkeypatch):
    """Keep tests hermetic: stub network metadata (display names + ISINs)."""
    import quant.data.names as names
    import quant.data.news as news
    import quant.data.registry_repair as registry_repair

    monkeypatch.setattr(names, "_yahoo_identity", lambda _s: ("", ""), raising=False)
    monkeypatch.setattr(registry_repair, "_yahoo_isin", lambda _s: None, raising=False)

    def _offline_fetch(*_a, **_k):
        raise RuntimeError("offline")

    monkeypatch.setattr(news, "_default_fetcher", _offline_fetch, raising=False)
