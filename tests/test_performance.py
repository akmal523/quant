"""
test_performance.py — Performance and memory tests (v10.6.5).

Hermetic: synthetic data only, no network, no live store.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_batch_query_performance():
    """The batch query returns the expected shape and is fast."""
    from quant.data.database import batch_query_portfolio_data

    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    start = time.time()
    data = batch_query_portfolio_data(symbols)
    elapsed = time.time() - start

    assert set(data.keys()) == {"prices", "fundamentals", "news"}
    assert elapsed < 5.0


def test_generators_properly_closed():
    """A lazy-load generator can be closed without leaking."""
    from quant.data.database import load_prices_lazy

    gen = load_prices_lazy("AAPL", chunk_size=1000)
    for i, _ in enumerate(gen):
        if i >= 3:
            break
    gen.close()  # generators support close(); must not raise


def test_context_managers_cleanup():
    """read_only_connection can be entered repeatedly without leaking."""
    from quant.data.database import read_only_connection

    with read_only_connection() as conn:
        assert conn.execute("SELECT 1").fetchone() == (1,)
    with read_only_connection() as conn:
        assert conn.execute("SELECT 1").fetchone() == (1,)


def test_cache_files_cleanup():
    """Cache entries are written and cleared without accumulating."""
    from quant.analytics.cache import cache_dir, clear_cache, disk_cache, get_cache_stats

    clear_cache()

    @disk_cache(max_age_days=1)
    def _double(x: int) -> int:
        return x * 2

    for i in range(5):
        _double(i)
    assert get_cache_stats()["num_entries"] > 0

    clear_cache()
    assert get_cache_stats()["num_entries"] == 0
    assert cache_dir().exists()
