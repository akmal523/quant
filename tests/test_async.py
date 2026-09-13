import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
# test_async.py
import asyncio
import time
from quant.data.async_fetcher import fetch_all_texts_concurrently

def test_async_speed():
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    start = time.time()
    results = asyncio.run(fetch_all_texts_concurrently(symbols))
    elapsed = time.time() - start
    
    assert len(results) == 3, "Failed fetching texts."
    assert elapsed < 20.0, f"Async fetch too slow: {elapsed}s. I/O blocking detected."
    assert "AAPL" in results, "Symbol mapping failed."
    
    print(f"Task 4 validation passed. {len(symbols)} tickers fetched in {elapsed:.2f}s.")

if __name__ == "__main__":
    test_async_speed()
