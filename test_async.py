# test_async.py
import asyncio
import time
from async_fetcher import fetch_all_texts_concurrently

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
