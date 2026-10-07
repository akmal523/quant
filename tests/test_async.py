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


def test_sec_edgar_uses_lexbor_backend():
    """v10.7.6: selectolax 1.0 removed the Modest backend.

    ``selectolax.parser`` now raises ImportError, so sec_edgar must import the
    lexbor parser. This guards the collection-time import that broke CI.
    """
    from selectolax.lexbor import LexborHTMLParser

    import quant.data.sec_edgar as sec_edgar

    assert sec_edgar.LexborHTMLParser is LexborHTMLParser


if __name__ == "__main__":
    test_async_speed()
