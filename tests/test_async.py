"""test_async.py — the concurrent text fetcher overlaps its I/O (hermetic).

The SEC fetch is replaced by a 0.5 s blocking stub, so the test needs no network:
three symbols fetched concurrently finish in well under the 1.5 s a serial loop
would take.
"""
import asyncio
import time

import quant.data.async_fetcher as async_fetcher


def test_async_fetch_runs_concurrently(monkeypatch):
    def _slow_8k(symbol):
        time.sleep(0.5)
        return f"8-K text for {symbol}"

    monkeypatch.setattr(async_fetcher, "fetch_latest_8k", _slow_8k)
    symbols = ["AAPL", "MSFT", "GOOGL"]

    start = time.time()
    results = asyncio.run(async_fetcher.fetch_all_texts_concurrently(symbols))
    elapsed = time.time() - start

    assert results == {s: f"8-K text for {s}" for s in symbols}
    assert elapsed < 1.2, f"fetches did not overlap: {elapsed:.2f}s"


def test_sec_edgar_uses_lexbor_backend():
    """v10.7.6: selectolax 1.0 removed the Modest backend.

    ``selectolax.parser`` now raises ImportError, so sec_edgar must import the
    lexbor parser. This guards the collection-time import that broke CI.
    """
    from selectolax.lexbor import LexborHTMLParser

    import quant.data.sec_edgar as sec_edgar

    assert sec_edgar.LexborHTMLParser is LexborHTMLParser
