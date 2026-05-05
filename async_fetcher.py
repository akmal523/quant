# async_fetcher.py
import asyncio
import aiohttp
import feedparser
from sec_edgar import fetch_latest_8k
from config import MAX_ASYNC_WORKERS

async def _fetch_news_async(symbol: str, session: aiohttp.ClientSession) -> str:
    url = f"https://finance.yahoo.com/rss/headline?s={symbol}"
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                xml = await response.text()
                feed = feedparser.parse(xml)
                headlines = [entry.title for entry in feed.entries[:5]]
                return " | ".join(headlines)
    except Exception:
        pass
    return ""

async def _fetch_single(symbol: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> tuple[str, str]:
    async with semaphore:
        # 1. Threaded SEC Fetch -> Bypasses GIL for synchronous library
        text = await asyncio.to_thread(fetch_latest_8k, symbol)
        
        # 2. Native Async News Fallback
        if not text:
            text = await _fetch_news_async(symbol, session)
            
        return symbol, text

async def fetch_all_texts_concurrently(symbols: list[str]) -> dict[str, str]:
    semaphore = asyncio.Semaphore(MAX_ASYNC_WORKERS)
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_single(sym, session, semaphore) for sym in symbols]
        results = await asyncio.gather(*tasks)
        
    return {sym: text for sym, text in results}
