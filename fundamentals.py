import os
import time
import sqlite3
import logging
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DB_PATH = "fundamentals_cache.sqlite"
CACHE_TTL_SECONDS = 7 * 24 * 3600  # 7 days

def _init_db() -> None:
    with sqlite3.connect(CACHE_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals (
                symbol TEXT PRIMARY KEY,
                pe REAL,
                peg REAL,
                roe REAL,
                updated_at REAL
            )
        """)

_init_db()

def _get_from_cache(symbol: str) -> dict | None:
    with sqlite3.connect(CACHE_DB_PATH) as conn:
        cursor = conn.execute(
            "SELECT pe, peg, roe, updated_at FROM fundamentals WHERE symbol = ?",
            (symbol,)
        )
        row = cursor.fetchone()
        if row:
            pe, peg, roe, updated_at = row
            if time.time() - updated_at < CACHE_TTL_SECONDS:
                return {"PE": pe, "PEG": peg, "ROE": roe}
    return None

def _save_to_cache(symbol: str, data: dict) -> None:
    with sqlite3.connect(CACHE_DB_PATH) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO fundamentals (symbol, pe, peg, roe, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (symbol, data.get("PE"), data.get("PEG"), data.get("ROE"), time.time()))

def _fetch_primary_api(symbol: str) -> dict | None:
    """
    Primary data hook. 
    Requires implementation of FMP, AlphaVantage, or similar institutional API.
    """
    # FMP_API_KEY = os.getenv("FMP_API_KEY")
    # url = f"https://financialmodelingprep.com/api/v3/key-metrics/{symbol}?apikey={FMP_API_KEY}"
    # Implement HTTP GET and parsing here.
    return None

def _fetch_yfinance_fallback(symbol: str) -> dict | None:
    try:
        info = yf.Ticker(symbol).info
        pe = info.get("trailingPE") or info.get("forwardPE")
        peg = info.get("pegRatio")
        roe = info.get("returnOnEquity")
        
        if any(v is not None for v in (pe, peg, roe)):
            return {"PE": pe, "PEG": peg, "ROE": roe}
        return None
    except Exception as e:
        logger.debug("yfinance fallback failed for %s: %s", symbol, e)
        return None

def get_fundamentals(symbol: str) -> dict:
    """
    Hierarchical fetcher: Cache -> Primary API -> yfinance Fallback.
    """
    cached = _get_from_cache(symbol)
    if cached:
        return cached

    data = _fetch_primary_api(symbol)
    
    if not data:
        data = _fetch_yfinance_fallback(symbol)

    if data:
        _save_to_cache(symbol, data)
        return data

    return {"PE": None, "PEG": None, "ROE": None}
