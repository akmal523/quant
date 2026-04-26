"""
fundamentals.py — Hierarchical fundamentals fetcher with ICR support.

Fetch order: SQLite cache → Alpha Vantage (if key set) → yfinance (exponential backoff).

New in this version:
  - ICR (Interest Coverage Ratio = EBIT / |Interest Expense|) fetched from income_stmt.
  - D/E and ICR cached alongside PE/PEG/ROE.
  - tenacity-based exponential backoff on all network calls.
  - Alpha Vantage free-tier hook (25 req/day; activate via ALPHA_VANTAGE_API_KEY in .env).
"""
from __future__ import annotations

import os
import time
import sqlite3
import logging

import requests
import yfinance as yf
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

CACHE_DB_PATH     = "fundamentals_cache.sqlite"
CACHE_TTL_SECONDS = 7 * 24 * 3600  # 7 days


# ── Database ──────────────────────────────────────────────────────────────────

def _init_db() -> None:
    with sqlite3.connect(CACHE_DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals (
                symbol           TEXT PRIMARY KEY,
                pe               REAL,
                peg              REAL,
                roe              REAL,
                debt_to_equity   REAL,
                ebit             REAL,
                interest_expense REAL,
                updated_at       REAL
            )
        """)
        # Non-destructive migration: add columns absent in older schema versions.
        for col, coltype in [
            ("debt_to_equity",   "REAL"),
            ("ebit",             "REAL"),
            ("interest_expense", "REAL"),
        ]:
            try:
                conn.execute(f"ALTER TABLE fundamentals ADD COLUMN {col} {coltype}")
            except Exception:
                pass  # Column already present — normal for existing databases


_init_db()


def _get_from_cache(symbol: str) -> dict | None:
    with sqlite3.connect(CACHE_DB_PATH) as conn:
        cur = conn.execute(
            """SELECT pe, peg, roe, debt_to_equity, ebit, interest_expense, updated_at
               FROM fundamentals WHERE symbol = ?""",
            (symbol,),
        )
        row = cur.fetchone()
    if row:
        pe, peg, roe, de, ebit, interest_exp, updated_at = row
        if time.time() - updated_at < CACHE_TTL_SECONDS:
            return {
                "PE": pe, "PEG": peg, "ROE": roe,
                "DebtToEquity": de,
                "EBIT": ebit, "InterestExpense": interest_exp,
            }
    return None


def _save_to_cache(symbol: str, data: dict) -> None:
    with sqlite3.connect(CACHE_DB_PATH) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO fundamentals
               (symbol, pe, peg, roe, debt_to_equity, ebit, interest_expense, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                symbol,
                data.get("PE"),  data.get("PEG"),  data.get("ROE"),
                data.get("DebtToEquity"),
                data.get("EBIT"), data.get("InterestExpense"),
                time.time(),
            ),
        )


# ── ICR Computation ───────────────────────────────────────────────────────────

def _compute_icr(data: dict) -> float | None:
    """
    Interest Coverage Ratio  =  EBIT / |Interest Expense|

    Interpretation:
      ICR >= 5.0 : Comfortable — debt serviced with room to spare
      ICR >= 3.0 : Adequate
      ICR >= 1.5 : Minimum viable — monitor closely in rising-rate environment
      ICR <  1.5 : Distress zone — operating profit insufficient to cover interest
    """
    ebit = data.get("EBIT")
    ie   = data.get("InterestExpense")
    if ebit is None or ie is None:
        return None
    ie_abs = abs(float(ie))
    if ie_abs < 1:          # Effectively debt-free; ratio is meaningless
        return None
    return round(float(ebit) / ie_abs, 2)


# ── Alpha Vantage (free tier) ─────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    before_sleep=before_sleep_log(logger, logging.DEBUG),
    reraise=False,
)
def _fetch_alpha_vantage(symbol: str) -> dict | None:
    """
    Alpha Vantage OVERVIEW endpoint — free tier: 25 req/day.
    Provides PE, PEG, ROE, D/E.  EBIT via EBITDA proxy (acceptable for ICR ordering).
    Requires ALPHA_VANTAGE_API_KEY in .env.
    """
    from config import ALPHA_VANTAGE_API_KEY
    if not ALPHA_VANTAGE_API_KEY:
        return None

    url = (
        f"https://www.alphavantage.co/query"
        f"?function=OVERVIEW&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    )
    resp = requests.get(url, timeout=12)
    resp.raise_for_status()
    d = resp.json()

    if "Symbol" not in d:
        # Rate-limit response or empty ticker
        logger.debug("[AV] No data for %s — may be rate-limited.", symbol)
        return None

    def _f(key: str) -> float | None:
        v = d.get(key)
        try:
            return float(v) if v and v not in ("None", "-", "") else None
        except (TypeError, ValueError):
            return None

    return {
        "PE":             _f("PERatio"),
        "PEG":            _f("PEGRatio"),
        "ROE":            _f("ReturnOnEquityTTM"),
        "DebtToEquity":   _f("DebtToEquityRatio"),
        "EBIT":           _f("EBITDA"),   # EBITDA used as ICR proxy (OVERVIEW only)
        "InterestExpense": None,           # Not available in OVERVIEW endpoint
    }


# ── yfinance Fallback ─────────────────────────────────────────────────────────

def _fetch_yfinance_fallback(symbol: str) -> dict | None:
    """
    yfinance with manual exponential backoff.
    Fetches PE, PEG, ROE, D/E plus EBIT and Interest Expense from income_stmt.
    """
    for attempt in range(3):
        try:
            ticker = yf.Ticker(symbol)
            info   = ticker.info or {}   # 401 responses return None, not a dict

            pe  = info.get("trailingPE") or info.get("forwardPE")
            peg = info.get("pegRatio")
            roe = info.get("returnOnEquity")
            de  = info.get("debtToEquity")

            ebit, interest_exp = None, None
            try:
                fin = ticker.income_stmt
                if fin is not None and not fin.empty:
                    for key in ("EBIT", "Ebit"):
                        if key in fin.index:
                            ebit = float(fin.loc[key].iloc[0])
                            break
                    for key in ("Interest Expense", "InterestExpense", "Interest Expense Non Operating"):
                        if key in fin.index:
                            interest_exp = float(fin.loc[key].iloc[0])
                            break
            except Exception as exc:
                logger.debug("[fundamentals] income_stmt parse error for %s: %s", symbol, exc)

            if any(v is not None for v in (pe, peg, roe, de, ebit)):
                return {
                    "PE": pe, "PEG": peg, "ROE": roe,
                    "DebtToEquity": de,
                    "EBIT": ebit, "InterestExpense": interest_exp,
                }
            return None

        except Exception as exc:
            if attempt < 2:
                wait = 2 ** attempt + 1
                logger.debug("[fundamentals] yfinance error for %s (attempt %d): %s. Retry in %ds",
                             symbol, attempt + 1, exc, wait)
                time.sleep(wait)
            else:
                logger.debug("[fundamentals] yfinance exhausted retries for %s: %s", symbol, exc)
    return None


# ── Public Interface ──────────────────────────────────────────────────────────

def get_fundamentals(symbol: str) -> dict:
    """
    Hierarchical fetcher: Cache → Alpha Vantage → yfinance fallback.

    Returns dict with keys:
      PE, PEG, ROE, DebtToEquity, EBIT, InterestExpense, ICR

    ICR is computed (not stored) as EBIT / |InterestExpense|.
    """
    cached = _get_from_cache(symbol)
    if cached:
        cached["ICR"] = _compute_icr(cached)
        return cached

    data = _fetch_alpha_vantage(symbol)

    if not data:
        data = _fetch_yfinance_fallback(symbol)

    if data:
        _save_to_cache(symbol, data)
        data["ICR"] = _compute_icr(data)
        return data

    return {
        "PE": None, "PEG": None, "ROE": None,
        "DebtToEquity": None, "EBIT": None, "InterestExpense": None, "ICR": None,
    }
