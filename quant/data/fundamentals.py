"""
fundamentals.py — P/E, PEG, ROE, debt/equity, EBIT and interest expense per stock.

Fetch order: the DuckDB ``fundamentals`` cache (7 days) -> Yahoo (``info`` +
``income_stmt``) in a thread with a hard timeout. Called by ``refresh`` for
single stocks only (ETFs have no fundamentals). The score model
(:mod:`quant.engine.score_model`) normalizes debt/equity (Yahoo reports percent)
and computes interest cover = EBIT / |interest expense|.

v10.8.10: the Alpha Vantage path was removed (it never supplied interest
expense, and its retry wrapper raised instead of returning None).
"""
from __future__ import annotations

import logging
import threading
import time

import yfinance as yf

from quant.data.database import get_connection

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 7 * 24 * 3600  # 7 days

# ── Database ──────────────────────────────────────────────────────────────────


def _get_from_cache(symbol: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        """SELECT pe, peg, roe, debt_to_equity, ebit, interest_expense, updated_at
           FROM fundamentals WHERE symbol = ?""",
        [symbol]
    ).fetchone()

    if row:
        pe, peg, roe, de, ebit, interest_exp, updated_at = row
        import time
        # A NULL updated_at means the cache row is stale/incomplete; refetch.
        if updated_at is not None and time.time() - updated_at < CACHE_TTL_SECONDS:
            return {
                "PE": pe, "PEG": peg, "ROE": roe,
                "DebtToEquity": de,
                "EBIT": ebit, "InterestExpense": interest_exp,
            }
    return None

def _save_to_cache(symbol: str, data: dict) -> None:
    conn = get_connection()
    import time
    conn.execute(
        """INSERT OR REPLACE INTO fundamentals
           (symbol, pe, peg, roe, debt_to_equity, ebit, interest_expense, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            symbol,
            data.get("PE"),  data.get("PEG"),  data.get("ROE"),
            data.get("DebtToEquity"),
            data.get("EBIT"), data.get("InterestExpense"),
            time.time(),
        ]
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

# ── yfinance Fallback ─────────────────────────────────────────────────────────

def _yf_worker(symbol: str, result: list) -> None:
    """Isolates yfinance network calls inside a thread-safe container."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

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
                for key in ("Interest Expense", "InterestExpense",
                            "Interest Expense Non Operating"):
                    if key in fin.index:
                        interest_exp = float(fin.loc[key].iloc[0])
                        break
        except Exception:
            pass

        if any(v is not None for v in (pe, peg, roe, de, ebit)):
            result[0] = {
                "PE": pe, "PEG": peg, "ROE": roe,
                "DebtToEquity": de,
                "EBIT": ebit, "InterestExpense": interest_exp,
            }
    except Exception:
        pass

def _fetch_yfinance_fallback(symbol: str) -> dict | None:
    """
    yfinance with strict daemon thread timeout.
    Prevents hanging if Yahoo blocks the IP or DNS fails (e.g. Train WiFi).
    """
    for attempt in range(3):
        result = [None]
        t = threading.Thread(target=_yf_worker, args=(symbol, result), daemon=True)
        t.start()

        # Give yfinance 8 seconds max to respond
        t.join(timeout=8.0)

        if t.is_alive():
            logger.debug(f"[fundamentals] yfinance timeout for {symbol} (attempt {attempt + 1}).")
            time.sleep(2) # Backoff before retry
            continue

        if result[0] is not None:
            return result[0]

    return None

# ── Public Interface ──────────────────────────────────────────────────────────

def get_fundamentals(symbol: str) -> dict:
    """
    Cache (7 days) -> Yahoo. Never raises.

    Returns dict with keys:
      PE, PEG, ROE, DebtToEquity, EBIT, InterestExpense, ICR

    ICR is computed (not stored) as EBIT / |InterestExpense|.
    """
    cached = _get_from_cache(symbol)
    if cached:
        cached["ICR"] = _compute_icr(cached)
        return cached

    data = _fetch_yfinance_fallback(symbol)

    if data:
        _save_to_cache(symbol, data)
        data["ICR"] = _compute_icr(data)
        return data

    return {
        "PE": None, "PEG": None, "ROE": None,
        "DebtToEquity": None, "EBIT": None, "InterestExpense": None, "ICR": None,
    }
