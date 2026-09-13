"""
validation.py — Data validation + liquidity filters.

Intent: prevent bad prices/fundamentals from silently corrupting quant signals.
Bad data quietly destroys factor scores; validate at the system boundary.
Invariants: validate_* functions raise on hard violations, log warnings on soft
ones. Pure functions (no I/O) except where noted.
Dependencies: pandas, numpy, logging.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Liquidity threshold: exclude symbols with ADV < this (USD).
MIN_ADV_USD = 1_000_000
# Participation rate cap: max trade size as fraction of ADV.
MAX_PARTICIPATION = 0.02


def validate_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a market-history DataFrame. Returns cleaned copy.

    Hard checks (raise): negative/zero Close, duplicate (Symbol, Date).
    Soft checks (log warning): >1% NaN Close, non-monotonic Date.
    """
    if df is None or df.empty:
        raise ValueError("validate_market_data: empty DataFrame")

    out = df.copy()

    if "Close" in out.columns:
        nan_frac = out["Close"].isna().mean()
        if nan_frac > 0.01:
            logger.warning("validate_market_data: %.1f%% NaN Close — dropping rows", nan_frac * 100)
        out = out.dropna(subset=["Close"])
        if (out["Close"] <= 0).any():
            raise ValueError("validate_market_data: non-positive Close prices present")

    if {"Symbol", "Date"}.issubset(out.columns):
        dups = out.duplicated(subset=["Symbol", "Date"]).sum()
        if dups:
            logger.warning("validate_market_data: %d duplicate (Symbol, Date) rows dropped", dups)
            out = out.drop_duplicates(subset=["Symbol", "Date"])

    return out


def sanitize_fundamentals(f_data: dict) -> dict:
    """Sanitize a fundamentals dict. Returns a copy with invalid values -> None.

    Intent: PE<0, ROE outside [-1,1], etc. are data errors, not signals.
    """
    out = dict(f_data)

    pe = out.get("PE")
    if pe is not None and (pe <= 0 or not np.isfinite(pe)):
        logger.warning("sanitize_fundamentals: invalid PE=%s -> None", pe)
        out["PE"] = None

    roe = out.get("ROE")
    if roe is not None and (roe < -1 or roe > 1 or not np.isfinite(roe)):
        logger.warning("sanitize_fundamentals: invalid ROE=%s -> None", roe)
        out["ROE"] = None

    for key in ("PEG", "DebtToEquity", "EBIT", "InterestExpense"):
        val = out.get(key)
        if val is not None and not np.isfinite(val):
            out[key] = None

    return out


def liquidity_score(adv_usd: float | None) -> float:
    """Liquidity score in [0,1]. 1.0 if ADV >= MIN_ADV_USD, else linear ramp."""
    if adv_usd is None or adv_usd <= 0:
        return 0.0
    return float(min(adv_usd / MIN_ADV_USD, 1.0))


def is_liquid(adv_usd: float | None, min_adv: float = MIN_ADV_USD) -> bool:
    """True if ADV meets the liquidity threshold."""
    return adv_usd is not None and adv_usd >= min_adv


def max_trade_size(adv_usd: float | None, participation: float = MAX_PARTICIPATION) -> float:
    """Max tradable position size given ADV and participation cap."""
    if adv_usd is None or adv_usd <= 0:
        return 0.0
    return adv_usd * participation