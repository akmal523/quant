"""
utils.py — lightweight display and formatting helpers.
No business logic; no external dependencies beyond numpy/pandas.
"""
import numpy as np
import pandas as pd


def fv(v, dec: int = 2, suffix: str = "", prefix: str = "") -> str:
    """Format a numeric value; returns '—' for None/NaN."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{prefix}{float(v):.{dec}f}{suffix}"


def fmt_sentiment(v) -> str:
    """Display AI sentiment score or the string 'No News' / other labels."""
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    return fv(v, 0)


def sentiment_color(v) -> str:
    """HTML hex color for an AI sentiment value."""
    if v is None or not isinstance(v, (int, float)):
        return "#888"
    s = float(v)
    if s >= 50:   return "#00C851"
    if s >= 20:   return "#7CB342"
    if s >= -20:  return "#FFC107"
    if s >= -50:  return "#FF6D00"
    return "#D50000"


def signal_color(sig: str) -> str:
    """HTML hex color for BUY / SELL / HOLD signal strings."""
    return {"BUY": "#00C851", "SELL": "#D50000", "HOLD": "#FFC107"}.get(sig, "#888")
