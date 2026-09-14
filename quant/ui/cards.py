"""
cards.py — streamlit-free card field resolution (H3.5).

Intent: ONE helper computes the Explore card's title/class/subtitle so the page
and the doctor probe read the SAME fields — F1-F3 become diagnosable forever and
can never diverge (decision-memo style: one source).

Invariants:
  - name/class/currency/ISIN prefer asset_registry (the working universe).
  - The CSV routing metadata (broker_registry.csv) is the FALLBACK only.
  - Never raises; missing cells degrade to the symbol / empty string.

Dependencies: quant.data.database (read path), quant.execution.taxonomy,
quant.ui.copy (pure).
"""
from __future__ import annotations

from quant.data.database import read_only_connection
from quant.execution.taxonomy import classify_instrument, resolve_broker
from quant.ui.copy import class_word


def explore_card_fields(symbol: str) -> dict:
    """Resolve the Explore card fields for ``symbol`` (H3.5, F1-F3).

    Returns: symbol, name, class, class_word, currency, isin, subtitle,
    isin_source. Registry values win; CSV routing metadata is the fallback.
    """
    row = None
    try:
        with read_only_connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(display_name, name) AS nm, instrument_class, "
                "currency, isin FROM asset_registry WHERE symbol = ?", [symbol]
            ).fetchone()
    except Exception:  # noqa: BLE001
        row = None

    nm = (row[0] if row and row[0] else "")
    cls = (row[1] if row and row[1] else "") or classify_instrument(symbol)
    reg_cur = (row[2] if row and row[2] else "")
    reg_isin = (row[3] if row and row[3] else "")
    broker = resolve_broker(symbol)

    currency = reg_cur or broker.get("currency", "")
    isin = reg_isin or broker.get("isin", "")
    name = nm or symbol
    word = class_word(cls)
    subtitle = " · ".join(p for p in (word, currency, isin) if p)
    return {
        "symbol": symbol,
        "name": name,
        "class": cls,
        "class_word": word,
        "currency": currency,
        "isin": isin,
        "subtitle": subtitle,
        "isin_source": broker.get("isin_source", ""),
    }
