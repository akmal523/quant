"""
tax_accounting.py — German tax accounting with Sparerpauschbetrag (v10.7.6, Part 2).

Intent: track realized gains and dividends through the year, apply the German
Sparerpauschbetrag (the annual tax-free allowance), estimate the Abgeltungssteuer
owed, and suggest tax-loss harvesting to offset gains before year-end.

Rules (Germany 2026):
  - Abgeltungsteuer 25 percent + 5.5 percent solidarity surcharge = 26.375 percent.
  - Sparerpauschbetrag: 1000 EUR single, 2000 EUR married, per year.
  - Realized losses offset realized gains within the same year.

The ledger is the ``trades`` table (a separate table from ``trade_log``, which is
TCA). This module is the single source of truth for realized gains and losses;
``quant.portfolio.tax_optimizer`` reads them through the helpers here.

Invariants:
  - ``calculate_yearly_tax_summary`` returns a dict with non-negative taxable
    income and estimated tax.
  - ``suggest_tax_loss_harvesting`` returns [] unless there are gains above the
    allowance and at least one position with an unrealized loss.
  - Reads use a short-lived read-only connection; writes use ``write_connection``.

Dependencies: pandas, quant.data.database.
"""
from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from quant.data.database import read_only_connection, write_connection

logger = logging.getLogger(__name__)

SPARERPAUSCHBETRAG_SINGLE = 1000.0
SPARERPAUSCHBETRAG_MARRIED = 2000.0
ABGELTUNGSSTEUER_RATE = 0.26375  # 25 percent + 5.5 percent solidarity surcharge


def _load_trades(year: int, action: str) -> pd.DataFrame:
    """Load trades for a year and action type (empty frame when unavailable)."""
    try:
        with read_only_connection() as conn:
            return conn.execute(
                "SELECT * FROM trades WHERE year(date) = ? AND action = ? "
                "ORDER BY date",
                [int(year), str(action)],
            ).df()
    except Exception as exc:  # noqa: BLE001
        logger.warning("trades read failed: %s", exc)
        return pd.DataFrame()


def _sum_column(frame: pd.DataFrame, column: str) -> float:
    """Sum a numeric column, treating missing values as zero."""
    if frame is None or frame.empty or column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return float(values.sum())


def calculate_yearly_tax_summary(year: int, filing_status: str = "single") -> dict:
    """Realized gains, dividends, allowance usage, and estimated tax for a year.

    Args:
        year: the tax year (for example 2026).
        filing_status: "single" (1000 EUR allowance) or "married" (2000 EUR).

    Returns a dict with the realized gains, dividends, total capital income, the
    allowance, how much of it is used and remaining, the taxable income, and the
    estimated tax. Taxable income and estimated tax are never negative.
    """
    sparerpauschbetrag = (
        SPARERPAUSCHBETRAG_MARRIED if filing_status == "married"
        else SPARERPAUSCHBETRAG_SINGLE
    )

    sells = _load_trades(year, action="sell")
    realized_gains = _sum_column(sells, "realized_pnl_eur")

    dividends = _load_trades(year, action="dividend")
    dividends_eur = _sum_column(dividends, "amount_eur")

    total_capital_income = realized_gains + dividends_eur
    sparerpauschbetrag_used = min(max(total_capital_income, 0.0), sparerpauschbetrag)
    taxable_income = max(0.0, total_capital_income - sparerpauschbetrag)
    estimated_tax = taxable_income * ABGELTUNGSSTEUER_RATE

    return {
        "realized_gains_eur": round(realized_gains, 2),
        "dividends_eur": round(dividends_eur, 2),
        "total_capital_income_eur": round(total_capital_income, 2),
        "sparerpauschbetrag_eur": sparerpauschbetrag,
        "sparerpauschbetrag_used_eur": round(sparerpauschbetrag_used, 2),
        "sparerpauschbetrag_remaining_eur": round(
            sparerpauschbetrag - sparerpauschbetrag_used, 2),
        "taxable_income_eur": round(taxable_income, 2),
        "estimated_tax_eur": round(estimated_tax, 2),
    }


def suggest_tax_loss_harvesting(year: int, holdings: pd.DataFrame) -> list[dict]:
    """Suggest selling losers to offset gains before year-end.

    Returns [] unless there are realized gains above the Sparerpauschbetrag and
    at least one holding with an unrealized loss. Suggestions are sorted by tax
    savings, highest first.
    """
    summary = calculate_yearly_tax_summary(year)

    if summary["realized_gains_eur"] <= 0:
        return []
    if summary["sparerpauschbetrag_remaining_eur"] >= summary["realized_gains_eur"]:
        return []
    if holdings is None or holdings.empty or "Broker_PnL_EUR" not in holdings.columns:
        return []

    losers = holdings[pd.to_numeric(holdings["Broker_PnL_EUR"], errors="coerce") < 0]
    suggestions: list[dict] = []
    for _, holding in losers.iterrows():
        unrealized_loss = -float(holding["Broker_PnL_EUR"])
        tax_savings = unrealized_loss * ABGELTUNGSSTEUER_RATE
        symbol = str(holding["Symbol"])
        suggestions.append({
            "symbol": symbol,
            "name": str(holding.get("name", symbol)),
            "unrealized_loss_eur": round(unrealized_loss, 2),
            "tax_savings_eur": round(tax_savings, 2),
            "reason": f"Selling {symbol} saves {tax_savings:.2f} EUR in taxes",
        })

    return sorted(suggestions, key=lambda x: x["tax_savings_eur"], reverse=True)


def realized_gains_ytd(year: int | None = None) -> float:
    """Sum of positive realized PnL for the year (0.0 when none)."""
    y = year or date.today().year
    sells = _load_trades(y, action="sell")
    if sells.empty or "realized_pnl_eur" not in sells.columns:
        return 0.0
    pnl = pd.to_numeric(sells["realized_pnl_eur"], errors="coerce").fillna(0.0)
    return float(pnl[pnl > 0].sum())


def realized_losses_ytd(year: int | None = None) -> float:
    """Sum of absolute negative realized PnL for the year (0.0 when none)."""
    y = year or date.today().year
    sells = _load_trades(y, action="sell")
    if sells.empty or "realized_pnl_eur" not in sells.columns:
        return 0.0
    pnl = pd.to_numeric(sells["realized_pnl_eur"], errors="coerce").fillna(0.0)
    return float(abs(pnl[pnl < 0].sum()))


def record_trade(
    date: str,
    symbol: str,
    action: str,
    amount_eur: float,
    shares: float | None = None,
    price_eur: float | None = None,
    fee_eur: float = 0,
    realized_pnl_eur: float | None = None,
) -> None:
    """Record a buy, sell, or dividend in ONE transaction (both ledgers).

    v10.8.0 (2.4): delegates to quant.engine.ledger so the flows and trades
    ledgers are written together and a sell computes its FIFO realized gain.
    """
    from quant.engine.ledger import record_transaction

    record_transaction(
        when=date, action=action, symbol=symbol, amount_eur=amount_eur,
        shares=shares, price_eur=price_eur, fee_eur=fee_eur,
        realized_pnl_eur=realized_pnl_eur)
