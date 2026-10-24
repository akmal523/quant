"""
test_tax_accounting.py — German tax accounting (v10.7.6, Part 2).

Asserts the Sparerpauschbetrag math, the married allowance, the tax-loss
harvesting gate, the trades ledger write/read, and the TaxOptimizer wiring.
Every test uses a distinct year so the session-scoped test DB cannot leak state
between tests; the ``mock_trades`` helper also removes its rows.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest

from quant.portfolio.tax_accounting import (
    ABGELTUNGSSTEUER_RATE,
    calculate_yearly_tax_summary,
)

_MARKER = "TAXTEST"


@contextmanager
def mock_trades(rows: list[dict]):
    """Insert trade rows into the isolated test DB, then remove them."""
    from quant.data.database import write_connection

    with write_connection() as conn:
        for r in rows:
            conn.execute(
                "INSERT INTO trades (date, symbol, action, shares, price_eur, "
                "amount_eur, fee_eur, realized_pnl_eur) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [r.get("date"), r.get("symbol", _MARKER), r.get("action"),
                 r.get("shares"), r.get("price_eur"), r.get("amount_eur", 0.0),
                 r.get("fee_eur", 0.0), r.get("realized_pnl_eur")],
            )
    try:
        yield
    finally:
        with write_connection() as conn:
            conn.execute("DELETE FROM trades WHERE symbol = ?", [_MARKER])


def test_yearly_tax_summary_no_gains():
    """No realized gains and no dividends means no tax."""
    summary = calculate_yearly_tax_summary(2001, filing_status="single")

    assert summary["realized_gains_eur"] == 0.0
    assert summary["dividends_eur"] == 0.0
    assert summary["taxable_income_eur"] == 0.0
    assert summary["estimated_tax_eur"] == 0.0
    assert summary["sparerpauschbetrag_remaining_eur"] == 1000.0


def test_yearly_tax_summary_within_allowance():
    """Gains plus dividends within the allowance are not taxed."""
    with mock_trades([
        {"date": "2002-03-15", "action": "sell", "realized_pnl_eur": 500.0},
        {"date": "2002-06-20", "action": "dividend", "amount_eur": 200.0},
    ]):
        summary = calculate_yearly_tax_summary(2002, filing_status="single")

    assert summary["total_capital_income_eur"] == 700.0
    assert summary["sparerpauschbetrag_used_eur"] == 700.0
    assert summary["taxable_income_eur"] == 0.0
    assert summary["estimated_tax_eur"] == 0.0


def test_yearly_tax_summary_above_allowance():
    """Gains above the allowance are taxed at 26.375 percent."""
    with mock_trades([
        {"date": "2003-08-10", "action": "sell", "realized_pnl_eur": 1200.0},
    ]):
        summary = calculate_yearly_tax_summary(2003, filing_status="single")

    assert summary["total_capital_income_eur"] == 1200.0
    assert summary["sparerpauschbetrag_used_eur"] == 1000.0
    assert summary["taxable_income_eur"] == 200.0
    assert summary["estimated_tax_eur"] == pytest.approx(200.0 * ABGELTUNGSSTEUER_RATE)


def test_yearly_tax_summary_married():
    """Married filing doubles the allowance to 2000 EUR."""
    with mock_trades([
        {"date": "2004-09-05", "action": "sell", "realized_pnl_eur": 1500.0},
    ]):
        summary = calculate_yearly_tax_summary(2004, filing_status="married")

    assert summary["sparerpauschbetrag_eur"] == 2000.0
    assert summary["sparerpauschbetrag_used_eur"] == 1500.0
    assert summary["taxable_income_eur"] == 0.0
    assert summary["estimated_tax_eur"] == 0.0
