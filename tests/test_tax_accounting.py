"""
test_tax_accounting.py — German tax accounting (v10.7.6, Part 2).

Asserts the Sparerpauschbetrag math, the married allowance, the tax-loss
harvesting gate, the trades ledger write/read, and the TaxOptimizer wiring.
Every test uses a distinct year so the session-scoped test DB cannot leak state
between tests; the ``mock_trades`` helper also removes its rows.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date

import pandas as pd
import pytest

from quant.portfolio.tax_accounting import (
    ABGELTUNGSSTEUER_RATE,
    calculate_yearly_tax_summary,
    realized_gains_ytd,
    realized_losses_ytd,
    record_trade,
    suggest_tax_loss_harvesting,
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


def test_suggest_tax_loss_harvesting_no_gains():
    """No realized gains means no harvesting suggestions."""
    holdings = pd.DataFrame([{"Symbol": "AAPL", "Broker_PnL_EUR": -100.0}])
    with mock_trades([]):
        suggestions = suggest_tax_loss_harvesting(2005, holdings)
    assert suggestions == []


def test_suggest_tax_loss_harvesting_within_allowance():
    """Gains within the allowance need no harvesting."""
    holdings = pd.DataFrame([{"Symbol": "AAPL", "Broker_PnL_EUR": -100.0}])
    with mock_trades([
        {"date": "2006-05-01", "action": "sell", "realized_pnl_eur": 500.0},
    ]):
        suggestions = suggest_tax_loss_harvesting(2006, holdings)
    assert suggestions == []


def test_suggest_tax_loss_harvesting_above_allowance():
    """Gains above the allowance plus losers yields sorted suggestions."""
    holdings = pd.DataFrame([
        {"Symbol": "AAPL", "Broker_PnL_EUR": -200.0},
        {"Symbol": "MSFT", "Broker_PnL_EUR": -100.0},
    ])
    with mock_trades([
        {"date": "2007-07-15", "action": "sell", "realized_pnl_eur": 1200.0},
    ]):
        suggestions = suggest_tax_loss_harvesting(2007, holdings)

    assert len(suggestions) == 2
    assert suggestions[0]["symbol"] == "AAPL"
    assert suggestions[0]["tax_savings_eur"] == pytest.approx(
        round(200.0 * ABGELTUNGSSTEUER_RATE, 2))
    assert suggestions[1]["symbol"] == "MSFT"
    assert suggestions[1]["tax_savings_eur"] == pytest.approx(
        round(100.0 * ABGELTUNGSSTEUER_RATE, 2))


def test_record_trade_persists():
    """A recorded trade appears in the trades ledger."""
    from quant.data.database import read_only_connection

    record_trade(
        date="2008-10-07",
        symbol=_MARKER,
        action="buy",
        amount_eur=1000.0,
        shares=5.0,
        price_eur=200.0,
        fee_eur=1.0,
    )
    try:
        with read_only_connection() as conn:
            result = conn.execute(
                "SELECT action, amount_eur FROM trades WHERE symbol = ?",
                [_MARKER],
            ).df()
        assert len(result) == 1
        assert result.iloc[0]["action"] == "buy"
        assert float(result.iloc[0]["amount_eur"]) == 1000.0
    finally:
        from quant.data.database import write_connection

        with write_connection() as conn:
            conn.execute("DELETE FROM trades WHERE symbol = ?", [_MARKER])


def test_tax_optimizer_reads_ledger():
    """The TaxOptimizer placeholders now read the trades ledger."""
    from quant.portfolio.tax_optimizer import TaxOptimizer

    year = date.today().year
    with mock_trades([
        {"date": f"{year}-05-01", "action": "sell", "realized_pnl_eur": 300.0},
        {"date": f"{year}-06-01", "action": "sell", "realized_pnl_eur": -100.0},
    ]):
        optimizer = TaxOptimizer(pd.DataFrame())
        assert optimizer._get_realized_gains_ytd() == 300.0
        assert optimizer._get_realized_losses_ytd() == 100.0
        assert realized_gains_ytd(year) == 300.0
        assert realized_losses_ytd(year) == 100.0
