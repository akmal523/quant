"""
test_portfolio_editor.py — Portfolio editor contract (v10.5.0, spec 7 / T6e).

Asserts: editor round-trip writes a valid CSV; validation rejects negative
values and unknown symbols with the specified warnings; atomic write leaves no
temp files.
"""
from __future__ import annotations

import os

import pandas as pd

from quant.portfolio.editor import save_portfolio, validate_positions


def _row(sym="EUNL.DE", price=125.0, value=281.28, pnl=3.28):
    return {"Symbol": sym, "Avg_Entry_Price": price,
            "Current_Value_EUR": value, "Broker_PnL_EUR": pnl}


def test_roundtrip_writes_valid_csv(tmp_path):
    df = pd.DataFrame([_row()])
    cleaned, warnings, errors = validate_positions(df, {"EUNL.DE"})
    assert not errors
    p = tmp_path / "portfolio.csv"
    save_portfolio(cleaned, str(p))
    back = pd.read_csv(p)
    assert list(back.columns) == [
        "Symbol", "Avg_Entry_Price", "Current_Value_EUR", "Broker_PnL_EUR",
    ]
    assert back["Symbol"].iloc[0] == "EUNL.DE"


def test_validation_rejects_negative_values():
    df = pd.DataFrame([_row(price=-1.0)])
    _, _, errors = validate_positions(df, {"EUNL.DE"})
    assert errors


def test_unknown_symbol_warns_not_errors():
    df = pd.DataFrame([_row(sym="ZZZ")])
    _, warnings, errors = validate_positions(df, {"AAA"})
    assert not errors
    assert any("not in universe" in w for w in warnings)


def test_atomic_write_leaves_no_temp(tmp_path):
    df = pd.DataFrame([_row()])
    p = tmp_path / "portfolio.csv"
    save_portfolio(df, str(p))
    leftovers = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
    assert leftovers == []


def test_invested_eur_recomputed():
    df = pd.DataFrame([_row(value=281.28, pnl=3.28)])
    cleaned, _, _ = validate_positions(df, {"EUNL.DE"})
    assert abs(float(cleaned["Invested_EUR"].iloc[0]) - 278.0) < 1e-6
