"""
test_portfolio_fx.py — Unit tests for the FX-aware, broker-synced portfolio audit.

Plan 3 (Phase 2/3/4): the audit reads Current_Value_EUR + Broker_PnL_EUR
directly from portfolio.csv. Invested_EUR = Current_Value_EUR - Broker_PnL_EUR,
Real_PnL_EUR = Broker_PnL_EUR (never price-guessed). Covers the FX helpers,
broker reconciliation loader, and the enhanced audit's new columns.
"""
from __future__ import annotations

import sys as _sys
import tempfile
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))


def _sample_portfolio_path(tmpdir: str) -> str:
    """Write a deterministic broker-synced portfolio CSV.

    Intent: tests must NOT depend on the mutable live ``data/portfolio.csv``
    (its rows change with real holdings). Hermetic + reproducible.
    """
    p = _Path(tmpdir) / "portfolio.csv"
    p.write_text(
        "Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\n"
        "EUNL.DE,125.03,281.25,12.00\n"
        "AMZN,220.55,150.41,5.20\n",
        encoding="utf-8",
    )
    return str(p)


def test_get_fx_to_eur_eur_is_one():
    """EUR-listed symbols convert at 1.0 (no FX)."""
    from quant.data.currency import get_fx_to_eur
    assert get_fx_to_eur("EUNL.DE") == 1.0
    assert get_fx_to_eur("SXRV.DE") == 1.0
    print("  [PASS] test_get_fx_to_eur_eur_is_one")


def test_get_fx_to_eur_usd_positive():
    """US-listed symbols convert via live EUR/USD (positive, sane range)."""
    from quant.data.currency import get_fx_to_eur
    fx = get_fx_to_eur("AMZN")
    assert 0.5 < fx < 1.5, f"USD->EUR multiplier out of range: {fx}"
    print(f"  [PASS] test_get_fx_to_eur_usd_positive: {fx:.4f}")


def test_load_portfolio_new_schema():
    """New schema emits Invested_EUR + backward-compat aliases."""
    from quant.portfolio.portfolio import load_portfolio
    with tempfile.TemporaryDirectory() as d:
        df = load_portfolio(_sample_portfolio_path(d))
    assert {"Symbol", "Avg_Entry_Price", "Current_Value_EUR", "Broker_PnL_EUR",
            "Invested_EUR"}.issubset(df.columns)
    eunl = df[df["Symbol"] == "EUNL.DE"].iloc[0]
    # Invested = Current_Value - Broker_PnL.
    assert abs(eunl["Invested_EUR"] - (281.25 - 12.00)) < 0.01
    # Legacy aliases for downstream callers.
    assert abs(eunl["Buy_Price"] - 125.03) < 0.01
    assert abs(eunl["Amount_EUR"] - 281.25) < 0.01
    print("  [PASS] test_load_portfolio_new_schema")
