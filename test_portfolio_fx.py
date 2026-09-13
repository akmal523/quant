"""
test_portfolio_fx.py — Unit tests for the FX-aware, broker-synced portfolio audit.

Plan 3 (Phase 2/3/4): the audit reads Current_Value_EUR + Broker_PnL_EUR
directly from portfolio.csv. Invested_EUR = Current_Value_EUR - Broker_PnL_EUR,
Real_PnL_EUR = Broker_PnL_EUR (never price-guessed). Covers the FX helpers,
broker reconciliation loader, and the enhanced audit's new columns.
"""
from __future__ import annotations

import pandas as pd


def test_get_fx_to_eur_eur_is_one():
    """EUR-listed symbols convert at 1.0 (no FX)."""
    from currency import get_fx_to_eur
    assert get_fx_to_eur("EUNL.DE") == 1.0
    assert get_fx_to_eur("SXRV.DE") == 1.0
    print("  [PASS] test_get_fx_to_eur_eur_is_one")


def test_get_fx_to_eur_usd_positive():
    """US-listed symbols convert via live EUR/USD (positive, sane range)."""
    from currency import get_fx_to_eur
    fx = get_fx_to_eur("AMZN")
    assert 0.5 < fx < 1.5, f"USD->EUR multiplier out of range: {fx}"
    print(f"  [PASS] test_get_fx_to_eur_usd_positive: {fx:.4f}")


def test_load_broker_data():
    """Broker_PnL_EUR now lives in portfolio.csv (broker_data.csv retired)."""
    from portfolio import load_broker_data
    data = load_broker_data("portfolio.csv")
    assert "AMZN" in data
    assert data["AMZN"] == 5.20
    print("  [PASS] test_load_broker_data")


def test_load_portfolio_new_schema():
    """New schema emits Invested_EUR + backward-compat aliases."""
    from portfolio import load_portfolio
    df = load_portfolio("portfolio.csv")
    assert {"Symbol", "Avg_Entry_Price", "Current_Value_EUR", "Broker_PnL_EUR",
            "Invested_EUR"}.issubset(df.columns)
    eunl = df[df["Symbol"] == "EUNL.DE"].iloc[0]
    # Invested = Current_Value - Broker_PnL.
    assert abs(eunl["Invested_EUR"] - (281.25 - 12.00)) < 0.01
    # Legacy aliases for downstream callers.
    assert abs(eunl["Buy_Price"] - 125.03) < 0.01
    assert abs(eunl["Amount_EUR"] - 281.25) < 0.01
    print("  [PASS] test_load_portfolio_new_schema")


def test_enhanced_audit_broker_math():
    """Real PnL comes from the broker, never price-guessed (DCA-safe)."""
    from database import init_db
    init_db()
    from portfolio import enhanced_portfolio_audit

    port = pd.DataFrame({
        "Symbol": ["EUNL.DE", "AMZN"],
        "Avg_Entry_Price": [125.03, 220.55],
        "Current_Value_EUR": [281.25, 150.41],
        "Broker_PnL_EUR": [12.00, 5.20],
    })
    scan = pd.DataFrame({
        "Symbol": ["EUNL.DE", "AMZN"],
        "Current_Price": [126.64, 257.06],
        "Structural_Grade": [80.0, 70.0],
        "Tactical_Grade": [60.0, 55.0],
        "Stewardship": [15.0, 15.0],
        "Active_Score": [70.0, 65.0],
        "Signal": ["HOLD", "HOLD"],
    })

    res = enhanced_portfolio_audit(port, scan, current_date="2026-09-11")
    for col in ["Real_PnL_EUR", "Real_PnL_Pct", "FX_Impact_EUR",
                "Invested_EUR", "Value_EUR", "Recon_Deviation", "Recon_Flag",
                "Current_Price_Native", "Current_Price_EUR"]:
        assert col in res.columns, f"missing column {col}"

    eunl = res[res["Symbol"] == "EUNL.DE"].iloc[0]
    # Broker truth: Invested = Value - Broker_PnL; Real PnL = Broker_PnL.
    assert abs(eunl["Invested_EUR"] - (281.25 - 12.00)) < 0.01
    assert abs(eunl["Real_PnL_EUR"] - 12.00) < 0.01
    assert abs(eunl["Real_PnL_Pct"] - (12.00 / 269.25 * 100)) < 0.01
    # EUR-listed: native == EUR price.
    assert abs(eunl["Current_Price_Native"] - eunl["Current_Price_EUR"]) < 0.01
    print("  [PASS] test_enhanced_audit_broker_math")


def test_reconciliation_flag():
    """Recon_Flag '[!]' fires when system estimate deviates > EUR 1.00."""
    from database import init_db
    init_db()
    from portfolio import enhanced_portfolio_audit

    # Force a large deviation: broker value far from shares * market price.
    port = pd.DataFrame({
        "Symbol": ["EUNL.DE"],
        "Avg_Entry_Price": [125.03],
        "Current_Value_EUR": [999.0],   # broker says 999 EUR
        "Broker_PnL_EUR": [12.00],
    })
    scan = pd.DataFrame({
        "Symbol": ["EUNL.DE"],
        "Current_Price": [126.64],
        "Structural_Grade": [80.0], "Tactical_Grade": [60.0],
        "Stewardship": [15.0], "Active_Score": [70.0], "Signal": ["HOLD"],
    })
    res = enhanced_portfolio_audit(port, scan, current_date="2026-09-11")
    row = res.iloc[0]
    assert row["Recon_Flag"] == "[!]", f"expected [!] flag, got {row['Recon_Flag']}"
    print("  [PASS] test_reconciliation_flag")


if __name__ == "__main__":
    test_get_fx_to_eur_eur_is_one()
    test_get_fx_to_eur_usd_positive()
    test_load_broker_data()
    test_load_portfolio_new_schema()
    test_enhanced_audit_broker_math()
    test_reconciliation_flag()
    print("\nAll FX/broker-sync audit tests passed.")