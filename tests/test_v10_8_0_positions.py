"""test_v10_8_0_positions.py — one position source (v10.8.0, 2.1).

Overview, My holdings and Monthly decision must read the same value and weight
per symbol from positions_now.
"""
from __future__ import annotations

import pytest


def _seed(tmp_path, monkeypatch):
    from quant import paths
    from quant.data import database
    from quant.engine import plans

    csv = tmp_path / "portfolio.csv"
    csv.write_text(
        "Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\n"
        "EUNL.DE,95.5,287.58,9.58\n"
        "AMZN,150.0,150.2,0.2\n")
    monkeypatch.setattr(paths, "DATA_PORTFOLIO", str(csv))

    database.init_db()
    with database.write_connection() as conn:
        conn.execute("DELETE FROM holdings_meta")
        conn.execute(
            "INSERT OR REPLACE INTO market_history (Date, Close, Symbol) "
            "VALUES (?, ?, ?)", ["2026-10-01", 100.0, "EUNL.DE"])
        conn.execute(
            "INSERT OR REPLACE INTO market_history (Date, Close, Symbol) "
            "VALUES (?, ?, ?)", ["2026-10-01", 200.0, "AMZN"])
    plans.clear_pending_sync()
    return csv


def test_positions_now_from_csv(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    from quant.engine.positions import positions_now

    pos = {p["symbol"]: p for p in positions_now()}
    assert pos["EUNL.DE"]["value_eur"] == pytest.approx(287.58)
    assert pos["AMZN"]["value_eur"] == pytest.approx(150.2)
    assert pos["EUNL.DE"]["estimated"] is False


def test_consumers_agree(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    import quant.ui.render as render
    from quant.engine.positions import positions_now

    positions = {p["symbol"]: p["value_eur"] for p in positions_now()}
    portfolio = render.load_portfolio()
    csv_values = {str(r["Symbol"]): float(r["Current_Value_EUR"])
                  for _, r in portfolio.iterrows()}
    for sym, value in positions.items():
        assert csv_values[sym] == pytest.approx(value)


def test_positions_now_empty_without_csv(tmp_path, monkeypatch):
    from quant import paths
    from quant.engine.positions import positions_now

    monkeypatch.setattr(paths, "DATA_PORTFOLIO", str(tmp_path / "missing.csv"))
    assert positions_now() == []
