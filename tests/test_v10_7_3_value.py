"""
test_v10_7_3_value.py — one value source (v10.7.3, Part 9.6).

Intent: after enter_actuals for 200 EUR, the Overview plaque and the holdings
Value column increase by about 200 EUR and carry the estimated label; the broker
statement CSV is unchanged.

Invariants: tests never touch the network or the real systemd.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from quant import paths
from quant.data import database
from quant.engine import plans
from quant.ui import copy as C

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")
_TEXTLIKE = ("markdown", "info", "warning", "error", "caption", "text",
             "title", "header", "subheader", "button", "metric")


def _all_text(at: AppTest) -> str:
    chunks: list[str] = []
    for attr in _TEXTLIKE:
        for el in getattr(at, attr, []):
            val = getattr(el, "label", None)
            if val is None:
                val = getattr(el, "value", "")
            chunks.append(str(val))
    return " ".join(chunks)


def _seed(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    csv = tmp_path / "portfolio.csv"
    csv.write_text(
        "Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\n"
        "5J50.DE,10.0,1000.0,50.0\n", encoding="utf-8")
    monkeypatch.setattr(paths, "DATA_PORTFOLIO", str(csv))
    conn = database.get_connection()
    conn.execute("DELETE FROM holdings_meta")
    conn.execute("DELETE FROM market_history WHERE Symbol = '5J50.DE'")
    conn.execute(
        "INSERT INTO market_history (Date, Close, Symbol) "
        "VALUES ('2026-09-30', 100.0, '5J50.DE')")
    conn.execute(
        "INSERT INTO holdings_meta (symbol, shares, sync_date, invested_at_sync) "
        "VALUES ('5J50.DE', 10.0, '2026-09-01', 1000.0)")
    plans.mark_pending_sync()
    return conn, csv


def test_value_moves_after_actuals(tmp_path, monkeypatch):
    conn, csv = _seed(tmp_path, monkeypatch)

    # Before: the estimate is 10 shares * 100 = 1000 EUR.
    at = AppTest.from_file(DASHBOARD, default_timeout=90)
    at.run()
    assert not at.exception, f"Overview raised: {at.exception}"
    assert "Invested: 1000 EUR" in _all_text(at)

    # Record a 200 EUR buy at 100 EUR/share -> 2 more shares -> 1200 EUR.
    plans.enter_actuals(
        conn, "2026-10",
        [{"symbol": "5J50.DE", "amount_eur": 200.0, "date": date(2026, 10, 1)}],
        price_lookup={"5J50.DE": 100.0})

    at2 = AppTest.from_file(DASHBOARD, default_timeout=90)
    at2.run()
    text2 = _all_text(at2)
    assert "Invested: 1200 EUR" in text2
    assert C.ESTIMATED_LABEL.format(date=C.fmt_date(date.today())) in text2

    # The holdings Value column reads the revalued estimate.
    at2.switch_page("pages/portfolio.py").run()
    assert not at2.exception, f"My holdings raised: {at2.exception}"
    values: set[str] = set()
    for el in at2.dataframe:
        frame = getattr(el, "value", None)
        if frame is not None and "Value (EUR)" in getattr(frame, "columns", []):
            values.update(frame["Value (EUR)"].astype(str))
    assert "1200.00 EUR" in values

    # The broker statement CSV is unchanged (broker is truth).
    assert "1000.0" in csv.read_text(encoding="utf-8")


def test_buy_for_unheld_symbol_creates_estimated_position(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    csv = tmp_path / "portfolio.csv"
    csv.write_text(
        "Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\n"
        "5J50.DE,10.0,1000.0,50.0\n", encoding="utf-8")
    monkeypatch.setattr(paths, "DATA_PORTFOLIO", str(csv))
    conn = database.get_connection()
    conn.execute("DELETE FROM holdings_meta")
    conn.execute("DELETE FROM market_history WHERE Symbol = 'AMZN'")
    conn.execute(
        "INSERT INTO market_history (Date, Close, Symbol) "
        "VALUES ('2026-09-30', 50.0, 'AMZN')")

    plans.enter_actuals(
        conn, "2026-10",
        [{"symbol": "AMZN", "amount_eur": 200.0, "date": date(2026, 10, 1)}],
        price_lookup={"AMZN": 50.0})

    row = conn.execute(
        "SELECT shares FROM holdings_meta WHERE symbol = 'AMZN'").fetchone()
    assert row is not None and abs(float(row[0]) - 4.0) < 1e-9
    assert plans.is_pending_sync()

    at = AppTest.from_file(DASHBOARD, default_timeout=90)
    at.run()
    at.switch_page("pages/portfolio.py").run()
    assert not at.exception, f"My holdings raised: {at.exception}"
    verdicts: set[str] = set()
    for el in at.dataframe:
        frame = getattr(el, "value", None)
        if frame is not None and "Verdict" in getattr(frame, "columns", []):
            verdicts.update(frame["Verdict"].astype(str))
    assert C.ESTIMATED_PENDING in verdicts
