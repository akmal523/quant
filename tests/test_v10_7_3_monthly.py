"""
test_v10_7_3_monthly.py — monthly decision fixes (v10.7.3, Part 9.7/9.9).

Intent: pre-approval actuals are ad-hoc buys (no reconciliation deviation line),
and the type-ahead autocomplete records a buy for a registry-known but unheld
symbol as an estimated position with a pending-sync marker.

Invariants: tests never touch the network or the real systemd.
"""
from __future__ import annotations

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
             "title", "header", "subheader", "button")


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
    conn.execute("DELETE FROM monthly_plans")
    conn.execute("DELETE FROM flows")
    conn.execute("DELETE FROM holdings_meta")
    conn.execute("DELETE FROM market_history WHERE Symbol = 'AMZN'")
    conn.execute(
        "INSERT INTO market_history (Date, Close, Symbol) "
        "VALUES ('2026-09-30', 50.0, 'AMZN')")
    conn.execute("DELETE FROM asset_registry WHERE symbol = 'AMZN'")
    conn.execute(
        "INSERT INTO asset_registry (symbol, name, display_name, instrument_class, "
        "isin, currency, universe_status) VALUES "
        "('AMZN', 'Amazon.com', 'Amazon.com', 'EQUITY', '', '', 'ACTIVE')")
    return conn


def _record_via_autocomplete(at: AppTest, query: str, amount: float) -> AppTest:
    at.text_input(key="monthly_extra_q").set_value(query).run()
    box = at.selectbox(key="monthly_extra_choice")
    assert box.options, "the autocomplete must offer matches"
    box.set_value(box.options[0]).run()
    at.number_input(key="monthly_extra_amount").set_value(amount).run()
    at.button(key="monthly_save_actuals").click().run()
    return at


def test_preapproval_actuals_no_deviation(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    at = AppTest.from_file(DASHBOARD, default_timeout=90)
    at.run()
    at.switch_page("pages/monthly.py").run()
    assert not at.exception, f"Monthly raised: {at.exception}"
    at = _record_via_autocomplete(at, "AMZN", 200.0)
    text = _all_text(at)
    assert C.MONTHLY_ADHOC_NOTE in text
    assert "Deviation" not in text
    assert "Planned 0" not in text


def test_autocomplete_creates_estimated_position(tmp_path, monkeypatch):
    conn = _seed(tmp_path, monkeypatch)
    at = AppTest.from_file(DASHBOARD, default_timeout=90)
    at.run()
    at.switch_page("pages/monthly.py").run()
    at = _record_via_autocomplete(at, "AMZN", 200.0)

    row = conn.execute(
        "SELECT shares FROM holdings_meta WHERE symbol = 'AMZN'").fetchone()
    assert row is not None and abs(float(row[0]) - 4.0) < 1e-9
    assert plans.is_pending_sync()
