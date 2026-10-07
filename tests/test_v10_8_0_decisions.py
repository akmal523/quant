"""test_v10_8_0_decisions.py — Overview and My holdings agree (v10.8.0, Phase 1).

The Overview table and My holdings must read the same value and status per
symbol. Before v10.8.0 the Overview read the frozen review artifact while My
holdings read the live broker CSV, so the two disagreed.
"""
from __future__ import annotations

import pandas as pd
import pytest


def _seed(tmp_path, monkeypatch):
    from quant import paths
    from quant.data import database
    from quant.engine import plans
    from quant.reporting import artifacts

    # A stale audit artifact with a deliberately wrong value.
    pd.DataFrame([{
        "Symbol": "EUNL.DE", "Tier": "CORE", "Value_EUR": 999.0,
        "Current_Weight": "99.0%", "Target_Weight": "50.0%", "Drift": "49.0%",
        "Recommendation": "HOLD",
    }]).to_csv(tmp_path / "portfolio_audit.csv", index=False)
    monkeypatch.setattr(artifacts, "OUTPUTS_DIR", str(tmp_path))

    # A live portfolio with a different value.
    csv = tmp_path / "portfolio.csv"
    csv.write_text(
        "Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\n"
        "EUNL.DE,95.5,287.58,9.58\n")
    monkeypatch.setattr(paths, "DATA_PORTFOLIO", str(csv))

    database.init_db()
    with database.write_connection() as conn:
        conn.execute("DELETE FROM holdings_meta")
    plans.clear_pending_sync()


def test_read_actions_overlays_live_value(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    from quant.reporting import artifacts

    actions = {a["symbol"]: a for a in artifacts.read_actions()}
    assert actions["EUNL.DE"]["value_eur"] == pytest.approx(287.58)


def test_overview_and_holdings_same_value(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    import quant.ui.render as render
    from quant.reporting import artifacts

    actions = {a["symbol"]: a["value_eur"] for a in artifacts.read_actions()}
    monthly = {h["symbol"]: h["value_eur"] for h in render._monthly_holdings()}
    for sym, value in actions.items():
        assert monthly[sym] == pytest.approx(value)
