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


# ── The ONE decision list, three groups (redesign 3.2) ────────────────────────

def _advice(kind, symbol, why="do it", eur=None, name=None):
    return {"kind": kind, "symbol": symbol, "company_name": name or symbol,
            "eur": eur, "why": why}


def test_blocked_holding_is_needs_your_input():
    from quant.engine.decisions import build_decision_list
    from quant.ui import copy as C

    holdings = [{"symbol": "AAA", "name": "Alpha", "blocked": True}]
    out = build_decision_list([], holdings, [])
    assert len(out) == 1
    assert out[0]["group"] == C.DECISION_GROUP_INPUT
    assert "AAA" in out[0]["reason"]


def test_actionable_advice_is_recommended():
    from quant.engine.decisions import build_decision_list
    from quant.ui import copy as C

    advice = [_advice("buy", "AAA", eur=100.0),
              _advice("sell_part", "BBB", eur=50.0),
              _advice("change_savings_plan", "CCC", eur=200.0)]
    out = build_decision_list(advice, [], [])
    assert all(d["group"] == C.DECISION_GROUP_RECOMMENDED for d in out)
    assert {d["symbol"] for d in out} == {"AAA", "BBB", "CCC"}


def test_keep_and_rejected_are_optional():
    from quant.engine.decisions import build_decision_list
    from quant.ui import copy as C

    advice = [_advice("keep", "AAA")]
    rejected = [{"symbol": "BBB", "considered_action": "sell",
                 "plain_reason": "below the fee hurdle"}]
    out = build_decision_list(advice, [], rejected)
    assert all(d["group"] == C.DECISION_GROUP_OPTIONAL for d in out)
    assert {d["symbol"] for d in out} == {"AAA", "BBB"}


def test_blocked_symbol_is_not_also_recommended():
    from quant.engine.decisions import build_decision_list
    from quant.ui import copy as C

    holdings = [{"symbol": "AAA", "name": "Alpha", "blocked": True}]
    advice = [_advice("buy", "AAA", eur=100.0)]
    out = build_decision_list(advice, holdings, [])
    groups = [d["group"] for d in out if d["symbol"] == "AAA"]
    assert groups == [C.DECISION_GROUP_INPUT]


def test_group_order_is_input_recommended_optional():
    from quant.engine.decisions import group_order
    from quant.ui import copy as C

    assert group_order() == [C.DECISION_GROUP_INPUT,
                             C.DECISION_GROUP_RECOMMENDED,
                             C.DECISION_GROUP_OPTIONAL]
