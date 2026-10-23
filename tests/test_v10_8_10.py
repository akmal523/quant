"""test_v10_8_10.py — v10.8.10: code matches the rulings it documents.

R-C4: the render path never writes (the ONE list is persisted by writers only).
T4: plan change fires from the valuation store; swaps only for Active/Speculative
scored holdings; a swap symbol carries no standalone buy/sell; the threat score is
the composite. Regime: one vocabulary (bull/bear/chop == rising/falling/mixed), the
analysis stage stores a probability-scaled label. R-P3: "-0" is unrepresentable.

Synthetic fixtures only (tests/conftest.py isolates the session).
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from quant.engine import action_taxonomy as T
from quant.engine import advice_state, today_list
from quant.ui import copy as C

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")


def _h(sym, tier, cur=0.2, tgt=0.2, value=100.0, **kw):
    return {"symbol": sym, "name": sym, "tier": tier, "current_weight": cur,
            "target_weight": tgt, "value_eur": value, **kw}


# ── R-C4: render is read-only ─────────────────────────────────────────────────

def _log_rows() -> list:
    from quant.data.database import write_connection

    with write_connection() as conn:
        return conn.execute("SELECT id, status FROM advice_log ORDER BY id").fetchall()


def test_portfolio_render_does_not_write_the_advice_log(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from quant import paths

    csv = tmp_path / "portfolio.csv"
    csv.write_text("Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\n"
                   "AAA,100.0,900.0,0.0\nBBB,100.0,100.0,0.0\n", encoding="utf-8")
    monkeypatch.setattr(paths, "DATA_PORTFOLIO", str(csv))
    before = _log_rows()
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.run()  # a rerun must not write either
    assert not at.exception
    assert _log_rows() == before


def test_annotate_is_read_only_and_matches_open_rows():
    from quant.data.database import write_connection

    item = {"verb": "sell_part", "symbol": "ANN", "horizon": "Today",
            "amount_eur": 10.0, "reason": "r", "class": "ALPHA"}
    with write_connection() as conn:
        conn.execute("DELETE FROM advice_log")
        synced = advice_state.sync(conn, [item], as_of=date(2026, 5, 1))
    before = _log_rows()
    fresh = {**item, "symbol": "NEW"}
    out = advice_state.annotate([item, fresh], as_of=date(2026, 5, 9))
    assert out[0]["id"] == synced[0]["id"]
    assert out[0]["open_since"] == date(2026, 5, 1)
    assert out[1]["id"] is None and out[1]["open_since"] == date(2026, 5, 9)
    assert _log_rows() == before


def test_mark_item_done_logs_an_unpersisted_item():
    from quant.data.database import write_connection

    item = {"verb": "buy", "symbol": "MID", "horizon": "This month"}
    with write_connection() as conn:
        conn.execute("DELETE FROM advice_log")
        assert advice_state.mark_item_done(conn, item) is True
        rows = conn.execute("SELECT symbol, status FROM advice_log").fetchall()
    assert rows == [("MID", advice_state.STATUS_DONE)]


# ── T4: plan change / swap / threat ───────────────────────────────────────────

def test_plan_change_uses_config_and_per_symbol_months():
    from quant.config import PLAN_CHANGE_MONTHS

    h = [_h("LT", "FORTRESS", cur=0.30, tgt=0.60)]
    assert T.plan_change_items(h, {"LT": PLAN_CHANGE_MONTHS - 1}) == []
    items = T.plan_change_items(h, {"LT": PLAN_CHANGE_MONTHS})
    assert [i["symbol"] for i in items] == ["LT"]


def test_months_under_target_reads_month_ends_from_the_store():
    from quant.data.database import write_connection

    with write_connection() as conn:
        conn.execute("DELETE FROM valuation")
        # Three month-ends with LT at 20% of the total, then one at 70%.
        for d, lt in ((date(2026, 3, 31), 70.0), (date(2026, 4, 30), 20.0),
                      (date(2026, 5, 29), 20.0), (date(2026, 6, 30), 20.0)):
            for sym, v in (("LT", lt), ("OT", 100.0 - lt)):
                conn.execute("INSERT INTO valuation (date, symbol, shares, price_eur, "
                             "fx, value_eur, source) VALUES (?, ?, 1, ?, 1, ?, 't')",
                             [d, sym, v, v])
        months = today_list.months_under_target(
            [_h("LT", "FORTRESS", tgt=0.60)], conn=conn)
        conn.execute("DELETE FROM valuation")
    assert months == {"LT": 3}


def test_swap_only_for_active_or_speculative_scored_holdings():
    ideas = [{"symbol": "IDEA", "name": "Idea", "score": 90.0}]
    holdings = [_h("LT", "FORTRESS"), _h("ODD", ""), _h("ACT", "ALPHA"),
                _h("UNS", "SPECULATIVE")]
    scores = {"LT": {"composite": 10}, "ODD": {"composite": 10},
              "ACT": {"composite": 10}}
    swaps = T.swap_items(holdings, ideas, scores)
    assert [s["symbol"] for s in swaps] == ["ACT"]  # UNS is unscored


def test_a_swap_symbol_has_no_standalone_buy_or_sell():
    holdings = [_h("ACT", "ALPHA", cur=0.9, tgt=0.1, value=900.0),
                _h("ETF", "FORTRESS", cur=0.1, tgt=0.9, value=100.0)]
    scores = {"ACT": {"composite": 50}}
    plain = today_list.build_items(holdings, scores=scores, ideas=[], months_under={})
    assert any(i["symbol"] == "ACT" and i["verb"] == "sell_part" for i in plain)
    items = today_list.build_items(
        holdings, scores=scores, months_under={},
        ideas=[{"symbol": "IDEA", "name": "Idea", "score": 95.0}])
    assert [i["verb"] for i in items if i["symbol"] == "ACT"] == ["swap"]


def test_threat_score_floor_reads_the_composite():
    h = [_h("LOW", "ALPHA")]
    items = T.threat_items(h, {"LOW": {"composite": 5.0, "structural_grade": 90.0}})
    assert any("below" in i["reason"] for i in items)


# ── Regime: one vocabulary ────────────────────────────────────────────────────

@pytest.mark.parametrize("label", ["bear", "falling", "Falling"])
def test_bear_regime_is_recognised_in_either_vocabulary(label):
    from quant.portfolio.regime_constraints import is_bear

    assert is_bear(label) and is_bear({"label": label})
    assert any(i["symbol"] is None for i in T.threat_items([], {}, {"label": label}))


def test_regime_stage_label_is_probability_scaled():
    import math

    from quant.data.database import write_connection
    from quant.engine import analysis
    from quant.portfolio.regime_constraints import regime_from_bull_prob
    from quant.reporting.artifacts import read_regime

    with write_connection() as conn:
        conn.execute("DELETE FROM market_history WHERE Symbol = ?",
                     [analysis.REGIME_SYMBOL])
        for i in range(300):
            close = 100.0 + 10.0 * math.sin(i / 15.0) + 0.05 * i
            conn.execute("INSERT INTO market_history (Symbol, Date, Close) "
                         "VALUES (?, ?, ?)",
                         [analysis.REGIME_SYMBOL, date.fromordinal(739000 + i), close])
        res = analysis._regime_stage(conn, "t", date(2026, 10, 1))
        conn.execute("DELETE FROM market_history WHERE Symbol = ?",
                     [analysis.REGIME_SYMBOL])
    # A probability, not score points (it used to be prob * 15).
    assert 0.0 <= res["prob"] <= 1.0
    assert res["label"] == regime_from_bull_prob(res["prob"])
    assert read_regime()["label"] == res["label"]
    with write_connection() as conn:
        conn.execute("DELETE FROM meta WHERE key = ?", [analysis.REGIME_META_KEY])


# ── R-P3: the delta badge ─────────────────────────────────────────────────────

@pytest.mark.parametrize("eur,pct,text", [
    (0, -0.04, "+0 EUR (-<0.1%)"),
    (-0.3, -0.01, "-<1 EUR (-<0.1%)"),
    (0.5, 0.1, "+1 EUR (+0.1%)"),
    (-0.5, -0.1, "-1 EUR (-0.1%)"),
    (2.5, 1.0, "+3 EUR (+1.0%)"),
])
def test_delta_badge_never_writes_minus_zero(eur, pct, text):
    _icon, _color, got = C.delta_badge(eur, pct)
    assert got == text
    assert "-0 " not in got and "-0.0%" not in got


def test_delta_badge_neutral_boundary_is_strictly_below_half():
    assert C.delta_badge(0.5, 0.1)[0] != C.delta_badge(0.49, 0.1)[0]
