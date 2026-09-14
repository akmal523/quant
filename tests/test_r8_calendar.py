"""
test_r8_calendar.py — Calendar lines (R8).

- savings-plan countdown arithmetic (same day, countdown, month wrap, clamp),
  all copy via the copy module.
- optional `savings_plan_day` in account.yaml (1-31), persisted and clamped.
- savings-plan routing detection (ETF/CASH -> savings plan).
- markets-closed freshness line on a weekend with a Friday bar.
- Today renders the countdown under the actions block only when a holding
  routes to a savings plan.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

import quant.ui.render as render  # noqa: E402
from quant.execution.routing import holding_routes_to_savings_plan  # noqa: E402
from quant.portfolio.account import AccountState, load_account, save_account  # noqa: E402
from quant.ui import copy as C  # noqa: E402,N812

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")
_TEXTLIKE = ("markdown", "info", "warning", "error", "caption", "text",
             "title", "header", "subheader")


def _all_text(at: AppTest) -> str:
    chunks = []
    for attr in _TEXTLIKE:
        for el in getattr(at, attr, []):
            chunks.append(str(getattr(el, "value", "")))
    return " ".join(chunks)


# ── savings-plan countdown arithmetic (copy module) ──────────────────────────

def test_savings_plan_line_same_day():
    assert C.savings_plan_line(date(2026, 9, 3), 3) == C.SAVINGS_TODAY


def test_savings_plan_line_countdown():
    # 10 Sep -> 15 Sep = 5 days; 15 Sep 2026 is a Tuesday.
    line = C.savings_plan_line(date(2026, 9, 10), 15)
    assert line == C.SAVINGS_COUNTDOWN.format(days=5, date="Tuesday 15 Sep 2026")


def test_savings_plan_line_month_wrap():
    # 28 Sep -> 3 Oct = 5 days (month wrap); 3 Oct 2026 is a Saturday.
    line = C.savings_plan_line(date(2026, 9, 28), 3)
    assert line == C.SAVINGS_COUNTDOWN.format(days=5, date="Saturday 3 Oct 2026")


def test_savings_plan_line_clamps_to_month_end():
    # day=31 in September clamps to 30 Sep; 10 Sep -> 30 Sep = 20 days.
    line = C.savings_plan_line(date(2026, 9, 10), 31)
    assert line == C.SAVINGS_COUNTDOWN.format(days=20, date="Wednesday 30 Sep 2026")


def test_savings_plan_date_carries_weekday():
    line = C.savings_plan_line(date(2026, 9, 10), 15)
    assert "Tuesday 15 Sep 2026" in line  # audit rule: weekday present


# ── account.yaml savings_plan_day ────────────────────────────────────────────

def test_account_reads_savings_plan_day(tmp_path):
    p = tmp_path / "account.yaml"
    p.write_text("base_currency: EUR\ncash_eur: 10.0\nrisk_profile: balanced\n"
                 "savings_plan_day: 15\n")
    assert load_account(str(p)).savings_plan_day == 15


def test_account_clamps_invalid_savings_plan_day(tmp_path):
    p = tmp_path / "account.yaml"
    p.write_text("risk_profile: balanced\nsavings_plan_day: 45\n")
    assert load_account(str(p)).savings_plan_day is None


def test_account_roundtrips_savings_plan_day(tmp_path):
    p = tmp_path / "account.yaml"
    save_account(AccountState("EUR", 50.0, "balanced", True, 7), str(p))
    a = load_account(str(p))
    assert a.savings_plan_day == 7


def test_account_without_key_is_none(tmp_path):
    p = tmp_path / "account.yaml"
    p.write_text("risk_profile: balanced\n")
    assert load_account(str(p)).savings_plan_day is None


# ── savings-plan routing ─────────────────────────────────────────────────────

def test_etf_routes_to_savings_plan():
    assert holding_routes_to_savings_plan("ETF")
    assert holding_routes_to_savings_plan("CASH")


def test_equity_does_not_route_to_savings_plan():
    assert not holding_routes_to_savings_plan("EQUITY")


# ── markets-closed freshness line ────────────────────────────────────────────

def test_markets_closed_on_weekend_with_friday_bar(monkeypatch):
    monkeypatch.setattr(render, "latest_bar_date", lambda: "2026-09-11")  # Friday
    line = render._markets_closed_line(today=date(2026, 9, 12))  # Saturday
    assert line == C.MARKETS_CLOSED.format(date="Friday 11 Sep 2026")


def test_markets_closed_empty_on_trading_day(monkeypatch):
    monkeypatch.setattr(render, "latest_bar_date", lambda: "2026-09-11")
    assert render._markets_closed_line(today=date(2026, 9, 14)) == ""  # Monday


def test_markets_closed_empty_without_bar(monkeypatch):
    monkeypatch.setattr(render, "latest_bar_date", lambda: "")
    assert render._markets_closed_line(today=date(2026, 9, 12)) == ""


# ── Today renders the line under the actions block ───────────────────────────

def _holding(symbol, status):
    return {"symbol": symbol, "name": symbol, "value_eur": 100.0,
            "current_weight": "50.0%", "target_weight": "50.0%",
            "status": status, "action": None, "blocked": False}


def _today(monkeypatch, portfolio_symbols, instrument_class, savings_day):
    monkeypatch.setattr(render, "read_history", lambda: pd.DataFrame())
    monkeypatch.setattr(render, "_render_value_chart", lambda h, a: None)
    monkeypatch.setattr(render, "latest_review",
                        lambda ok_only=False: {"review_status": "ok",
                                               "review_ts": "2026-09-14T18:00:00",
                                               "latest_bar": "2026-09-11"})
    monkeypatch.setattr(render, "read_regime", lambda: {"state": "estimated",
                                                        "label": "rising",
                                                        "confidence": "high"})
    monkeypatch.setattr(render, "read_actions",
                        lambda: [_holding(s, C.STATUS_ON_TRACK) for s in portfolio_symbols])
    monkeypatch.setattr(render, "load_portfolio",
                        lambda: pd.DataFrame({"Symbol": portfolio_symbols,
                                              "Amount_EUR": [100.0] * len(portfolio_symbols)}))
    monkeypatch.setattr(render, "load_account",
                        lambda: AccountState("EUR", 10.0, "balanced", True, savings_day))
    monkeypatch.setattr(render, "resolve_broker",
                        lambda s: {"isin": "X", "instrument_class": instrument_class})
    monkeypatch.setattr(render, "get_structure", lambda s: "PLAIN")
    monkeypatch.setattr(render, "latest_bar_date", lambda: "")
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page("pages/today.py").run()
    return _all_text(at)


def test_today_shows_countdown_when_savings_plan_holding(monkeypatch):
    text = _today(monkeypatch, ["EUNL.DE"], "ETF", 1)
    assert "Savings plan executes in" in text


def test_today_hides_countdown_without_savings_plan_holding(monkeypatch):
    text = _today(monkeypatch, ["AMZN"], "EQUITY", 1)
    assert "Savings plan executes" not in text


def test_today_hides_countdown_when_day_unset(monkeypatch):
    text = _today(monkeypatch, ["EUNL.DE"], "ETF", None)
    assert "Savings plan executes" not in text
