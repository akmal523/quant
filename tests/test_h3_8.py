"""
test_h3_8.py — Advice engine oracle + legacy/mixed-run, discovery names (H3.8).

M4 the advice engine: drift vs threshold (oracle = v10.4 numbers), unified status.
M1 single-artifact header; M2 update-only latest dir keeps charts; M3 Growth
annotation percent-only; M5 discovery names; M6 status fallback; M7 legacy rows;
M8 generic own-session copy; M9 one catalogue as-of string.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

import quant.ui.render as render  # noqa: E402
from quant.data.database import get_connection  # noqa: E402
from quant.ui.copy import (  # noqa: E402
    CHART_NO_HISTORY,
    EMPTY_NO_MARKET_DATA,
    ERROR_RUNNING,
    HEADER_REVIEW_PREPARED,
    PAGE_SETTINGS,
    PAGE_TODAY,
    SCORES_AS_OF,
    STATUS_ON_TRACK,
    fmt_ts,
    status_for,
)
from quant.ui.search import discovery_candidates  # noqa: E402

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")
_TEXTLIKE = ("markdown", "info", "warning", "error", "caption", "text",
             "title", "header", "subheader")


def _all_text(at: AppTest) -> str:
    chunks = []
    for attr in _TEXTLIKE:
        for el in getattr(at, attr, []):
            chunks.append(str(getattr(el, "value", "")))
    for el in getattr(at, "expander", []):
        chunks.append(str(getattr(el, "label", "")))
    return " ".join(chunks)


def _page(page: str) -> AppTest:
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page(f"pages/{page.lower()}.py").run()
    return at


def _explore(monkeypatch, symbol, label):
    monkeypatch.setattr(render, "load_index", lambda: [])
    monkeypatch.setattr(render, "search", lambda idx, q, n=10: [
        {"symbol": symbol, "label": label, "name": symbol, "isin": ""}])
    monkeypatch.setattr(render, "read_scores", lambda s: {
        "structural_grade": None, "tactical_grade": None, "active_score": None})
    monkeypatch.setattr(render, "resolve_broker", lambda s: {
        "isin": "X", "currency": "EUR", "isin_source": "user", "instrument_class": "ETF"})
    monkeypatch.setattr(render, "load_news", lambda s: [])
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page("pages/explore.py").run()
    at.text_input[0].set_value("x").run()
    return _all_text(at)


# ─ M4: the advice engine oracle ─────────────────────────────────────────────

def test_m4_actions_oracle():
    from quant.portfolio.portfolio import enhanced_portfolio_audit
    from quant.reporting.actions import build_actions

    get_connection().execute("DELETE FROM rebalance_log")
    rows = [("5J50.DE", 8.28, 145.61, -5.39), ("AMZN", 220.55, 150.41, 0.3),
            ("EUNL.DE", 125.03, 281.28, 3.28), ("SXRV.DE", 1460.0, 253.1, -0.8)]
    portfolio = pd.DataFrame([
        dict(Symbol=s, Avg_Entry_Price=a, Current_Value_EUR=v, Broker_PnL_EUR=p)
        for s, a, v, p in rows])
    scan = pd.DataFrame([{"Symbol": s, "Current_Price": 1.0, "Structural_Grade": 50,
                          "Tactical_Grade": 50} for s in portfolio["Symbol"]])
    audit = enhanced_portfolio_audit(portfolio, scan, current_date="2026-09-14")
    eunl = audit[audit["Symbol"] == "EUNL.DE"].iloc[0]
    sxrv = audit[audit["Symbol"] == "SXRV.DE"].iloc[0]
    assert eunl["Drift"] == "-16.1%"
    assert sxrv["Drift"] == "10.5%"
    acts = {a["symbol"]: a for a in build_actions(audit) if not a["blocked"]}
    assert acts["EUNL.DE"]["action"] == "BUY MORE"
    assert acts["SXRV.DE"]["action"] == "TRIM"


def test_status_from_drift_not_on_track_for_overshoot():
    # A 7.5-point SECTOR overshoot is never "On track".
    status = status_for("HOLD: SECTOR wait 10d", drift_frac=0.075, threshold=0.05)
    assert status != STATUS_ON_TRACK


# ── M1: single-artifact header ───────────────────────────────────────────────

def test_today_header_single_artifact_no_borrowed_close(monkeypatch):
    monkeypatch.setattr(render, "latest_review",
                        lambda ok_only=False: {"review_ts": "2026-09-13"})
    text = _all_text(_page(PAGE_TODAY))
    assert HEADER_REVIEW_PREPARED.format(prepared="Sunday 13 Sep 2026") in text
    assert "Review of unknown close" not in text


# ── M2: an update-only latest dir keeps charts ───────────────────────────────

def test_explore_history_survives_update_only_latest(monkeypatch):
    conn = get_connection()
    conn.execute("DELETE FROM market_history WHERE Symbol='ZZHIST'")
    conn.execute("INSERT INTO market_history (Date, Symbol, Close) "
                 "VALUES ('2026-09-14','ZZHIST',10.0)")
    conn.execute("INSERT INTO market_history (Date, Symbol, Close) "
                 "VALUES ('2026-09-13','ZZHIST',9.0)")
    try:
        text = _explore(monkeypatch, "ZZHIST", "ZZHIST")
        assert CHART_NO_HISTORY.format(name="ZZHIST") not in text
    finally:
        conn.execute("DELETE FROM market_history WHERE Symbol='ZZHIST'")


# ── M3: Growth annotation is percent-only ────────────────────────────────────

def test_growth_annotation_is_percent_only():
    df = pd.DataFrame({"review_ts": pd.to_datetime(["2026-06-01", "2026-09-13"]),
                       "value_eur": [100.0, 104.2]})
    growth = render._range_annotation(df, True)
    value = render._range_annotation(df, False)
    assert "EUR" not in growth
    assert "EUR" in value


# ── M5: discovery names ──────────────────────────────────────────────────────

def test_discovery_sentence_with_real_name():
    conn = get_connection()
    conn.execute("DELETE FROM universe_master WHERE symbol='ZZAAPL'")
    conn.execute("INSERT INTO universe_master (symbol, name) VALUES ('ZZAAPL','Apple Computer')")
    try:
        cands = discovery_candidates("apple")
        assert any(c["symbol"] == "ZZAAPL" for c in cands)
    finally:
        conn.execute("DELETE FROM universe_master WHERE symbol='ZZAAPL'")


# ── M6: status fallback under a locked read ──────────────────────────────────

def test_settings_status_falls_back_to_update_state(monkeypatch):
    conn = get_connection()
    conn.execute("DELETE FROM market_history")
    monkeypatch.setattr(render, "read_update_state", lambda: {
        "ts": "2026-09-14T12:00:00", "instruments": 5, "prices_through": "2026-09-14"})
    try:
        text = _all_text(_page(PAGE_SETTINGS))
        assert "5 instruments" in text
        assert "prices through 14 Sep 2026" in text
        assert EMPTY_NO_MARKET_DATA not in text
    finally:
        pass


# ── M7: legacy rows kept, phantom hidden ─────────────────────────────────────

def test_reviews_keep_legacy_rows_hide_phantom(tmp_path, monkeypatch):
    from quant.reporting import artifacts

    monkeypatch.setattr(artifacts, "OUTPUTS_DIR", str(tmp_path))
    d = tmp_path / "run_2026-09-13_232316"
    d.mkdir()
    (d / "metrics.json").write_text(json.dumps({"review_ts": "2026-09-13"}),
                                    encoding="utf-8")
    conn = get_connection()
    conn.execute("DELETE FROM portfolio_history")
    from datetime import datetime
    ok_ts = datetime(2026, 9, 13, 22, 33, 0)
    phantom = datetime(2026, 9, 13, 23, 24, 0)
    for ts, val in ((ok_ts, 830.4), (None, 830.4), (phantom, 830.4)):
        conn.execute("INSERT INTO portfolio_history (review_ts, value_eur, invested_eur, "
                     "cash_eur, pnl_eur) VALUES (?, ?, 0, 0, 0)", [ts, val])
    try:
        text = _all_text(_page(PAGE_SETTINGS))
        assert fmt_ts(ok_ts) in text
        assert fmt_ts(phantom) not in text
    finally:
        conn.execute("DELETE FROM portfolio_history")


# ── M8: generic own-session copy ─────────────────────────────────────────────

def test_own_session_copy_is_generic():
    assert ERROR_RUNNING == "An operation is already running."


# ─ M9: one catalogue as-of string ───────────────────────────────────────────

def test_explore_uses_only_catalogue_as_of(monkeypatch):
    monkeypatch.setattr(render, "latest_review",
                        lambda ok_only=False: {"review_ts": "2026-09-13T23:23:00"})
    text = _explore(monkeypatch, "AMZN", "Amazon (AMZN)")
    assert SCORES_AS_OF.format(date="Sunday 13 Sep 2026, 23:23") in text
    assert "From the review of" not in text
