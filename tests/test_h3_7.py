"""
test_h3_7.py — Legacy-artifact semantics, update freshness, chart legibility (H3.7).

L1 legacy (no review_status) counts as ok and run_latest never shadows a review.
L2 the Settings status line reads the update artifact (live count/date).
L3 the Reviews list drops rows newer than the last ok review.
L4 fmt_review_ts omits a bogus 00:00 for a date-only source.
L5 sub-3-day ranges label by hh:mm; the annotation sits in the top margin.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

import quant.ui.render as render  # noqa: E402
from quant.data.database import get_connection  # noqa: E402
from quant.reporting import artifacts  # noqa: E402
from quant.ui.copy import (  # noqa: E402
    GUIDE_NO_REVIEW,
    PAGE_SETTINGS,
    PAGE_TODAY,
    fmt_review_ts,
    fmt_ts,
)

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")
_TEXTLIKE = ("markdown", "info", "warning", "error", "caption", "text",
             "title", "header", "subheader")


def _all_text(at: AppTest) -> str:
    chunks = []
    for attr in _TEXTLIKE:
        for el in getattr(at, attr, []):
            chunks.append(str(getattr(el, "value", "")))
    return " ".join(chunks)


def _page(page: str) -> AppTest:
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page(f"pages/{page.lower()}.py").run()
    return at


def _legacy_outputs(tmp_path, name="run_2026-09-13_232316", payload=None):
    d = tmp_path / name
    d.mkdir()
    (d / "metrics.json").write_text(
        json.dumps(payload or {"review_ts": "2026-09-13"}), encoding="utf-8")
    (tmp_path / "run_latest").mkdir()          # non-timestamped artifact
    return str(d)


# ── L1: legacy ok + run_latest exclusion ─────────────────────────────────────

def test_legacy_review_counts_as_ok_and_run_latest_excluded(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "OUTPUTS_DIR", str(tmp_path))
    _legacy_outputs(tmp_path)
    assert artifacts.latest_review(ok_only=True).get("review_ts") == "2026-09-13"
    assert artifacts.latest_review().get("review_ts") == "2026-09-13"
    assert artifacts.latest_ok_run_dir().endswith("run_2026-09-13_232316")


def test_today_renders_header_on_legacy_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "OUTPUTS_DIR", str(tmp_path))
    _legacy_outputs(tmp_path)
    text = _all_text(_page(PAGE_TODAY))
    assert GUIDE_NO_REVIEW not in text
    assert "Review of" in text


# ─ L3: phantom-row filter ───────────────────────────────────────────────────

def test_review_cutoff_and_phantom_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "OUTPUTS_DIR", str(tmp_path))
    _legacy_outputs(tmp_path)
    assert artifacts.latest_ok_review_ts() == datetime(2026, 9, 13, 23, 23, 16)

    conn = get_connection()
    conn.execute("DELETE FROM portfolio_history")
    ok_ts = datetime(2026, 9, 13, 22, 33, 0)
    phantom = datetime(2026, 9, 13, 23, 24, 0)
    for ts in (ok_ts, phantom):
        conn.execute("INSERT INTO portfolio_history (review_ts, value_eur, "
                     "invested_eur, cash_eur, pnl_eur) VALUES (?, 830.4, 0, 0, 0)",
                     [ts])
    try:
        text = _all_text(_page(PAGE_SETTINGS))
        assert fmt_ts(ok_ts) in text
        assert fmt_ts(phantom) not in text
    finally:
        conn.execute("DELETE FROM portfolio_history")


# ── L2: update freshness line ────────────────────────────────────────────────

def test_settings_status_uses_update_artifact(monkeypatch):
    conn = get_connection()
    conn.execute("DELETE FROM market_history")
    conn.execute("INSERT INTO market_history (Date, Symbol, Close) "
                 "VALUES ('2026-09-14','AMZN',100.0)")
    conn.execute("INSERT INTO market_history (Date, Symbol, Close) "
                 "VALUES ('2026-09-14','AAPL',200.0)")
    monkeypatch.setattr(render, "read_update_state",
                        lambda: {"ts": "2026-09-14T12:02:04", "instruments": 2})
    try:
        text = _all_text(_page(PAGE_SETTINGS))
        assert "2 instruments" in text
        assert "prices through 14 Sep 2026" in text
        assert "refreshed 14 Sep 2026, 12:02" in text
    finally:
        conn.execute("DELETE FROM market_history")


def test_update_state_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "OUTPUTS_DIR", str(tmp_path))
    artifacts.write_update_state({"ts": "2026-09-14T12:00:00", "instruments": 7})
    assert artifacts.read_update_state()["instruments"] == 7


# ── L4: review timestamp formatter ───────────────────────────────────────────

def test_fmt_review_ts_omits_bogus_time():
    assert fmt_review_ts("2026-09-13") == "Sunday 13 Sep 2026"
    assert fmt_review_ts("2026-09-13T23:23:00") == "Sunday 13 Sep 2026, 23:23"
    assert fmt_review_ts(datetime(2026, 9, 13, 23, 23)) == "Sunday 13 Sep 2026, 23:23"


# ── L5: chart legibility ─────────────────────────────────────────────────────

def test_axis_tickformat_sub_three_days():
    same_day = pd.DataFrame({"review_ts": pd.to_datetime(
        ["2026-09-13T09:00:00", "2026-09-13T11:00:00"])})
    wide = pd.DataFrame({"review_ts": pd.to_datetime(
        ["2026-06-01", "2026-09-01"])})
    assert render._axis_tickformat(same_day) == "%H:%M"
    assert render._axis_tickformat(wide) == "%d %b"


def test_annotation_sits_in_top_margin_on_white_box():
    kw = render._annotation_kwargs("+4.2% since 1 Jun 2026 (34.80 EUR)")
    assert kw["yref"] == "paper" and kw["y"] == 1.0
    assert kw["bgcolor"]
