"""test_v10_8_5.py — v10.8.5 phase tests.

Phase A (T1): the Full analysis Research tab renders (no StreamlitDuplicateElementId);
its funnel table appears; every page has a content marker (never a blank page).
Phase B (T2): the MTM store writes one point per trading day and the day delta equals
total(D) − total(D−1).
Phase C (T3): the five decision verb classes fire on their triggers.
Phase D (T5/T6): the Holdings analysis drill-down renders; no duplicated ticker/word.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")
_PAGES = {
    "Portfolio": "pages/portfolio.py",
    "Update holdings": "pages/update.py",
    "History": "pages/history.py",
    "Full analysis": "pages/analysis.py",
    "Settings": "pages/settings.py",
}


def _run(page: str) -> AppTest:
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page(page).run()
    return at


def _text(at: AppTest) -> str:
    chunks = []
    for attr in ("markdown", "info", "warning", "error", "caption", "text",
                 "title", "header", "subheader"):
        for el in getattr(at, attr, []):
            chunks.append(str(getattr(el, "value", "")))
    return " ".join(chunks)


def _frames(at: AppTest) -> list:
    return [getattr(el, "value", None) for el in getattr(at, "dataframe", [])]


def _frames_text(at: AppTest) -> str:
    out = []
    for v in _frames(at):
        if v is not None:
            try:
                out.append(str(v.to_string()))
            except Exception:  # noqa: BLE001
                out.append(str(v))
    return " ".join(out)


# ── Phase A (T1) ──────────────────────────────────────────────────────────────

def test_full_analysis_renders_without_crash():
    """D1/D2: the Research tab no longer raises StreamlitDuplicateElementId."""
    at = _run("pages/analysis.py")
    assert not at.exception, f"Full analysis raised: {at.exception}"


def test_research_tab_shows_funnel_or_explicit_empty_state():
    """T1: Research contains the funnel table OR a reason + Run analysis, never blank."""
    at = _run("pages/analysis.py")
    assert not at.exception
    has_stage = any(
        v is not None and hasattr(v, "columns") and "Stage" in v.columns
        for v in _frames(at))
    text = _text(at)
    assert has_stage or "Run analysis" in text, "Research tab rendered nothing"


def test_every_page_has_a_content_marker():
    """T7: a blank page is a bug — each page shows a table row or an empty state."""
    for title, page in _PAGES.items():
        at = _run(page)
        assert not at.exception, f"{title} raised: {at.exception}"
        has_rows = any(
            v is not None and hasattr(v, "shape") and v.shape[0] >= 1
            for v in _frames(at))
        assert has_rows or _text(at).strip(), f"{title} rendered nothing"


def test_research_has_an_inspect_control_when_rows_exist():
    """v10.8.6 (A1): the drawer is one lazy inspect select, not per-row expanders."""
    at = _run("pages/analysis.py")
    assert not at.exception
    has_stage = any(
        v is not None and hasattr(v, "columns") and "Stage" in v.columns
        for v in _frames(at))
    if has_stage:
        labels = [str(getattr(sb, "label", "")) for sb in getattr(at, "selectbox", [])]
        assert any("Inspect" in label for label in labels)


# ── Phase B (T2): per-trading-day MTM + day delta ─────────────────────────────

def test_backfill_writes_one_point_per_trading_day():
    from quant.data.database import write_connection
    from quant.engine import mtm
    from quant.reporting.artifacts import read_value_series

    sym = "TESTBF"
    with write_connection() as conn:
        conn.execute("DELETE FROM valuation")
        conn.execute("DELETE FROM portfolio_total")
        conn.execute("DELETE FROM holdings_meta")
        conn.execute("DELETE FROM trades")
        conn.execute("DELETE FROM flows")
        conn.execute("DELETE FROM market_history WHERE Symbol = ?", [sym])
        conn.execute(
            "INSERT INTO holdings_meta (symbol, shares, sync_date, invested_at_sync) "
            "VALUES (?, 10.0, ?, 100.0)", [sym, date(2026, 7, 1)])
        for d, close in ((date(2026, 7, 1), 10.0), (date(2026, 7, 2), 11.0),
                         (date(2026, 7, 3), 12.0)):
            conn.execute(
                "INSERT OR REPLACE INTO market_history (Symbol, Date, Close) "
                "VALUES (?, ?, ?)", [sym, str(d), close])
        mtm.backfill(conn, fx_lookup=lambda _s: 1.0, start=date(2026, 7, 1),
                     end=date(2026, 7, 3), source="test")

    series = read_value_series()
    assert len(series) == 3, "the series must have one point per trading day"
    vals = [float(v) for v in series["value_eur"]]
    assert vals == [100.0, 110.0, 120.0]
    eur, _pct, when = mtm.day_delta()
    assert abs(eur - (vals[-1] - vals[-2])) < 1e-9
    assert str(when)[:10] == "2026-07-03"
    assert str(mtm.latest_price_date())[:10] == "2026-07-03"
    per = mtm.read_symbol_series()
    assert len(per.get(sym, [])) == 3


# ── Phase C (T3): decision taxonomy ───────────────────────────────────────────

def test_taxonomy_rebalance_from_drift():
    from quant.engine import action_taxonomy as T

    h = [{"symbol": "AAA", "name": "AAA", "tier": "ALPHA",
          "current_weight": 0.40, "target_weight": 0.20}]
    items = T.rebalance_items(h)
    assert len(items) == 1 and items[0]["verb"] == T.VERB_REBALANCE
    assert "off its" in items[0]["reason"]


def test_taxonomy_fortress_never_rebalances():
    from quant.engine import action_taxonomy as T

    h = [{"symbol": "EUNL.DE", "name": "World", "tier": "FORTRESS",
          "current_weight": 0.80, "target_weight": 0.50}]
    assert T.rebalance_items(h) == []


def test_taxonomy_threat_spec_stop():
    from quant.engine import action_taxonomy as T

    h = [{"symbol": "BBB", "name": "BBB", "tier": "SPECULATIVE", "pnl_pct": -0.55}]
    items = T.threat_items(h)
    assert items and items[0]["verb"] == T.VERB_THREAT
    assert "stop-loss" in items[0]["reason"]


def test_taxonomy_threat_score_floor():
    from quant.engine import action_taxonomy as T

    h = [{"symbol": "CCC", "name": "CCC", "tier": "ALPHA"}]
    items = T.threat_items(h, scores={"CCC": {"structural_grade": 30}})
    assert any(i["verb"] == T.VERB_THREAT and "floor" in i["reason"] for i in items)


def test_taxonomy_swap_pairs_a_target():
    from quant.engine import action_taxonomy as T

    h = [{"symbol": "DDD", "name": "DDD", "tier": "ALPHA", "value_eur": 100.0}]
    ideas = [{"symbol": "EEE", "name": "EEE", "score": 71}]
    items = T.swap_items(h, ideas, scores={"DDD": {"active_score": 42}})
    assert len(items) == 1 and items[0]["verb"] == T.VERB_SWAP
    assert items[0]["target_symbol"] == "EEE"


def test_taxonomy_plan_change_requires_k_months():
    from quant.engine import action_taxonomy as T

    h = [{"symbol": "FFF", "name": "FFF", "tier": "FORTRESS",
          "current_weight": 0.20, "target_weight": 0.50}]
    assert T.plan_change_items(h, months_under=2) == []
    items = T.plan_change_items(h, months_under=3)
    assert items and items[0]["verb"] == T.VERB_PLAN


def test_taxonomy_build_ranks_today_before_this_month():
    from quant.engine import action_taxonomy as T

    h = [{"symbol": "AAA", "tier": "SPECULATIVE", "pnl_pct": -0.6, "value_eur": 50},
         {"symbol": "FFF", "tier": "FORTRESS", "current_weight": 0.2,
          "target_weight": 0.5}]
    items = T.build(h, months_under=3)
    horizons = [i["horizon"] for i in items]
    assert horizons == sorted(horizons, key=lambda x: 0 if x == "Today" else 1)


# ── Phase D (T5/T6): holdings drill-down + no duplicated ticker/word ──────────

def test_holdings_analysis_tab_renders():
    """T5: the Holdings analysis tab shows a table and opens per-row drawers."""
    at = _run("pages/analysis.py")
    assert not at.exception
    assert any(
        v is not None and hasattr(v, "columns") and "Drift (pp)" in v.columns
        for v in _frames(at)) or "No holdings yet" in _text(at)


def test_no_duplicated_ticker_or_word_in_the_ui():
    """T6/D8/D14: no ') (' duplication and no repeated word on any page."""
    import re

    for title, page in _PAGES.items():
        at = _run(page)
        assert not at.exception, f"{title} raised: {at.exception}"
        text = _text(at) + " " + _frames_text(at)
        assert not re.search(r"\)\s*\(", text), f"adjacent paren groups on {title}"
        dup = re.search(r"\b([A-Za-z]{3,})\s+\1\b", text)
        assert not dup, f"repeated word {dup.group(0)!r} on {title}"
