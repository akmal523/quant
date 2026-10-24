"""test_v10_8_8_phase2.py — v10.8.8 Phase 2: T2 (derived Holdings Analysis + drawer).

Ruling R-P2: status columns are DERIVED, never placeholders. The Reason cell is
computed from advice state and weights; the Alerts cell counts the open
Threat-verb triggers; the coverage line is the ONE helper and never a bare "-".
The drawer has a fixed four-section order, each with an explicit empty state.

Synthetic fixtures only; the user's real DB and data dir are never touched.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from quant.engine import holdings_analysis as HA
from quant.engine import manifest as M

# Run the analysis page directly: the dashboard's home page syncs the advice log
# as a side effect, which would mask the "no open advice" empty state.
ANALYSIS = str(Path(__file__).resolve().parents[1] / "quant" / "pages" / "analysis.py")


# ── Reason cell (pure) ────────────────────────────────────────────────────────

def test_reason_cell_with_open_advice():
    """R-P2: open advice => '<Verb> ~<amount> EUR, open since <date>'."""
    advice = [{"symbol": "EUNL.DE", "verb": "sell_part", "amount_eur": 65.0,
               "open_since": date(2026, 10, 1)}]
    r = HA.reason_cell("EUNL.DE", advice, drift=0.02, target=0.20)
    assert r == "Sell part ~65 EUR, open since 1 Oct 2026"


def test_reason_cell_without_open_advice():
    """R-P2: no open advice => the drift sentence; never 'held'."""
    r = HA.reason_cell("EUNL.DE", [], drift=0.023, target=0.20)
    assert r == "no open advice; drift +2.3 pp vs 20% target"
    assert "held" not in r.lower()


def test_reason_cell_ignores_other_symbols_advice():
    advice = [{"symbol": "AMZN", "verb": "buy", "amount_eur": 50.0,
               "open_since": date(2026, 10, 1)}]
    r = HA.reason_cell("EUNL.DE", advice, drift=0.0, target=0.20)
    assert r.startswith("no open advice")


# ── Alerts cell (pure) ────────────────────────────────────────────────────────

def test_alert_count_matches_the_advice_state():
    """R-P2: the Alerts count is the open Threat-verb triggers; 0 when none."""
    holdings = [{"symbol": "X", "tier": "SPECULATIVE", "entry_price": 100.0,
                 "current_price": 40.0}]
    assert HA.alert_count("X", holdings) == 1   # stop-loss hit
    assert HA.alert_count("Y", holdings) == 0   # truly zero


def test_alert_count_zero_when_no_threats():
    holdings = [{"symbol": "X", "tier": "ALPHA", "entry_price": 100.0,
                 "current_price": 100.0}]
    assert HA.alert_count("X", holdings) == 0


# ── Coverage line (pure) ──────────────────────────────────────────────────────

def test_coverage_line_never_a_bare_dash():
    """R-P2: a bare '-' for coverage is forbidden."""
    line = M.coverage_line()
    assert line.startswith("coverage:")
    assert line.strip() != "coverage: -"
    assert "not available" in line or "/" in line


def test_coverage_line_from_the_manifest():
    from quant.data.database import write_connection

    with write_connection() as conn:
        M.record_stage(conn, "test-cov-run", "scoring", M.STATUS_OK,
                       coverage="5/5 symbols")
    line = M.coverage_line()
    assert line.startswith("coverage:")
    assert "5/5 symbols" in line
    assert "as of" in line


# ── Grep guards ───────────────────────────────────────────────────────────────

def test_no_inline_coverage_dash_in_ui():
    """R-P2: coverage is rendered by the ONE helper, never inline."""
    ui = Path(__file__).resolve().parents[1] / "quant" / "ui"
    for p in ui.glob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert not re.search(r"coverage\s*\{", text), f"inline coverage in {p.name}"
        assert "coverage -" not in text, f"bare coverage dash in {p.name}"


def test_no_held_placeholder_in_ui():
    """R-P2: the Reason cell is never the placeholder 'held'."""
    pages = (Path(__file__).resolve().parents[1] / "quant" / "ui" / "pages.py"
             ).read_text(encoding="utf-8")
    assert '"Reason": "held"' not in pages
    assert "'Reason': 'held'" not in pages


# ── AppTest: the drawer's four sections ───────────────────────────────────────

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture
def seeded_analysis(tmp_path, monkeypatch):
    """EUNL.DE has news + SEC + open advice; AMZN has none."""
    from quant import paths
    from quant.data.database import write_connection

    csv = tmp_path / "portfolio.csv"
    csv.write_text(
        "Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\n"
        "EUNL.DE,100.0,500.0,0.0\n"
        "AMZN,150.0,300.0,0.0\n", encoding="utf-8")
    monkeypatch.setattr(paths, "DATA_PORTFOLIO", str(csv))
    with write_connection() as conn:
        conn.execute("DELETE FROM holdings_meta")
        conn.execute("DELETE FROM market_history")
        conn.execute("DELETE FROM nlp_evidence")
        conn.execute("DELETE FROM advice_log")
        for sym in ("EUNL.DE", "AMZN"):
            conn.execute(
                "INSERT INTO holdings_meta (symbol, shares, sync_date, invested_at_sync) "
                "VALUES (?, ?, ?, ?)", [sym, 1.0, "2026-10-13", 100.0])
            conn.execute(
                "INSERT OR REPLACE INTO market_history (Symbol, Date, Close) "
                "VALUES (?, ?, ?)", [sym, "2026-10-13", 100.0])
        conn.execute(
            "INSERT INTO nlp_evidence (symbol, source, title, published_at, score) "
            "VALUES (?, ?, ?, ?, ?)",
            ["EUNL.DE", "news", "World ETF rallies", "2026-10-13", 0.5])
        conn.execute(
            "INSERT INTO advice_log (verb, symbol, amount_eur, reason, horizon, "
            "class, created_at, open_since, status) "
            "VALUES (?, ?, ?, ?, ?, ?, now(), ?, ?)",
            ["sell_part", "EUNL.DE", 65.0, "over target", "Today", "ALPHA",
             "2026-10-01", "open"])
    return csv


def _analysis_page() -> AppTest:
    at = AppTest.from_file(ANALYSIS, default_timeout=90)
    at.run()
    return at


def _text(at: AppTest) -> str:
    chunks = []
    for attr in ("markdown", "info", "warning", "error", "caption", "text",
                 "title", "header", "subheader"):
        for el in getattr(at, attr, []):
            chunks.append(str(getattr(el, "value", "")))
    return " ".join(chunks)


def _holdings_frame(at: AppTest):
    for el in at.dataframe:
        v = getattr(el, "value", None)
        if v is not None and hasattr(v, "columns") and "Reason" in v.columns:
            return v
    return None


def test_holdings_analysis_cells_are_derived(seeded_analysis):
    """R-P2: the Reason/Alerts cells match the synthesized advice state."""
    at = _analysis_page()
    assert not at.exception, f"Full analysis raised: {at.exception}"
    df = _holdings_frame(at)
    assert df is not None, "no Holdings analysis table with a Reason column"
    assert "Alerts" in df.columns
    row = df[df["Symbol"].astype(str).str.contains("EUNL", na=False)]
    assert not row.empty
    reason = str(row.iloc[0]["Reason"])
    assert "Sell part" in reason and "open since" in reason
    assert reason != "held"
    assert str(row.iloc[0]["Alerts"]).isdigit()


def test_drawer_shows_all_sections_with_data(seeded_analysis):
    """Acceptance: all four sections present for a symbol with news data."""
    at = _analysis_page()
    at.selectbox(key="ha_inspect").select("EUNL.DE").run()
    assert not at.exception
    text = _text(at)
    for header in ("Price", "Score", "News", "Open advice"):
        assert header in text, f"drawer section {header!r} missing"
    assert "World ETF rallies (positive)" in text


def test_drawer_missing_data_shows_reasoned_empty_states(seeded_analysis):
    """Acceptance: a symbol without news/SEC shows reasoned empty states."""
    at = _analysis_page()
    at.selectbox(key="ha_inspect").select("AMZN").run()
    assert not at.exception
    text = _text(at)
    for header in ("Price", "Score", "News", "Open advice"):
        assert header in text, f"drawer section {header!r} missing"
    assert "no news fetched yet" in text
    assert "no open advice" in text
