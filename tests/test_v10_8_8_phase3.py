"""test_v10_8_8_phase3.py — v10.8.8 Phase 3: T3 (delta-badge contract).

Ruling R-P3: ONE pure helper owns every delta badge in the product: sign ->
icon/color/text with explicit rounding rules. "-0.0%" is unrepresentable. No
inline badge formatting exists anywhere else in the UI.

Synthetic fixtures only; the user's real DB and data dir are never touched.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from quant.ui import copy as C
from quant.ui import palette as P

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")

_UP = "\u25b2"
_DOWN = "\u25bc"
_FLAT = "\u2022"


# ── The case table (pure) ─────────────────────────────────────────────────────

def test_delta_badge_case_table():
    """Acceptance: the three specified cases map to the exact icon/color/text."""
    assert C.delta_badge(-2, -0.02) == (_DOWN, P.LOSS, "-2 EUR (-<0.1%)")
    assert C.delta_badge(11, 1.0) == (_UP, P.GAIN, "+11 EUR (+1.0%)")
    assert C.delta_badge(0, 0) == (_FLAT, P.TEXT_MUTED, "+0 EUR (+0.0%)")


def test_delta_badge_never_minus_zero():
    """R-P3: '-0.0%' is unrepresentable."""
    for eur, pct in [(-2, -0.02), (2, 0.02), (0, 0), (5, 0.0), (-5, 0.0)]:
        _, _, text = C.delta_badge(eur, pct)
        assert "-0.0%" not in text, f"badge text {text!r} contains -0.0%"


def test_delta_badge_sign_drives_icon_and_color():
    """R-P3: the icon/color come from the SIGN OF EUR, not the percent."""
    assert C.delta_badge(-2, 5.0)[0] == _DOWN
    assert C.delta_badge(-2, 5.0)[1] == P.LOSS
    assert C.delta_badge(2, -5.0)[0] == _UP
    assert C.delta_badge(2, -5.0)[1] == P.GAIN
    # |eur| < 0.5 -> neutral, even with a non-zero percent.
    assert C.delta_badge(0.2, 3.0)[0] == _FLAT
    assert C.delta_badge(0.2, 3.0)[1] == P.TEXT_MUTED


# ── Grep guard: no inline badge formatting ────────────────────────────────────

def test_no_inline_badge_formatting_in_ui():
    """R-P3: only copy.delta_badge formats a delta badge."""
    ui = Path(__file__).resolve().parents[1] / "quant" / "ui"
    pat = re.compile(r":\+[,.0-9]*f\}\s*EUR\s*\(")
    for p in ui.glob("*.py"):
        if p.name == "copy.py":
            continue
        text = p.read_text(encoding="utf-8")
        assert not pat.search(text), f"inline badge formatting in {p.name}"


# ── AppTest: the Portfolio header badge on a down day ─────────────────────────

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture
def seeded_down_day(tmp_path, monkeypatch):
    """A portfolio whose stored total fell from 100 to 98 (a down day).

    The ledger (holdings_meta + two market bars) is seeded too, so the startup
    valuation catch-up reproduces the same down day whether or not it runs.
    """
    from quant import paths
    from quant.data.database import write_connection

    csv = tmp_path / "portfolio.csv"
    csv.write_text(
        "Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\n"
        "EUNL.DE,100.0,98.0,-2.0\n", encoding="utf-8")
    monkeypatch.setattr(paths, "DATA_PORTFOLIO", str(csv))
    with write_connection() as conn:
        conn.execute("DELETE FROM portfolio_total")
        conn.execute("DELETE FROM valuation")
        conn.execute("DELETE FROM holdings_meta")
        conn.execute("DELETE FROM market_history WHERE Symbol = ?", ["EUNL.DE"])
        conn.execute(
            "INSERT INTO holdings_meta (symbol, shares, sync_date, invested_at_sync) "
            "VALUES (?, ?, ?, ?)", ["EUNL.DE", 1.0, "2026-10-12", 100.0])
        conn.execute(
            "INSERT OR REPLACE INTO market_history (Symbol, Date, Close) "
            "VALUES (?, ?, ?)", ["EUNL.DE", "2026-10-12", 100.0])
        conn.execute(
            "INSERT OR REPLACE INTO market_history (Symbol, Date, Close) "
            "VALUES (?, ?, ?)", ["EUNL.DE", "2026-10-13", 98.0])
        conn.execute(
            "INSERT INTO portfolio_total (date, value_eur, cash_eur) "
            "VALUES (?, ?, NULL)", ["2026-10-12", 100.0])
        conn.execute(
            "INSERT INTO portfolio_total (date, value_eur, cash_eur) "
            "VALUES (?, ?, NULL)", ["2026-10-13", 98.0])
    return csv


def test_portfolio_header_badge_on_a_down_day(seeded_down_day):
    """Acceptance: a down day shows the down icon, red style, no '-0.0%'."""
    at = AppTest.from_file(DASHBOARD, default_timeout=90)
    at.run()
    assert not at.exception, f"Portfolio raised: {at.exception}"
    html = " ".join(str(m.value) for m in at.markdown)
    assert _DOWN in html, "no down icon in the header badge"
    assert P.LOSS in html, "the down badge is not red"
    assert "-0.0%" not in html
