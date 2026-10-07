"""test_v10_8_0_reconcile.py — the broker-total reconciliation check (v10.8.0).

Intent (redesign 3.1, defect 2.8): the My holdings page lets the user enter the
total their broker shows. When the sum of the entered positions differs from
that total by more than one euro, a plain warning names both figures and the
difference. When they agree, a quiet caption confirms it.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from quant import paths
from quant.ui import copy

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")
_TEXTLIKE = ("markdown", "info", "warning", "error", "caption", "text",
             "title", "header", "subheader")


def _all_text(at: AppTest) -> str:
    return " ".join(str(getattr(el, "value", ""))
                    for attr in _TEXTLIKE for el in getattr(at, attr, []))


def _seed(tmp_path, monkeypatch):
    csv = tmp_path / "portfolio.csv"
    csv.write_text(
        "Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\n"
        "AAA,10.0,500.0,0.0\n"
        "BBB,10.0,352.23,0.0\n", encoding="utf-8")
    monkeypatch.setattr(paths, "DATA_PORTFOLIO", str(csv))
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)


def _portfolio_page(monkeypatch, tmp_path) -> AppTest:
    _seed(tmp_path, monkeypatch)
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page("pages/portfolio.py").run()
    return at


def test_mismatch_shows_plain_warning(tmp_path, monkeypatch):
    at = _portfolio_page(monkeypatch, tmp_path)
    # The positions sum to 852.23; the broker shows 1075.94.
    at.number_input(key="reconcile_total").set_value(1075.94).run()
    text = _all_text(at)
    assert "add up to" in text
    assert "1,075.94" in text or "1075.94" in text


def test_match_shows_quiet_caption(tmp_path, monkeypatch):
    at = _portfolio_page(monkeypatch, tmp_path)
    at.number_input(key="reconcile_total").set_value(852.23).run()
    assert copy.RECONCILE_MATCH in _all_text(at)


def test_zero_funnel_is_reported_honestly():
    """An empty funnel cache reports zeros, never a fabricated candidate."""
    from quant.ui import render

    counts = render._funnel_counts()
    assert counts["entered"] == 0
    assert counts["survived"] == 0
    assert render._candidate_list() == {"long": [], "active": [], "bets": []}
