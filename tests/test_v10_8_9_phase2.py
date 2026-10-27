"""test_v10_8_9_phase2.py — v10.8.9 Phase 2: T2 (chart overlay rules).

Ruling R-F2: chart overlays are view-bound. The benchmark toggle exists only in
Growth view; its state is preserved across view switches. The chart carries no
caption that duplicates a header metric (the header keeps Total value, the delta
badge, "Prices as of" and "since first purchase").

Synthetic fixtures only; the user's real DB and data dir are never touched
(tests/conftest.py isolates the session in an ephemeral DuckDB file).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from quant.ui import copy as C

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")

# A caption that restates a header metric ("+5.2% since 13 Sep 2026").
_DUPLICATE_CAPTION = re.compile(r"[+-]\d+(\.\d+)?% since \d{1,2} \w{3} \d{4}")

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402


@pytest.fixture
def seeded_chart(tmp_path, monkeypatch):
    """A portfolio with two stored total points so the chart (and its view
    control) actually renders."""
    from quant import paths
    from quant.data.database import write_connection
    from quant.engine import mtm

    csv = tmp_path / "portfolio.csv"
    csv.write_text(
        "Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\n"
        "EUNL.DE,100.0,102.0,2.0\n", encoding="utf-8")
    monkeypatch.setattr(paths, "DATA_PORTFOLIO", str(csv))
    # Every AppTest run re-executes the dashboard, so its startup valuation
    # catch-up runs too; with newer prices left by an earlier test it rebuilds
    # the store and wipes the two seeded points (order-dependent failure).
    monkeypatch.setattr(mtm, "catch_up", lambda *a, **k: {"ok": True, "skipped": True})
    with write_connection() as conn:
        conn.execute("DELETE FROM portfolio_total")
        conn.execute(
            "INSERT INTO portfolio_total (date, value_eur, cash_eur) "
            "VALUES (?, ?, NULL)", ["2026-10-12", 100.0])
        conn.execute(
            "INSERT INTO portfolio_total (date, value_eur, cash_eur) "
            "VALUES (?, ?, NULL)", ["2026-10-13", 102.0])
    return csv


def _portfolio() -> AppTest:
    at = AppTest.from_file(DASHBOARD, default_timeout=90)
    at.run()
    return at


def _checkbox_labels(at: AppTest) -> list[str]:
    return [str(getattr(cb, "label", "")) for cb in at.checkbox]


def _benchmark(at: AppTest):
    for cb in at.checkbox:
        if str(getattr(cb, "label", "")) == C.LABEL_BENCHMARK:
            return cb
    return None


def _text(at: AppTest) -> str:
    chunks = []
    for attr in ("markdown", "caption", "text", "info", "warning", "error"):
        for el in getattr(at, attr, []):
            chunks.append(str(getattr(el, "value", "")))
    return " ".join(chunks)


# ── F2: the benchmark toggle is Growth-only ───────────────────────────────────

def test_benchmark_checkbox_absent_in_value_view(seeded_chart):
    """Acceptance: the default (Value) view has no benchmark checkbox node."""
    at = _portfolio()
    assert not at.exception, f"Portfolio raised: {at.exception}"
    assert C.LABEL_BENCHMARK not in _checkbox_labels(at)


def test_benchmark_checkbox_present_in_growth_view(seeded_chart):
    """Acceptance: switching to Growth view renders the benchmark checkbox."""
    at = AppTest.from_file(DASHBOARD, default_timeout=90)
    at.session_state["val_mode"] = C.GROWTH_VIEW
    at.run()
    assert not at.exception, f"Portfolio raised: {at.exception}"
    assert C.LABEL_BENCHMARK in _checkbox_labels(at)


def test_benchmark_state_survives_a_view_switch(seeded_chart):
    """R-F2: toggling in Growth, leaving and returning keeps the value."""
    at = AppTest.from_file(DASHBOARD, default_timeout=90)
    at.session_state["val_mode"] = C.GROWTH_VIEW
    at.run()
    cb = _benchmark(at)
    assert cb is not None, "no benchmark checkbox in Growth view"
    cb.check().run()
    assert _benchmark(at) is not None
    assert _benchmark(at).value is True

    at.session_state["val_mode"] = C.VALUE
    at.run()
    assert C.LABEL_BENCHMARK not in _checkbox_labels(at)

    at.session_state["val_mode"] = C.GROWTH_VIEW
    at.run()
    restored = _benchmark(at)
    assert restored is not None, "the checkbox did not return in Growth view"
    assert restored.value is True, "the benchmark state was not preserved"


# ── F3: no caption duplicates a header metric ─────────────────────────────────

def test_no_caption_duplicates_a_header_metric(seeded_chart):
    """Acceptance: no text node restates '<+x%> since <date>' outside the header."""
    at = _portfolio()
    text = _text(at)
    m = _DUPLICATE_CAPTION.search(text)
    assert not m, f"a chart caption duplicates a header metric: {m.group(0)!r}"
    # The header's own line is "since first purchase: <x%>" and does not match.
    assert "since first purchase" in text


def test_header_line_is_not_matched_by_the_pattern():
    """Self-test: the allowed header line never matches the banned pattern."""
    sample = "since first purchase: +2.0%"
    assert not _DUPLICATE_CAPTION.search(sample)
    assert _DUPLICATE_CAPTION.search("+5.2% since 13 Sep 2026")
