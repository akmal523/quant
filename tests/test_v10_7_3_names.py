"""
test_v10_7_3_names.py — names everywhere + funnel transparency (v10.7.3, Part 9.2/9.8).

Intent: prove the ONE display-name resolver and that rendered surfaces show a
company name (never a bare symbol when a registry name exists), and that Find
investments renders the funnel transparency line and the near-miss expander.

Invariants: tests never touch the network or the real systemd.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant import paths
from quant.data import database, names
from quant.ui import copy as C

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")
_TEXTLIKE = ("markdown", "info", "warning", "error", "caption", "text",
             "title", "header", "subheader", "button")


def _all_text(at: AppTest) -> str:
    chunks: list[str] = []
    for attr in _TEXTLIKE:
        for el in getattr(at, attr, []):
            val = getattr(el, "label", None)
            if val is None:
                val = getattr(el, "value", "")
            chunks.append(str(val))
    return " ".join(chunks)


def _seed_registry(symbol: str, display: str) -> None:
    conn = database.get_connection()
    conn.execute("DELETE FROM asset_registry WHERE symbol = ?", [symbol])
    conn.execute(
        "INSERT INTO asset_registry (symbol, name, display_name, instrument_class, "
        "isin, currency, universe_status) VALUES (?, ?, ?, 'ETF', '', '', 'ACTIVE')",
        [symbol, display, display],
    )


# ── The resolver chain ────────────────────────────────────────────────────────

def test_display_name_chain(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    _seed_registry("5J50.DE", "Global Aero & Defense")
    assert names.display_name("5J50.DE") == "Global Aero & Defense"
    # Cached probe path (written once by the daily run).
    (tmp_path / "universe_names.json").write_text(
        json.dumps({"MU": "Micron Technology"}), encoding="utf-8")
    assert names.display_name("MU") == "Micron Technology"
    # Fallback: an unknown symbol resolves to itself.
    assert names.display_name("ZZZZ") == "ZZZZ"


# ── Rendered surfaces show company names ──────────────────────────────────────

def test_holdings_table_shows_company_name(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    csv = tmp_path / "portfolio.csv"
    csv.write_text(
        "Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\n"
        "5J50.DE,10.0,1000.0,50.0\n", encoding="utf-8")
    monkeypatch.setattr(paths, "DATA_PORTFOLIO", str(csv))
    _seed_registry("5J50.DE", "Global Aero & Defense")

    at = AppTest.from_file(DASHBOARD, default_timeout=90)
    at.run()
    at.switch_page("pages/portfolio.py").run()
    assert not at.exception, f"My holdings raised: {at.exception}"

    seen: set[str] = set()
    for el in at.dataframe:
        frame = getattr(el, "value", None)
        if frame is not None and "Name" in getattr(frame, "columns", []):
            seen.update(frame["Name"].astype(str))
    assert "Global Aero & Defense" in seen
    assert "5J50.DE" not in seen


# ── Funnel transparency line + near misses ────────────────────────────────────

def test_funnel_line_and_near_misses(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    conn = database.get_connection()
    conn.execute("DELETE FROM funnel_survivors")
    conn.execute("DELETE FROM universe_master")
    for i in range(3):
        conn.execute("INSERT INTO universe_master (symbol) VALUES (?)", [f"U{i}"])
    # A symbol no other test scores, so it stays below the conviction bar.
    conn.execute(
        "INSERT INTO funnel_survivors (symbol, score, updated_at) VALUES ('ZZZZ', 50.0, 0)")

    at = AppTest.from_file(DASHBOARD, default_timeout=90)
    at.run()
    at.switch_page("pages/explore.py").run()
    assert not at.exception, f"Find investments raised: {at.exception}"
    text = _all_text(at)
    assert C.FUNNEL_LINE.format(entered=3, survived=1, conviction=0) in text
    # The near-miss caption renders inside the expander.
    assert "ZZZZ (ZZZZ): no scores yet" in text
