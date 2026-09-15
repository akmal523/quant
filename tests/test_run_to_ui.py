"""
test_run_to_ui.py — Run -> UI integration guard (v10.5.3, spec 1.4).

Intent: the guard for the whole "artifact did not reach the UI" class. Seed the
fixture DB with 300 bars of IWDA.AS + two holdings, run the REAL review step, then
render Today/Explore via AppTest and assert the trend line, the action cards, and
a score bar equal the values in the run artifacts. If any wire is re-broken this
fails.

Hermetic seams: the review hard-wires a ProcessPoolExecutor, FinBERT, and network
fetchers, so the test injects an inline executor and stubs those collaborators.
No production behavior changes; the artifact-producing code path is the real one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

import quant.data.async_fetcher as async_fetcher  # noqa: E402
import quant.features.build_features as build_features  # noqa: E402
import quant.main as main_mod  # noqa: E402
from quant import paths  # noqa: E402
from quant.data.database import get_connection  # noqa: E402
from quant.reporting import artifacts  # noqa: E402
from quant.ui import copy as C  # noqa: E402

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")
SYMBOL = "IWDA.AS"
_TEXTLIKE = ("markdown", "info", "warning", "error", "caption", "text",
             "title", "header", "subheader")


def _all_text(at: AppTest) -> str:
    chunks: list[str] = []
    for attr in _TEXTLIKE:
        for el in getattr(at, attr, []):
            chunks.append(str(getattr(el, "value", "")))
    return " ".join(chunks)


# ── Hermetic collaborators ────────────────────────────────────────────────────

class _StubScorer:
    """FinBERT stand-in. No model download; neutral score."""

    def __init__(self, *a, **k) -> None:
        pass

    def score_document(self, text: str) -> dict:
        return {"score": 0.0, "reasoning": "stub", "doc_hash": None}


class _InlineExecutor:
    """ProcessPoolExecutor stand-in that runs the callable in-process."""

    def __init__(self, max_workers=None) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def submit(self, fn, *args, **kwargs):
        class _F:
            def __init__(self, value):
                self._value = value

            def result(self):
                return self._value

        return _F(fn(*args, **kwargs))


async def _no_texts(symbols) -> dict:
    return {s: "" for s in symbols}


def _raise_skip():
    raise RuntimeError("synthetic features unavailable in test")


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _seed_market_history() -> None:
    conn = get_connection()
    dates = pd.bdate_range("2024-01-01", periods=300).strftime("%Y-%m-%d")
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.normal(0.05, 1.0, 300))
    for d, c in zip(dates, close):
        conn.execute(
            "INSERT OR REPLACE INTO market_history "
            "(Date, Open, High, Low, Close, Volume, Symbol, Sector, Instrument_Class) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [d, float(c), float(c) + 1.0, float(c) - 1.0, float(c), 1e6,
             SYMBOL, "ETF", "ETF"],
        )
    conn.execute(
        "INSERT OR REPLACE INTO asset_registry "
        "(symbol, name, instrument_class, isin, universe_status, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [SYMBOL, "MSCI World (IWDA)", "ETF", "IE00B4L5Y983", "CORE", 0.0],
    )


def _write_portfolio(tmp_path) -> str:
    path = tmp_path / "portfolio.csv"
    path.write_text(
        "Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\n"
        f"{SYMBOL},100.00,100.00,0.00\n"
        "EUNL.DE,125.03,281.28,3.28\n",
        encoding="utf-8",
    )
    return str(path)


def _run_review(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(paths, "OUTPUTS_DIR", out)
    monkeypatch.setattr(paths, "DATA_PORTFOLIO", _write_portfolio(tmp_path))
    monkeypatch.setattr(artifacts, "OUTPUTS_DIR", str(out))
    monkeypatch.setattr(artifacts, "_RUN_DIR", None)
    monkeypatch.setattr(artifacts, "latest_run_dir", lambda: _latest(out))

    monkeypatch.setattr(main_mod, "NLPScorer", _StubScorer)
    monkeypatch.setattr(main_mod, "get_fundamentals", lambda _s: {})
    monkeypatch.setattr(main_mod, "notify_daily", lambda *a, **k: None)
    monkeypatch.setattr(main_mod, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(main_mod, "as_completed", lambda fs: list(fs))
    monkeypatch.setattr(async_fetcher, "fetch_all_texts_concurrently", _no_texts)
    monkeypatch.setattr(build_features, "latest_features", _raise_skip)

    _seed_market_history()
    main_mod.main()
    return out


def _latest(out: Path) -> str | None:
    runs = [d for d in out.iterdir() if d.is_dir() and d.name.startswith("run_")]
    return str(max(runs, key=lambda p: p.name)) if runs else None


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_run_artifacts_to_today_and_explore(tmp_path, monkeypatch):
    out = _run_review(tmp_path, monkeypatch)

    run_dir = _latest(out)
    assert run_dir, "review produced no run dir"
    metrics = json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))
    reg = metrics["regime"]
    assert reg["state"] in ("estimated", "insufficient_history", "failed")

    actions = artifacts.read_actions()
    scores = artifacts.read_scores(SYMBOL)
    assert scores["structural_grade"] is not None

    # Today renders the artifact trend line.
    today = AppTest.from_file(DASHBOARD, default_timeout=120)
    today.run()
    text = _all_text(today)
    if reg["state"] == "estimated":
        assert C.MARKET_TREND.format(label=reg["label"],
                                     confidence=reg["confidence"]) in text
    elif reg["state"] == "failed":
        assert C.MARKET_TREND_FAILED in text
    else:
        assert C.MARKET_TREND_INSUFFICIENT in text

    # Today renders the artifact action cards (or the nothing sentence).
    cards = [h for h in actions if h.get("action") and not h.get("blocked")]
    if cards:
        a = cards[0]
        target = (a.get("target_weight") or "").rstrip("%") or "?"
        pct = abs(float(str(a.get("drift", "0")).rstrip("%") or 0))
        template = C.ACTION_ADD if a["action"] == "BUY MORE" else C.ACTION_SELL
        expected = template.format(amount=f"{a['amount_eur']:.0f}", symbol=a["symbol"],
                                   name=a.get("name") or a["symbol"],
                                   pct=f"{pct:.0f}", target=target)
        assert expected.split(".")[0] in text, f"action card missing: {expected}"
    else:
        assert C.EMPTY_NOTHING_TO_DO in text

    # Explore renders the artifact score bar for the symbol.
    explore = AppTest.from_file(DASHBOARD, default_timeout=120)
    explore.run()
    explore.switch_page("pages/explore.py").run()
    explore.text_input[0].set_value(SYMBOL).run()
    explore_text = _all_text(explore)
    assert f"Overall score: {C.fmt_score(scores['active_score'])}" in explore_text
