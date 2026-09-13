"""
test_regime_masking.py — Regime masking contract (v10.5.2, A2).

Two cases:
  1. A review with a regime in metrics.json renders a regime word on Today.
  2. A forced regime failure (regime_error, no market_regime) surfaces in Health
     and Today as "unavailable"; it must NOT be masked as the missing-history
     empty state.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from quant.portfolio import history  # noqa: E402
from quant.reporting import artifacts  # noqa: E402
from quant.ui import copy as C  # noqa: E402

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")

_TEXTLIKE = ("markdown", "info", "warning", "error", "caption", "text",
             "title", "header", "subheader")


def _all_text(at: AppTest) -> str:
    chunks: list[str] = []
    for attr in _TEXTLIKE:
        for el in getattr(at, attr, []):
            chunks.append(str(getattr(el, "value", "")))
    return " ".join(chunks)


_PAGE_FILE = {
    C.PAGE_TODAY: "pages/today.py",
    C.PAGE_PORTFOLIO: "pages/portfolio.py",
    C.PAGE_EXPLORE: "pages/explore.py",
    C.PAGE_SETTINGS: "pages/settings.py",
}


def _run_page(page: str) -> AppTest:
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page(_PAGE_FILE[page]).run()
    return at


def _metrics_run(tmp_path, payload: dict) -> str:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    return str(run_dir)


def test_review_regime_is_displayed(tmp_path, monkeypatch):
    history.record_review(value_eur=1000.0, invested_eur=900.0,
                          cash_eur=100.0, pnl_eur=100.0)
    run_dir = _metrics_run(tmp_path, {"regime": {"state": "estimated", "label": "rising",
                                               "confidence": "high", "as_of": "2026-09-13"},
                                       "review_ts": "2026-09-13"})
    monkeypatch.setattr(artifacts, "latest_run_dir", lambda: run_dir)

    text = _all_text(_run_page(C.PAGE_TODAY))
    assert C.MARKET_TREND.format(label="rising", confidence="high") in text


def test_regime_failure_is_health_not_missing_history(tmp_path, monkeypatch):
    history.record_review(value_eur=1000.0, invested_eur=900.0,
                          cash_eur=100.0, pnl_eur=100.0)
    run_dir = _metrics_run(tmp_path, {"regime": {"state": "failed", "error": "boom",
                                              "as_of": "2026-09-13"},
                                       "review_ts": "2026-09-13"})
    monkeypatch.setattr(artifacts, "latest_run_dir", lambda: run_dir)

    today_text = _all_text(_run_page(C.PAGE_TODAY))
    assert C.MARKET_TREND_FAILED in today_text
    # The masking hole is closed: a failed computation is not missing history.
    assert "not enough history" not in today_text

    settings_text = _all_text(_run_page(C.PAGE_SETTINGS))
    assert C.HEALTH_REGIME_FAILED in settings_text
