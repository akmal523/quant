"""
test_v10_7_2_phase_b.py — News pillar: diagnose, then demote honestly.

Covers (v10.7.2, Part 2):
  - diagnostic counts on a synthetic cache;
  - absent status prevents torch import in the scorer path;
  - the exact absent line renders on Overview and the briefing;
  - --enable flips the status;
  - Friday recomputation logic with the 30-day window.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from quant import paths
from quant.engine import news_pillar
from quant.ui import copy as C

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")
_TEXTLIKE = ("markdown", "info", "warning", "error", "caption", "text",
             "title", "header", "subheader", "success")


def _write_cache(tmp_path, cache: dict) -> None:
    (tmp_path / "news_cache.json").write_text(json.dumps(cache), encoding="utf-8")


# ── Diagnostic ────────────────────────────────────────────────────────────────

def test_diagnostic_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    now = datetime.now()
    recent = now.isoformat()
    _write_cache(tmp_path, {
        "AMZN": {"retrieved_at": now.timestamp(), "items": [
            {"headline": "A" * 60, "scorer": "model", "score": 5.0,
             "published_at": recent},
            {"headline": "short", "scorer": "default", "score": 0.0,
             "published_at": recent},
        ]},
        "EUNL.DE": {"retrieved_at": now.timestamp(), "items": [
            {"headline": "B" * 60, "scorer": "default", "score": 0.0,
             "published_at": recent},
        ]},
    })
    rows = {r["symbol"]: r for r in news_pillar.diagnostic(["AMZN", "EUNL.DE", "MISSING"])}
    assert rows["AMZN"]["total"] == 2
    assert rows["AMZN"]["long_enough"] == 1
    assert rows["AMZN"]["model"] == 1
    assert rows["AMZN"]["default"] == 1
    assert rows["AMZN"]["dominant_reason"] == C.NEWS_DOCTOR_REASON_SHORT
    assert rows["EUNL.DE"]["default"] == 1
    assert rows["MISSING"]["total"] == 0


def test_diagnostic_never_imports_torch(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    _write_cache(tmp_path, {})
    import sys

    before = set(sys.modules)
    news_pillar.diagnostic(["AMZN"])
    loaded = set(sys.modules) - before
    # The diagnostic may PROBE availability (find_spec) but must never LOAD torch.
    assert "torch" not in loaded
    assert "transformers" not in loaded


# ── Absent status + lazy-import guard ─────────────────────────────────────────

def test_absent_prevents_torch_import(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    news_pillar.write_status(news_pillar.STATUS_ABSENT)

    import importlib.util

    calls: list[str] = []
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a, **k: (calls.append(name), real(name, *a, **k))[1])

    from quant.data import news

    news._default_scorer_resolved = False
    news._default_scorer_value = None
    assert news._build_default_scorer() is None
    assert "torch" not in calls
    assert "transformers" not in calls


def test_enable_flips_status(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    news_pillar.write_status(news_pillar.STATUS_ABSENT)
    assert news_pillar.is_absent() is True
    news_pillar.enable()
    assert news_pillar.is_absent() is False


def test_news_doctor_enable_cli(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    news_pillar.write_status(news_pillar.STATUS_ABSENT)
    from quant.cli import _cmd_news_doctor

    rc = _cmd_news_doctor(argparse.Namespace(enable=True))
    assert rc == 0
    assert news_pillar.is_absent() is False
    assert C.NEWS_DOCTOR_ENABLED in capsys.readouterr().out


def test_news_doctor_cli_prints_header(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    from quant.cli import _cmd_news_doctor

    rc = _cmd_news_doctor(argparse.Namespace(enable=False))
    out = capsys.readouterr().out
    assert rc == 0
    assert C.NEWS_DOCTOR_HEADER in out


# ── Friday recomputation + 30-day window ──────────────────────────────────────

def test_friday_recompute_sets_absent_then_active(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    news_pillar.recompute_status(date(2026, 10, 2))  # a Friday
    assert news_pillar.read_status()["status"] == news_pillar.STATUS_ABSENT

    now = datetime.now()
    _write_cache(tmp_path, {"AMZN": {"retrieved_at": now.timestamp(), "items": [
        {"headline": "x" * 60, "scorer": "model", "score": 1.0,
         "published_at": now.isoformat()}]}})
    news_pillar.recompute_status(date(2026, 10, 2))
    assert news_pillar.read_status()["status"] == news_pillar.STATUS_ACTIVE


def test_recompute_30_day_window(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    old = (datetime.now() - timedelta(days=40)).isoformat()
    _write_cache(tmp_path, {"AMZN": {"retrieved_at": 0, "items": [
        {"headline": "x" * 60, "scorer": "model", "score": 1.0, "published_at": old}]}})
    news_pillar.recompute_status(date(2026, 10, 2))
    assert news_pillar.read_status()["status"] == news_pillar.STATUS_ABSENT


# ── The exact absent line on Overview and the briefing ────────────────────────

def test_absent_line_on_overview(tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import quant.ui.render as render

    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    news_pillar.write_status(news_pillar.STATUS_ABSENT)
    monkeypatch.setattr(render, "latest_review", lambda ok_only=False: {})
    monkeypatch.setattr(render, "read_history", lambda: pd.DataFrame())
    monkeypatch.setattr(render, "read_regime", lambda: {"state": "insufficient_history"})
    monkeypatch.setattr(render, "read_actions", lambda: [])
    monkeypatch.setattr(render, "load_portfolio",
                        lambda: pd.DataFrame([{"Symbol": "AMZN", "Amount_EUR": 10.0}]))

    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page("pages/today.py").run()
    text = " ".join(str(getattr(el, "value", ""))
                    for attr in _TEXTLIKE for el in getattr(at, attr, []))
    assert C.NEWS_PILLAR_ABSENT in text


def test_absent_line_in_briefing(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    news_pillar.write_status(news_pillar.STATUS_ABSENT)
    from quant.portfolio.account import AccountState
    from quant.reporting.briefing import build_briefing_md

    account = AccountState("EUR", 0.0, "balanced", False, None)
    md = build_briefing_md(
        as_of="2026-10-02", version="10.7.2", regime_label="mixed",
        regime_prob=0.5, regime_source="hmm", audit_df=pd.DataFrame(),
        account=account, total_value=0.0, pnl_eur=0.0, pnl_pct=0.0,
        with_news=0, without_news=0, latest_bar="2026-10-01",
    )
    assert C.NEWS_PILLAR_ABSENT in md
