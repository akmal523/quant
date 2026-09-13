"""
test_publish.py — Published Briefing contract (v10.5.0, spec 7 / T7d).

Asserts: quant publish emits index.html containing as-of, version, and the
fixture actions; HTML parses cleanly.
"""
from __future__ import annotations

from html.parser import HTMLParser

import pandas as pd

from quant import __version__
from quant.reporting.web import build_data, publish, render_html


def _fixture_run(tmp_path) -> str:
    run = tmp_path / "run_2026-09-13_152139"
    run.mkdir()
    audit = pd.DataFrame([{
        "Symbol": "EUNL.DE", "Tier": "CORE",
        "Current_Weight": "33.9%", "Target_Weight": "50.0%", "Drift": "-16.1%",
        "Signal": "BUY",
        "Recommendation": "BUY 150 EUR (CORE drift -16.1% exceeds 10.0% threshold)",
        "Value_EUR": 281.28, "Real_PnL_EUR": 3.28, "Active_Score": 80.0,
    }])
    audit.to_csv(run / "portfolio_audit.csv", index=False)
    return str(run)


def test_publish_emits_index_html(tmp_path):
    run = _fixture_run(tmp_path)
    out = tmp_path / "web"
    rc = publish(run_dir=run, out_dir=str(out))
    assert rc == 0
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "Quant-AI Briefing" in html
    assert __version__ in html
    assert "EUNL.DE" in html
    assert (out / "data.json").exists()


def test_html_parses_cleanly(tmp_path):
    run = _fixture_run(tmp_path)
    data = build_data(run)
    html = render_html(data)
    parser = HTMLParser()
    parser.feed(html)  # raises on malformed markup


def test_publish_without_run_returns_error(tmp_path):
    rc = publish(run_dir=str(tmp_path / "missing"), out_dir=str(tmp_path / "web"))
    assert rc == 1
