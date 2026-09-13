"""
test_cli_output.py — Terse CLI contract (v10.5.0, spec 7 / T1g).

Asserts: default stdout is at most 20 lines, contains the required lines, and
contains no INFO] leakage; --verbose contains the log details.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

from quant.cli import build_parser, output
from quant.reporting.actions import format_action_line


def test_parser_has_verbose_and_publish():
    parser = build_parser()
    args = parser.parse_args(["--verbose", "run"])
    assert args.verbose is True
    args2 = parser.parse_args(["publish"])
    assert args2.command == "publish"


def test_detail_goes_to_log_not_stdout(tmp_path):
    log = tmp_path / "pipeline.log"
    rep = output.configure(verbose=False, log_path=str(log))
    buf = io.StringIO()
    with redirect_stdout(buf):
        rep.line("quant run 10.5.0")
        rep.detail("  [OK] AAPL: fetched")
    out = buf.getvalue()
    assert "quant run 10.5.0" in out
    assert "AAPL" not in out
    assert "AAPL" in log.read_text(encoding="utf-8")
    rep.close()


def test_verbose_echoes_detail(tmp_path):
    log = tmp_path / "pipeline.log"
    rep = output.configure(verbose=True, log_path=str(log))
    buf = io.StringIO()
    with redirect_stdout(buf):
        rep.detail("  [OK] AAPL: fetched")
    assert "AAPL" in buf.getvalue()
    rep.close()


def test_terse_run_output_under_20_lines(tmp_path):
    log = tmp_path / "pipeline.log"
    rep = output.configure(verbose=False, log_path=str(log))
    buf = io.StringIO()
    with redirect_stdout(buf):
        rep.line("quant run 10.5.0")
        rep.line("  regime bull, p=1.00 (fit IWDA.AS, as-of 2026-09-13)")
        rep.line("  scanned 53 symbols; 2 actions, 1 blocked")
        rep.line("  actions")
        rep.line(format_action_line({
            "symbol": "EUNL.DE", "action": "BUY MORE", "amount_eur": 150.0,
            "reason": "drift -16.1% vs CORE target 50.0% (threshold 10.0%)",
            "blocked": False,
        }))
        rep.line(format_action_line({
            "symbol": "SWRD.L", "action": "BLOCKED", "amount_eur": None,
            "reason": "ISIN missing; add to data/broker_registry.csv", "blocked": True,
        }))
        rep.line("  portfolio 830.40 EUR; PnL -2.61 EUR (-0.31%); cash 83.04 EUR; risk balanced")
        rep.line("  evidence: 9 symbols with news, 44 without (sentiment neutral, confidence low)")
        rep.line("  report outputs/run_x/briefing.md")
    out = buf.getvalue()
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) <= 20
    assert "INFO]" not in out
    rep.close()


def test_cli_returns_2_on_error(monkeypatch, tmp_path):
    from quant import cli

    monkeypatch.setattr(cli.output, "default_log_path", lambda: str(tmp_path / "p.log"))

    def _boom(_args):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "_cmd_run", _boom)
    rc = cli.main(["run"])
    assert rc == 2
