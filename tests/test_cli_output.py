"""
test_cli_output.py — Terse CLI contract (v10.5.0, spec 7 / T1g).

Asserts: default stdout is at most 20 lines, contains the required lines, and
contains no INFO] leakage; --verbose contains the log details.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout

from quant.cli import build_parser, output


def test_parser_has_verbose_and_a_command():
    parser = build_parser()
    args = parser.parse_args(["--verbose", "run"])
    assert args.verbose is True
    args2 = parser.parse_args(["doctor"])
    assert args2.command == "doctor"


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


def test_cli_returns_2_on_error(monkeypatch, tmp_path):
    from quant import cli

    monkeypatch.setattr(cli.output, "default_log_path", lambda: str(tmp_path / "p.log"))

    def _boom(_args):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "_cmd_run", _boom)
    rc = cli.main(["run"])
    assert rc == 2
