"""
test_cli_hygiene.py — CLI hygiene + publish truthfulness (H3.2).
"""
from __future__ import annotations

import os

import pandas as pd


def test_progress_silent_when_piped(capsys):
    """Zero progress chunks when stdout is not a TTY (pipes/CI)."""
    from quant.cli.output import Reporter

    r = Reporter(verbose=False, log_path=None)
    r.progress("  fetched 1/74")
    r.end_progress()
    assert "fetched" not in capsys.readouterr().out


def test_publish_prints_existing_path(tmp_path, capsys):
    from quant.reporting import web

    run = tmp_path / "run_a"
    run.mkdir()
    (run / "metrics.json").write_text("{}", encoding="utf-8")
    pd.DataFrame([{
        "Symbol": "X", "Tier": "ACTIVE", "Value_EUR": 1.0, "Drift": "0.0%",
        "Current_Weight": "1.0%", "Target_Weight": "1.0%", "Signal": "HOLD",
    }]).to_csv(run / "portfolio_audit.csv", index=False)

    rc = web.publish(run_dir=str(run), out_dir=str(tmp_path / "webout"))
    out = capsys.readouterr().out
    assert rc == 0
    assert out.splitlines()[0].startswith("quant publish ")
    printed = [ln for ln in out.splitlines() if ln.startswith("published ")][0]
    path = printed.split("published ", 1)[1]
    assert os.path.exists(path)
