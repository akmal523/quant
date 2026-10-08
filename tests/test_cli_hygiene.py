"""
test_cli_hygiene.py — CLI hygiene + publish truthfulness (H3.2).
"""
from __future__ import annotations


def test_progress_silent_when_piped(capsys):
    """Zero progress chunks when stdout is not a TTY (pipes/CI)."""
    from quant.cli.output import Reporter

    r = Reporter(verbose=False, log_path=None)
    r.progress("  fetched 1/74")
    r.end_progress()
    assert "fetched" not in capsys.readouterr().out


