"""
test_doctor.py — Doctor output shape (H2 commit 1).

Uses a RECORDED metadata probe fixture (D1): the doctor renders the raw longName
(or the recorded error) without any network call.
"""
from __future__ import annotations

import json
from pathlib import Path

from quant import paths
from quant.data import names


def test_doctor_output_shape(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    recorded = json.loads(
        (Path(__file__).parent / "fixtures" / "metadata_probe.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(names, "probe_metadata", lambda s: recorded[s])

    from quant.cli import _cmd_doctor

    rc = _cmd_doctor(None)
    out = capsys.readouterr().out
    assert rc == 0
    assert "db:" in out
    assert "registry rows:" in out
    assert "missing display_name:" in out
    assert "probe AMZN:" in out
    assert "Amazon.com, Inc." in out
    assert "names state:" in out
    assert "runner lock:" in out
