"""
test_v10_7_2_phase_d.py — First-week experience: quant setup and the checklist.

Covers (v10.7.2, Part 4):
  - --check prints all-not-done for a fresh environment;
  - --check prints all-done for a configured fake environment;
  - the step functions are pure status-plus-action units (no input());
  - the checklist text matches docs/first_week.md.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from quant import paths
from quant.engine import backup, setup
from quant.reporting import artifacts

DOCS = Path(__file__).resolve().parents[1] / "docs" / "first_week.md"


def _fresh(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(paths, "DATA_TIERS", str(tmp_path / "data" / "tiers.csv"))
    monkeypatch.setattr(artifacts, "read_update_state", lambda: {})
    from quant.engine import notify, scheduler

    monkeypatch.setattr(scheduler, "is_installed", lambda: False)
    monkeypatch.setattr(notify, "load_config", lambda path=None: {"channel": "none"})
    monkeypatch.setattr(backup, "last_backup_at", lambda: None)


def _configured(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path / "outputs")
    tiers = tmp_path / "data" / "tiers.csv"
    tiers.parent.mkdir(parents=True, exist_ok=True)
    tiers.write_text("symbol,tier\nAMZN,ALPHA\nEUNL.DE,FORTRESS\n", encoding="utf-8")
    monkeypatch.setattr(paths, "DATA_TIERS", str(tiers))
    monkeypatch.setattr(artifacts, "read_update_state",
                        lambda: {"ts": "2026-10-01T18:45:00"})
    from quant.engine import notify, scheduler

    monkeypatch.setattr(scheduler, "is_installed", lambda: True)
    monkeypatch.setattr(notify, "load_config", lambda path=None: {"channel": "telegram"})
    monkeypatch.setattr(backup, "last_backup_at", lambda: datetime.now())


def test_check_fresh_all_not_done(tmp_path, monkeypatch):
    _fresh(tmp_path, monkeypatch)
    lines = setup.check_lines()
    assert lines[0] == "Setup status"
    # Steps 1-5 are not done; step 6 (checklist) is always available.
    for line in lines[1:6]:
        assert "not done" in line
    assert "not done" not in lines[6]


def test_check_configured_all_done(tmp_path, monkeypatch):
    _configured(tmp_path, monkeypatch)
    lines = setup.check_lines()
    for line in lines[1:6]:
        assert "done" in line
        assert "not done" not in line


def test_steps_are_pure_units(tmp_path, monkeypatch):
    _fresh(tmp_path, monkeypatch)
    steps = setup.check_steps()
    assert len(steps) == 6
    market = next(s for s in steps if s.key == "market")
    assert market.done is False
    assert callable(market.action)  # a status-plus-action unit, no input()


def test_run_interactive_skips_without_input(tmp_path, monkeypatch):
    _fresh(tmp_path, monkeypatch)
    printed: list[str] = []
    rc = setup.run_interactive(input_fn=lambda _p: "skip", print_fn=printed.append)
    assert rc == 0
    text = "\n".join(printed)
    assert "Setup status" not in text  # interactive prints step lines, not the header
    assert "First-week checklist" in text
    assert "1. Set up the Telegram bot (quant notify-setup)." in text


def test_checklist_matches_docs():
    doc = DOCS.read_text(encoding="utf-8")
    for item in setup.FIRST_WEEK_CHECKLIST:
        assert item in doc, item
