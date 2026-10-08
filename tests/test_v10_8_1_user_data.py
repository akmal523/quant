"""
test_v10_8_1_user_data.py — user state never resets (v10.8.1, Part A).

Guards the A1 root causes: user state must resolve OUTSIDE the repository, the
one-time migration must copy (never move/delete) legacy repo state, and it must
be idempotent and never overwrite existing user data.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from quant import paths
from quant.data import user_data


def test_user_state_never_under_repo_root():
    """A2: no user-state path may resolve under the repository root."""
    code = Path(paths.CODE_ROOT).resolve()
    for value in (paths.PROJECT_ROOT, Path(paths.DATA_PORTFOLIO),
                  Path(paths.DATA_ACCOUNT), Path(paths.DB_FILE)):
        assert code not in Path(value).resolve().parents, f"{value} is under the repo"


def test_project_root_is_the_quant_data_dir_override():
    # conftest sets QUANT_DATA_DIR to a throwaway dir for the whole session.
    import os

    assert str(paths.PROJECT_ROOT) == os.environ["QUANT_DATA_DIR"]
    assert paths.PROJECT_ROOT == Path(paths.DATA_DIR).parent


def test_migration_is_idempotent_and_copies_only(tmp_path, monkeypatch):
    """Migration copies legacy files, never deletes/overwrites, and runs once."""
    legacy = tmp_path / "repo" / "data"
    legacy.mkdir(parents=True)
    (legacy / "portfolio.csv").write_text(
        "Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\nAMZN,1,2,0\n",
        encoding="utf-8")
    data = tmp_path / "user" / "data"
    monkeypatch.setattr(paths, "LEGACY_DATA_DIR", legacy)
    monkeypatch.setattr(paths, "DATA_DIR", data)
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path / "user")
    monkeypatch.setattr(paths, "CODE_ROOT", tmp_path / "repo")
    monkeypatch.setattr(paths, "BUNDLED_DATA_DIR", tmp_path / "bundled")

    copied = user_data.migrate_legacy_user_data()
    assert "portfolio.csv" in copied
    # Copy only: the legacy original is untouched.
    assert (legacy / "portfolio.csv").exists()
    # Second run is a no-op (marker written).
    assert user_data.migrate_legacy_user_data() == []


def test_migration_never_overwrites_existing(tmp_path, monkeypatch):
    legacy = tmp_path / "repo" / "data"
    legacy.mkdir(parents=True)
    (legacy / "portfolio.csv").write_text("OLD\n", encoding="utf-8")
    data = tmp_path / "user" / "data"
    data.mkdir(parents=True)
    mine = data / "portfolio.csv"
    mine.write_text("MINE\n", encoding="utf-8")
    monkeypatch.setattr(paths, "LEGACY_DATA_DIR", legacy)
    monkeypatch.setattr(paths, "DATA_DIR", data)
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path / "user")
    monkeypatch.setattr(paths, "CODE_ROOT", tmp_path / "repo")
    monkeypatch.setattr(paths, "BUNDLED_DATA_DIR", tmp_path / "bundled")

    user_data.migrate_legacy_user_data()
    assert mine.read_text(encoding="utf-8") == "MINE\n"


def test_simulated_update_keeps_user_data(tmp_path, monkeypatch):
    """A4#1: a pull that untracks/deletes the repo file leaves user data intact."""
    repo = tmp_path / "repo"
    (repo / "data").mkdir(parents=True)
    pf = repo / "data" / "portfolio.csv"
    pf.write_text("Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\n"
                  "AMZN,150.0,160.0,10.0\n", encoding="utf-8")
    # a real git checkout: track the file, then untrack it in a later commit.
    try:
        for cmd in (["git", "init", "-q"], ["git", "config", "user.email", "t@t"],
                    ["git", "config", "user.name", "t"],
                    ["git", "add", "data/portfolio.csv"], ["git", "commit", "-qm", "track"],
                    ["git", "rm", "-q", "--cached", "data/portfolio.csv"],
                    ["git", "commit", "-qm", "untrack"]):
            subprocess.run(cmd, cwd=repo, check=True, capture_output=True)
    except Exception:  # noqa: BLE001
        import pytest

        pytest.skip("git unavailable")

    user = tmp_path / "user"
    udata = user / "data"
    udata.mkdir(parents=True)
    monkeypatch.setattr(paths, "PROJECT_ROOT", user)
    monkeypatch.setattr(paths, "DATA_DIR", udata)
    monkeypatch.setattr(paths, "LEGACY_DATA_DIR", repo / "data")
    monkeypatch.setattr(paths, "CODE_ROOT", repo)
    monkeypatch.setattr(paths, "BUNDLED_DATA_DIR", tmp_path / "bundled")

    # The user data is copied to the user dir before the pull.
    user_data.migrate_legacy_user_data()
    # Simulate the pull deleting the (now untracked) repo file.
    pf.unlink()
    kept = udata / "portfolio.csv"
    assert kept.exists()
    assert "AMZN" in kept.read_text(encoding="utf-8")
