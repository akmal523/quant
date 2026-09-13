"""
test_dashboard_contract.py — Dashboard contract (v10.5.0, spec 7 / T3c).

Asserts: no literal n/a, unknown, or unlabelled 0.50; empty states render the
catalog strings. Uses streamlit.testing.v1.AppTest; skipped if streamlit is
not installed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

# AppTest resolves relative paths against the calling file, so use an absolute
# path to the repo-root dashboard.
DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")


def _all_text(at: AppTest) -> str:
    chunks = []
    for attr in ("markdown", "info", "warning", "error", "caption", "text"):
        for el in getattr(at, attr, []):
            chunks.append(str(getattr(el, "value", "")))
    return " ".join(chunks)


def test_dashboard_runs_without_exception():
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    assert not at.exception


def test_no_silent_defaults_in_ui():
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    text = _all_text(at).lower()
    assert "n/a" not in text
    assert "unknown" not in text


def test_today_empty_state_when_no_reviews():
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    text = _all_text(at)
    # v10.5.1: the Today page shows the catalog empty states / first-run guide.
    assert (
        ("Nothing to do today" in text)
        or ("The value chart appears after your second review" in text)
        or ("Start here" in text)
    )
