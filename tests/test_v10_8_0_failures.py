"""test_v10_8_0_failures.py — a failed computation is a visible failure (v10.8.0, 3.2).

An exception in a computation that feeds a user-visible conclusion must show a
plain failure state, never an empty success state.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

import quant.ui.render as render  # noqa: E402
from quant.ui import copy as ui_copy  # noqa: E402

DASHBOARD = str(Path(__file__).resolve().parents[1] / "quant" / "dashboard.py")


def _all_text(at: AppTest) -> str:
    chunks = []
    for attr in ("markdown", "info", "warning", "error", "caption", "text",
                 "title", "header", "subheader"):
        for el in getattr(at, attr, []):
            chunks.append(str(getattr(el, "value", "")))
    return " ".join(chunks)


def test_advice_failure_shows_failure_state(monkeypatch):
    # Inject the failure at the render layer (an allowed boundary), not in the
    # decision core: _monthly_holdings raising makes build_advice's argument
    # evaluation fail, which the Overview must surface as a failure state.
    def _boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(render, "read_actions", lambda: [])
    monkeypatch.setattr(render, "_monthly_holdings", _boom)

    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    at.switch_page("pages/today.py").run()
    text = _all_text(at)
    assert "Could not check what to do" in text
    assert ui_copy.NOTHING_TO_DO_WEEK not in text
