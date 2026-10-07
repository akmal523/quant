"""test_v10_8_0_install.py — install and dependency-policy guards (v10.8.0, 1.3).

A fresh install must import the review's text fetcher, the declared dependency
ranges must match the code, and the optional text-fetch step must degrade
visibly instead of crashing the review.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_async_fetcher_imports():
    """A fresh install must be able to import the review's text fetcher."""
    import quant.data.async_fetcher  # noqa: F401


def test_sec_edgar_uses_lexbor():
    from selectolax.lexbor import LexborHTMLParser

    import quant.data.sec_edgar as sec_edgar

    assert sec_edgar.LexborHTMLParser is LexborHTMLParser


def test_selectolax_is_bounded():
    text = (ROOT / "pyproject.toml").read_text()
    assert re.search(r'"selectolax>=0\.3\.16,<1\.0"', text)


def test_streamlit_minimum_is_real():
    """The declared minimum must cover st.segmented_control and width='stretch'."""
    text = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'"streamlit>=([0-9.]+)"', text)
    assert m, "streamlit minimum not declared"
    major, minor = (int(x) for x in m.group(1).split(".")[:2])
    assert (major, minor) >= (1, 49), f"streamlit>={m.group(1)} is too low"


def test_review_text_fetch_is_guarded():
    """The optional text-fetch import must not crash the review (1.3)."""
    src = (ROOT / "quant" / "main.py").read_text()
    assert "Text fetch unavailable" in src
    assert "from quant.data.async_fetcher import fetch_all_texts_concurrently" in src


def test_news_outage_message_after_three_failures(tmp_path, monkeypatch):
    from quant import paths
    from quant.data import news

    monkeypatch.setattr(paths, "OUTPUTS_DIR", tmp_path)
    for _ in range(3):
        news.record_fetch_result(False)
    assert news.outage_message() is not None
