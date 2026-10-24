"""test_v10_8_0_install.py — install and dependency-policy guards (v10.8.0, 1.3).

A fresh install must import the review's text fetcher, the declared dependency
ranges must match the code, and the optional text-fetch step must degrade
visibly instead of crashing the review.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_minimum_is_real():
    """The declared minimum must cover st.segmented_control and width='stretch'."""
    text = (ROOT / "pyproject.toml").read_text()
    m = re.search(r'"streamlit>=([0-9.]+)"', text)
    assert m, "streamlit minimum not declared"
    major, minor = (int(x) for x in m.group(1).split(".")[:2])
    assert (major, minor) >= (1, 49), f"streamlit>={m.group(1)} is too low"
