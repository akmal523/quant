"""test_v10_8_0_ui_robustness.py — UI data-frame robustness (v10.8.0).

The holdings table mixed floats and "" in the Structure/Tactics columns, which
made Streamlit's Arrow conversion fail ("Could not convert '' ... to double").
These tests pin the fix: every score cell is a string, so the column is
homogeneous and Arrow-compatible.
"""
from __future__ import annotations

import pandas as pd


def test_score_cell_is_always_a_string():
    from quant.ui.render import _score_cell

    assert _score_cell(85.0) == "85"
    assert _score_cell(85.4) == "85"
    assert _score_cell(0.0) == "0"
    assert _score_cell(None) == ""
    assert _score_cell("") == ""
    assert _score_cell("n/a") == "n/a"


def test_score_column_is_arrow_compatible():
    """A column built from _score_cell has one dtype and converts to Arrow."""
    import pyarrow as pa

    from quant.ui.render import _score_cell

    values = [85.0, None, 0.0, "", 72.3]
    col = pd.Series([_score_cell(v) for v in values])
    assert all(isinstance(v, str) for v in col)
    # The conversion that used to raise now succeeds.
    table = pa.Table.from_pandas(pd.DataFrame({"Structure": col}))
    assert table.num_rows == len(values)
