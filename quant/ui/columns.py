"""
columns.py — the ONE column-spec contract for the broker editor + entry check
(v10.8.8, R-P1).

Intent: the broker editor grew column-by-column across versions until it
overflowed the container and kept an unlabeled index/action column, and live
values were added as MORE grid columns instead of a separate read-only view.
This module is the single source of truth for the editor grid and the read-only
Entry-check table:

  - the editor grid contains ONLY typed inputs plus the two read-only identity
    columns (``Name`` = clean human display name, ``Instrument`` = the exact
    broker ticker); the index is hidden;
  - derived and live data live OUTSIDE the grid, in the read-only Entry-check
    table (``Instrument | Shares (derived) | Typed value (EUR) | Live value
    (EUR) | Delta %``);
  - every column has a non-empty header and, where applicable, a currency in the
    header;
  - no table exceeds the default Streamlit container width (a column budget).

Addendum A1 (v10.8.8): the ``Name`` cell renders the display name WITHOUT any
ticker in parentheses; the parenthesized form is allowed only in
single-identity-column tables (e.g. the home Holdings table).

Addendum A2 (v10.8.8): the Entry-check identity column is ``Instrument`` (not
``Symbol``) and carries the same broker ticker as the grid, so the two tables
join visually on one key.

Invariants:
  - Pure module: no streamlit import at module load (``editor_column_config``
    imports it lazily), so the contract is importable and testable without a UI.
  - ``EDITOR_COLUMNS`` and ``CHECK_COLUMNS`` are the ONLY column lists; the
    renderer derives its config, order and disabled set from them.

Dependencies: typing only (streamlit imported lazily).
"""
from __future__ import annotations

from typing import NamedTuple

from quant.ui import copy as C


class ColumnSpec(NamedTuple):
    """One column of a table contract.

    ``name`` is the internal dataframe key; ``header`` is the non-empty display
    header; ``currency`` is the unit shown in the header ("" when not money);
    ``editable`` is True for a typed input; ``width_priority`` is the Streamlit
    width ("small" / "medium" / "large"). ``numeric`` and ``fmt`` are rendering
    details derived from the spec (kept out of the contract's five named fields).
    """

    name: str
    header: str
    currency: str
    editable: bool
    width_priority: str
    numeric: bool = False
    fmt: str = "%.2f"


# ── The editor grid (R-P1): identity + typed inputs only ──────────────────────
# Exactly six columns, in order. No remove/action column (owner decision). The
# header strings live in quant.ui.copy (the ONE string module).
EDITOR_COLUMNS: list[ColumnSpec] = [
    ColumnSpec("Name", C.COLUMN_HEADERS["Name"], "", False, "medium",
               numeric=False),
    ColumnSpec("Symbol", C.COLUMN_HEADERS["Symbol"], "", False, "small",
               numeric=False),
    ColumnSpec("Avg_Entry_Price", C.COLUMN_HEADERS["Avg_Entry_Price"], "paid ccy",
               True, "small", numeric=True, fmt="%.2f"),
    ColumnSpec("Current_Value_EUR", C.COLUMN_HEADERS["Current_Value_EUR"], "EUR",
               True, "small", numeric=True, fmt="%.2f"),
    ColumnSpec("Broker_PnL_EUR", C.COLUMN_HEADERS["Broker_PnL_EUR"], "EUR",
               True, "small", numeric=True, fmt="%.2f"),
    ColumnSpec("Plan_EUR_month", C.COLUMN_HEADERS["Plan_EUR_month"], "EUR/m",
               True, "small", numeric=True, fmt="%.0f"),
]

# The editor budget: the six contract columns (no optional Remove column).
EDITOR_MAX_COLUMNS = 7

# ── The read-only Entry-check table (R-P1): the ONLY place live and typed meet ─
# A2: the identity column is "Instrument" (internal key Symbol), the same broker
# ticker as the grid.
CHECK_COLUMNS: list[ColumnSpec] = [
    ColumnSpec("Symbol", C.ENTRY_CHECK_HEADERS[0], "", False, "small",
               numeric=False),
    ColumnSpec("Shares", C.ENTRY_CHECK_HEADERS[1], "", False, "small",
               numeric=True, fmt="%.4f"),
    ColumnSpec("Typed", C.ENTRY_CHECK_HEADERS[2], "EUR", False, "small",
               numeric=True, fmt="%.2f"),
    ColumnSpec("Live", C.ENTRY_CHECK_HEADERS[3], "EUR", False, "small",
               numeric=True, fmt="%.2f"),
    ColumnSpec("Delta", C.ENTRY_CHECK_HEADERS[4], "%", False, "small",
               numeric=True, fmt="%.1f"),
]

CHECK_MAX_COLUMNS = 5


def editor_headers() -> list[str]:
    """The editor grid's display headers, in order (the contract)."""
    return [c.header for c in EDITOR_COLUMNS]


def check_headers() -> list[str]:
    """The Entry-check table's display headers, in order (the contract)."""
    return [c.header for c in CHECK_COLUMNS]


def editor_column_order() -> list[str]:
    """The editor grid's internal keys, in order."""
    return [c.name for c in EDITOR_COLUMNS]


def check_column_order() -> list[str]:
    """The Entry-check table's internal keys, in order."""
    return [c.name for c in CHECK_COLUMNS]


def editor_disabled() -> list[str]:
    """The editor grid's read-only keys (identity columns)."""
    return [c.name for c in EDITOR_COLUMNS if not c.editable]


def _column_config(specs: list[ColumnSpec]) -> dict:
    """Build a Streamlit ``column_config`` dict from a spec list (lazy import)."""
    import streamlit as st

    out: dict = {}
    for c in specs:
        if c.numeric:
            out[c.name] = st.column_config.NumberColumn(
                c.header, format=c.fmt, width=c.width_priority)
        else:
            out[c.name] = st.column_config.TextColumn(
                c.header, width=c.width_priority)
    return out


def editor_column_config() -> dict:
    """The editor grid's Streamlit column config, derived from the contract."""
    return _column_config(EDITOR_COLUMNS)


def check_column_config() -> dict:
    """The Entry-check table's Streamlit column config, derived from the contract."""
    return _column_config(CHECK_COLUMNS)
