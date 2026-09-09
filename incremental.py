"""
incremental.py — Incremental Processing / Change Detection (Part 3, Gap #4).

Intent: detect which symbols need reprocessing based on data changes, so the
pipeline is O(changed) instead of O(all). On a typical day only 2-3 symbols
change; processing only those yields a ~10x speedup.

Invariants:
  - detect_changes returns True iff the data hash changed since last run.
  - State is persisted to a JSON file.
  - Pure logic; file I/O only for state.

Dependencies: hashlib, json, pathlib, pandas.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


class IncrementalProcessor:
    """Detects which symbols need reprocessing based on data changes."""

    def __init__(self, state_file: str = ".processing_state.json"):
        self.state_file = Path(state_file)
        self.state = self._load_state()

    def _load_state(self) -> dict:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_state(self) -> None:
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def detect_changes(self, df: pd.DataFrame, symbol: str) -> bool:
        """Return True if data changed since last processing."""
        if df is None or df.empty:
            return False
        # Hash only numeric columns (datetime objects are not byte-stable).
        numeric = df.select_dtypes(include="number").tail(10)
        if numeric.empty:
            return False
        recent_hash = hashlib.md5(numeric.to_numpy().tobytes()).hexdigest()

        last_hash = self.state.get(symbol, {}).get("data_hash")
        if last_hash == recent_hash:
            return False

        self.state[symbol] = {
            "data_hash": recent_hash,
            "last_processed": pd.Timestamp.now().isoformat(),
            "last_row_date": str(df["Date"].iloc[-1]) if "Date" in df.columns else "",
        }
        self._save_state()
        return True