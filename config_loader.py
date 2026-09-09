"""
config_loader.py — YAML Configuration Loader (Part 3, Simplification #2).

Intent: move tunable parameters out of Python code into a version-controllable
YAML file so non-programmers can adjust thresholds without editing code.
Supports nested dot-path access: config.get('scoring.factor_weights.momentum').

Invariants:
  - get(path, default) returns the value at a dot path or the default.
  - Attribute access returns top-level sections.
  - Pure I/O at construction; read-only afterwards.

Dependencies: yaml, pathlib.
"""
from __future__ import annotations

from pathlib import Path

import yaml


class Config:
    """Loads and provides access to a YAML configuration file."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path) as f:
            self._data = yaml.safe_load(f) or {}

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._data.get(name)

    def get(self, path: str, default=None):
        """Get a nested config value: config.get('a.b.c', default)."""
        keys = path.split(".")
        value = self._data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def as_dict(self) -> dict:
        """Return the full config as a dict."""
        return self._data