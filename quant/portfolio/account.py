"""
account.py — Account-level inputs from data/account.yaml (v10.5.0, spec 2.3).

Intent: single source of truth for cash, risk profile, and base currency.
Replaces the account_state DuckDB table. A missing file yields an explicit empty
state (cash_eur=None), never a silent 0.00 (R1).

Invariants:
  - load_account() never raises; returns an AccountState.
  - cash_eur is None when the file is missing or the key is absent.
  - risk_profile is one of RISK_PROFILES keys, else DEFAULT_RISK_PROFILE.
  - risk_limits() returns the documented tuple for the profile.

Dependencies: yaml, quant.paths, quant.config.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import yaml

from quant import paths
from quant.config import RISK_PROFILES, DEFAULT_RISK_PROFILE, BASE_CURRENCY


@dataclass(frozen=True)
class AccountState:
    """Typed account inputs. cash_eur is None when not set (empty state)."""

    base_currency: str
    cash_eur: float | None
    risk_profile: str
    loaded: bool

    def risk_limits(self) -> tuple[float, float, float, float, float]:
        """Return (safety_min, core_min, alpha_max, max_position, cash_floor)."""
        return RISK_PROFILES.get(self.risk_profile, RISK_PROFILES[DEFAULT_RISK_PROFILE])

    @property
    def cash_is_set(self) -> bool:
        return self.cash_eur is not None


def load_account(path: str | None = None) -> AccountState:
    """Load data/account.yaml. Missing/invalid file -> empty cash state."""
    path = path or paths.DATA_ACCOUNT
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return AccountState(BASE_CURRENCY, None, DEFAULT_RISK_PROFILE, loaded=False)
    except Exception:  # noqa: BLE001
        return AccountState(BASE_CURRENCY, None, DEFAULT_RISK_PROFILE, loaded=False)

    cash = data.get("cash_eur")
    try:
        cash_eur = float(cash) if cash is not None else None
    except (TypeError, ValueError):
        cash_eur = None

    profile = str(data.get("risk_profile", DEFAULT_RISK_PROFILE))
    if profile not in RISK_PROFILES:
        profile = DEFAULT_RISK_PROFILE
    base = str(data.get("base_currency", BASE_CURRENCY))
    return AccountState(base, cash_eur, profile, loaded=True)


def save_account(state: AccountState, path: str | None = None) -> None:
    """Atomically write account.yaml (temp file + rename). No temp left behind."""
    path = path or paths.DATA_ACCOUNT
    payload = {
        "base_currency": state.base_currency,
        "cash_eur": state.cash_eur,
        "risk_profile": state.risk_profile,
    }
    directory = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
