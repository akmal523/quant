"""live_portfolio.py — canonical fixtures for the v10.7.4 decision grid.

Every grid test builds its inputs from one of these functions; no test hardcodes
magic numbers inline. All randomness is seeded, so fixtures are deterministic.

Invariants:
  - Pure functions; no I/O, no network.
  - ``live_shaped()`` mirrors the real 4-symbol portfolio (EUNL, SXRV, AMZN,
    5J50) with real tiers and targets.
  - ``synthetic_portfolio(n)`` returns n valid holdings with tiers in
    ``VALID_TIERS``.
  - ``degenerate_cases()`` returns named edge portfolios.
  - ``adversarial()`` returns malformed inputs for the loader/doctor tests.
"""
from __future__ import annotations

import random

from quant.config import TARGET_WEIGHTS_INVESTED, VALID_TIERS

SEED = 20261002

# The real portfolio: (symbol, name, tier, value_eur, conviction).
LIVE = [
    ("EUNL.DE", "iShares Core MSCI World", "FORTRESS", 3400.0, 0.0),
    ("SXRV.DE", "iShares Nasdaq 100", "ALPHA", 2000.0, 0.0),
    ("AMZN", "Amazon.com", "ALPHA", 2900.0, 0.0),
    ("5J50.DE", "Global Aero & Defense", "FORTRESS", 1700.0, 0.0),
]

# Drift presets used by the classification matrix (Part 1.1).
DRIFT_NEAR = 0.05
DRIFT_FAR = 0.20
DRIFT_UNDER = -0.20


def holding(
    symbol: str,
    name: str,
    tier: str,
    value_eur: float,
    *,
    total: float | None = None,
    target: float | None = None,
    current_weight: float | None = None,
    conviction: float = 0.0,
    cooldown_until=None,
    **extra,
) -> dict:
    """Build one holding dict with a consistent weight pair.

    ``current_weight`` defaults to value/total; ``target_weight`` defaults to
    ``TARGET_WEIGHTS_INVESTED``. Extra keys (structural, tactical, entry_price,
    current_price, pnl_pct, pnl_eur, liquidity_score, ...) pass through.
    """
    if current_weight is None:
        current_weight = (float(value_eur) / total) if total else 0.0
    h = {
        "symbol": symbol,
        "name": name,
        "tier": tier,
        "value_eur": float(value_eur),
        "current_weight": float(current_weight),
        "target_weight": (
            TARGET_WEIGHTS_INVESTED.get(symbol, 0.0) if target is None else float(target)
        ),
        "conviction": float(conviction),
        "cooldown_until": cooldown_until,
    }
    h.update(extra)
    return h


def live_shaped() -> list[dict]:
    """The real 4-symbol portfolio with real tiers and targets."""
    total = sum(v for _, _, _, v, _ in LIVE)
    return [
        holding(s, n, t, v, total=total, conviction=c)
        for s, n, t, v, c in LIVE
    ]


def live_tiers() -> dict[str, str]:
    """The tier map for the live-shaped portfolio."""
    return {s: t for s, _, t, _, _ in LIVE}


def synthetic_portfolio(n: int, seed: int = SEED) -> list[dict]:
    """n symbols with randomized but valid tiers, values, and targets."""
    rng = random.Random(seed)
    raw = [
        {
            "symbol": f"SYN{i:03d}",
            "name": f"Synthetic {i:03d}",
            "tier": rng.choice(list(VALID_TIERS)),
            "value_eur": round(rng.uniform(50.0, 5000.0), 2),
            "conviction": rng.choice([0.0, 40.0, 60.0, 80.0, 90.0]),
        }
        for i in range(n)
    ]
    total = sum(r["value_eur"] for r in raw) or 1.0
    return [
        holding(
            r["symbol"], r["name"], r["tier"], r["value_eur"], total=total,
            target=round(rng.uniform(0.0, 0.5), 4), conviction=r["conviction"],
        )
        for r in raw
    ]


# ── Scenario builders (the matrix rows) ───────────────────────────────────────

def fortress_case(drift: float, value: float = 1000.0, cooldown=None, **extra) -> dict:
    """A single FORTRESS holding whose current weight is target + drift."""
    target = 0.20
    return holding(
        "EUNL.DE", "iShares Core MSCI World", "FORTRESS", value,
        total=value, target=target, current_weight=target + drift,
        cooldown_until=cooldown, **extra,
    )


def alpha_case(
    drift: float, value: float = 500.0, cooldown=None, conviction: float = 0.0, **extra
) -> dict:
    """A single ALPHA holding whose current weight is target + drift."""
    target = 0.20
    return holding(
        "AMZN", "Amazon.com", "ALPHA", value,
        total=value, target=target, current_weight=target + drift,
        conviction=conviction, cooldown_until=cooldown, **extra,
    )


def spec_case(
    value: float = 40.0, pnl_pct: float | None = None, conviction: float = 0.0, **extra
) -> dict:
    """A single SPECULATIVE holding (bets weight is 100 percent on its own)."""
    return holding(
        "BET", "Small Bet", "SPECULATIVE", value,
        total=value, target=0.02, conviction=conviction, pnl_pct=pnl_pct, **extra,
    )


def spec_portfolio(
    spec_value: float = 40.0,
    base_value: float = 10_000.0,
    pnl_pct: float | None = None,
    conviction: float = 0.0,
) -> list[dict]:
    """A large FORTRESS base plus one SPECULATIVE holding.

    Keeps the bets weight below the 2 percent cap unless ``spec_value`` is large
    enough to breach it, so the take-profit and cap-violation rows are isolated.
    """
    total = base_value + spec_value
    base = holding(
        "EUNL.DE", "iShares Core MSCI World", "FORTRESS", base_value,
        total=total, target=0.50,
    )
    spec = holding(
        "BET", "Small Bet", "SPECULATIVE", spec_value,
        total=total, target=0.02, conviction=conviction, pnl_pct=pnl_pct,
    )
    return [base, spec]


def degenerate_cases() -> dict[str, list[dict]]:
    """Named edge portfolios (Part 5.1)."""
    one_fortress = [
        holding("EUNL.DE", "iShares Core MSCI World", "FORTRESS", 1000.0,
                total=1000.0, target=0.50)
    ]
    hundred_alpha = [
        holding(f"ALP{i:03d}", f"Alpha {i:03d}", "ALPHA", 100.0,
                total=100.0 * 100, target=0.01)
        for i in range(100)
    ]
    all_cash = [
        holding("XEON.DE", "Money Market", "FORTRESS", 500.0,
                total=500.0, target=1.0)
    ]
    negative = [
        holding("BAD", "Bad Value", "ALPHA", -100.0, total=100.0, target=0.10)
    ]
    return {
        "zero": [],
        "one_fortress": one_fortress,
        "hundred_alpha": hundred_alpha,
        "all_cash": all_cash,
        "negative_value": negative,
    }


def adversarial() -> dict:
    """Malformed inputs for the loader/doctor tests (Part 5.2)."""
    return {
        "orphan_tiers": {"NOT_HELD": "FORTRESS"},
        "missing_prices": [
            holding("NOPRICE", "No Price", "ALPHA", 500.0, total=500.0, target=0.20)
        ],
        "duplicate_rows": [
            holding("AMZN", "Amazon.com", "ALPHA", 1000.0, total=2000.0, target=0.20),
            holding("AMZN", "Amazon.com", "ALPHA", 1000.0, total=2000.0, target=0.20),
        ],
        "negative_value": [
            holding("BAD", "Bad Value", "ALPHA", -100.0, total=100.0, target=0.10)
        ],
        "portfolio_csv": (
            "Symbol,Avg_Entry_Price,Current_Value_EUR,Broker_PnL_EUR\n"
            "AMZN,100,1000,50\n"
            "AMZN,100,1000,50\n"
            "BAD,10,-100,0\n"
        ),
        "tiers_csv": (
            "symbol,tier,last_updated,notes\n"
            "NOT_HELD,FORTRESS,2026-10-01,orphan tier assignment\n"
        ),
    }
