"""
config.py — All runtime settings. Values override via .env.
"""
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ── Currency ──────────────────────────────────────────────────────────────────
BASE_CURRENCY  = "EUR"

# ── FinBERT ───────────────────────────────────────────────────────────────────
FINBERT_MODEL         = "ProsusAI/finbert"

# ── Data Sources ──────────────────────────────────────────────────────────────
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

# ── Async I/O ─────────────────────────────────────────────────────────────────
MAX_ASYNC_WORKERS = 10

# ── Scoring Weights (must sum to 100) ─────────────────────────────────────────
WEIGHT_STEWARDSHIP  = 30
WEIGHT_TECHNICAL    = 15

# ── Backtest ──────────────────────────────────────────────────────────────────
COMMISSION_SLIPPAGE  = 0.0015
WFO_IS_DAYS   = 365
WFO_OOS_DAYS  = 90
WFO_STEP_DAYS = 90

# ── Trade Republic Execution Reality (Phase 4) ────────────────────────────────
# TR fee structure: 1 EUR per active trade, 0 EUR Sparplan buy / 1 EUR sell.
# Round-trip active trade = 2 EUR. This is the asymmetric hurdle rate.
ROUND_TRIP_FEE_EUR = 2.0
ACTIVE_TRADE_FEE_EUR = 1.0
SPARPLAN_BUY_FEE_EUR = 0.0
SPARPLAN_SELL_FEE_EUR = 1.0

# ── v10.7.0: The three money pools (root fix) ─────────────────────────────────
BETS_MAX      = 0.02   # small bets (SPECULATIVE) at most 2 percent of invested
# max single position is per risk profile (see RISK_PROFILES below).

# ── v10.7.0: Risk Profiles (invested pool only) ───────────────────────────────
# The user picks one profile in data/account.yaml. Tuple order:
#   (long_term_min, active_max, max_position)
# All three are fractions of the INVESTED pool. The old cash floor is removed:
# operational cash is not an investment buffer and is never constrained.
RISK_PROFILES = {
    "conservative": (0.50, 0.40, 0.25),
    "balanced":     (0.40, 0.50, 0.35),
    "aggressive":   (0.30, 0.65, 0.45),
}
DEFAULT_RISK_PROFILE = "balanced"

# Plain-word consequences, one sentence per profile (invested pool only).
RISK_PROFILE_DESCRIPTIONS = {
    "conservative": ("At least 50 percent of your invested money in long-term "
                     "assets, at most 40 percent in active positions."),
    "balanced": ("At least 40 percent of your invested money in long-term assets, "
                 "at most 50 percent in active positions."),
    "aggressive": ("At least 30 percent of your invested money in long-term assets, "
                   "at most 65 percent in active positions."),
}

# ── v10.7.0: Alert thresholds (level-triggered, Section 5) ────────────────────
ALERT_STRUCTURAL_FLOOR = 40.0        # structural grade below this -> alert
ALERT_STRUCTURAL_DROP = 20.0         # dropped >= 20 points since last monthly decision
ALERT_TACTICAL_DROP = 25.0           # tactical grade dropped >= 25 points within window
ALERT_POSITION_CRASH = 0.15          # holding value down >= 15 percent within window
ALERT_SPECULATIVE_STOP = -0.50       # SPECULATIVE down 50 percent from recorded entry

# ── v10.7.0: Scheduler slots (local time, Section 3) ──────────────────────────
SCHEDULE_DAILY_SLOT = "18:45"        # full daily job after market close
SCHEDULE_MORNING_SLOT = "07:45"      # second daily run (catches a missed evening)

# ── v10.7.2: Runner lock thresholds (Part 1) ──────────────────────────────────
HARD_LOCK_MINUTES = 240
# Retry budget while a live owner holds the lock below the stale threshold.
LOCK_RETRY_ATTEMPTS = 5
LOCK_RETRY_TOTAL_SECONDS = 60.0
# Writer connection retry budget on DuckDB lock/IO errors (Part 1.2).
WRITE_RETRY_ATTEMPTS = 5
WRITE_RETRY_TOTAL_SECONDS = 60.0

# ── v10.7.0: Sizing laws (Section 6) ──────────────────────────────────────────
SELL_MIN_EUR = 25.0                  # below this, no sell advice (fee relevance)
SELL_ROUND_STEP_EUR = 5.0            # round sell amounts down to 5 EUR steps
BUY_ROUND_STEP_EUR = 10.0            # round buy amounts to 10 EUR steps
UNTOUCHABLE_POSITION_EUR = 100.0     # never sell a position below this value

# ── Plan 3 (Phase 1): Multi-Stage Funnel thresholds ───────────────────────────
# Stage 1 (liquidity/viability): filter the 1000+ universe_master down to a
# viable pool. min daily DOLLAR volume = price * shares traded.
FUNNEL_MIN_PRICE = 5.0
FUNNEL_MIN_DAILY_VOLUME = 1_000_000   # min daily dollar volume (price*volume)
FUNNEL_STAGE1_TARGET = 300             # soft cap on Stage 1 survivors
FUNNEL_TOP_N = 24                      # Stage 2 survivors for heavy analysis
FUNNEL_MAX_WORKERS = 10

# CORE sleeve: immutable, never graduated, never demoted. Broad market, water,
# bonds, and money market ETFs. These are the long-term accumulation vehicles.
CORE_ETFS = [
    "URTH", "IWDA.AS", "EUNL.DE", "VOO", "CSPX.L", "EIMI.L",
    "SWRD.L", "EXW1.DE", "EXIA.DE", "ISF.L", "IEAG.L", "ITPS.L",
    "XEON.DE", "SXRV.DE", "IQQQ.DE", "GLUG.L", "PHO",
]

# ── Strategic Rebalancing Tiers (v11) ─────────────────────────────────────────
# Intent: differentiate buy-and-hold vs active trading assets so the bot stops
# emitting SELL on core ETFs over minor noise. Tier lists take PRECEDENCE over
# CORE_ETFS for tier assignment (e.g. SXRV.DE is in CORE_ETFS but classified
# SATELLITE here). classify_asset() checks these lists first.
# Invariants: a symbol maps to exactly one tier; unknown symbols default ACTIVE.
CORE_ASSETS = [
    "EUNL.DE", "IWDA.AS", "VOO", "CSPX.L", "URTH", "EIMI.L",
    "SWRD.L", "EXW1.DE", "EXIA.DE", "ISF.L", "IEAG.L", "ITPS.L",
    "XEON.DE", "IQQQ.DE", "GLUG.L", "PHO",
]
SATELLITE_ASSETS = ["SXRV.DE", "QQQ", "TQQQ", "IQQQ.DE"]
ACTIVE_ASSETS = ["AMZN", "AAPL", "TSLA", "MSFT", "NVDA", "GOOGL", "META"]
SECTOR_ASSETS = ["5J50.DE", "URA", "TAN", "ICLN", "XLE", "XLK"]

# Target portfolio weights per tier (must sum to 1.0). Used for drift analysis.
# Intent: 50% broad market, 20% growth/leveraged, 20% individual stocks,
# 10% sector plays. First-run baseline eases in (no forced rebalance).
TARGET_WEIGHTS = {
    "CORE": 0.50,
    "SATELLITE": 0.20,
    "ACTIVE": 0.20,
    "SECTOR": 0.10,
}

# ── v10.7.3 (Part 5.1): ONE invested-pool target map ──────────────────────────
# The single source of per-symbol targets for the invested pool. build_advice,
# the ONE list and the drift computations all read this map (via
# quant.engine.weights.target_weight), so two screens can never disagree. Seeded from the
# legacy tier map normalized to the invested pool; overridable per symbol by an
# optional target_pct column in data/tiers.csv.
TARGET_WEIGHTS_INVESTED = {
    "EUNL.DE": 0.50,
    "SXRV.DE": 0.20,
    "AMZN": 0.20,
    "5J50.DE": 0.10,
}

# Rebalancing thresholds & frequencies.
REBALANCE_DRIFT_THRESHOLD = 0.05   # generic drift trigger (5%)
MIN_TRADE_SIZE_EUR = 50.0          # minimum trade to clear the 2 EUR round-trip

# v10.8.0 (2.3): ONE tier vocabulary. The legacy 4-tier keys (CORE/SATELLITE/
# ACTIVE/SECTOR) are gone; legacy values are carried over through
# LEGACY_TIER_MAPPING. FORTRESS rebalances only on >10% drift and quarterly.
# Days between allowed rebalances per tier.
REBALANCE_FREQUENCY_DAYS = {
    "FORTRESS": 90,     # quarterly (was CORE)
    "ALPHA": 7,         # weekly
    "SPECULATIVE": 7,   # weekly
}

# Per-tier drift thresholds (fraction) that trigger a rebalance.
REBALANCE_DRIFT_TIERS = {
    "FORTRESS": 0.10,   # only major drift (was CORE)
    "ALPHA": 0.05,
    "SPECULATIVE": 0.05,
}

# ── v10.8.7 (T4): decision-taxonomy thresholds (ONE config module) ────────────
# Every verb threshold lives here; no magic numbers in the UI or the engine.
THREAT_DRAWDOWN = 0.20        # 20% below entry -> Threat
SCORE_FLOOR = 40.0            # composite below this -> Threat
SPEC_STOP = -0.50             # Speculative stop-loss
SPEC_TAKE = 1.00              # Speculative take-profit
SWAP_HURDLE = 0.05            # 5 points after fees/tax/confidence -> Swap
PLAN_CHANGE_MONTHS = 3        # Long-term under target for K plan runs -> Plan change
PLAN_CHANGE_GAP = 0.10        # under-target gap that triggers a plan change

# ── v10.8.7 (T2): scoring + ideas ─────────────────────────────────────────────
# A watched (non-held) symbol with a composite at or above this floor becomes an
# Idea and flows into Research, swap candidates, the invest plan and This month.
IDEA_FLOOR = 55.0
# The score a missing component contributes (R-C2): deliberately below neutral,
# so a symbol with prices only cannot reach IDEA_FLOOR.
SCORE_MISSING_PENALTY = 12.0
# Component weights for the composite (must sum to 1.0). Technical is always
# present (prices); the others fall back to SCORE_MISSING_PENALTY when missing.
SCORE_COMPONENT_WEIGHTS = {
    "technical": 0.40,
    "fundamental": 0.30,
    "sentiment": 0.20,
    "stewardship": 0.10,
}
# Confidence tier mapping (documented in CONTEXT.md): present/total ratio.
CONFIDENCE_HIGH_RATIO = 0.8
CONFIDENCE_MEDIUM_RATIO = 0.5

# ── Fast Filter / Screener Thresholds ─────────────────────────────────────────
FILTER_MAX_PE  = 25.0
FILTER_MIN_ROE = 0.20

# ── Fundamental Grade Thresholds ──────────────────────────────────────────────
STRUCT_MAX_PEG = 1.5

# ── Stewardship (Non-Financials) ──────────────────────────────────────────────
STW_GEN_MAX_DE  = 0.5
STW_GEN_MID_DE  = 1.0
STW_GEN_MIN_ROE = 0.10
STW_GEN_HI_ROE  = 0.20
STW_GEN_MIN_ICR = 5.0

# ── Stewardship (Financials/Banks) ────────────────────────────────────────────
STW_FIN_MIN_PB  = 1.0
STW_FIN_MAX_PB  = 1.5
STW_FIN_MIN_ICR = 3.0

# ── Allocation Logic Thresholds ───────────────────────────────────────────────
MIN_STRUCT_GRADE_FOR_BUY = 75
MIN_TACT_GRADE_FOR_BUY   = 70

# ── Data Confidence ──────────────────────────────────────────────────────────
# When no SEC/News data is available, the sentiment component is unreliable.
# This penalty reduces the tactical grade to prevent false BUY signals.
SENTIMENT_NO_DATA_PENALTY = 15.0  # max points deducted when NLP data is missing

# ── v10.4.0: Data Quality Assertions (Phase 1) ────────────────────────────────
# Hard CI/CD gate. A violation aborts the pipeline (unlike the soft repair in
# data_quality.py). A >50% one-day drop without a split flag is corruption.
ASSERT_MAX_DAILY_DROP = 0.50

# ── v10.6.2: Three-Tier Architecture (FORTRESS / ALPHA / SPECULATIVE) ────────
# Ruling R-TIER-1: the legacy 4-tier system (CORE/SATELLITE/ACTIVE/SECTOR) is
# replaced by the 3-tier system. Tier assignments live in data/tiers.csv
# (user-editable); portfolio.csv stays broker-synced. Legacy symbols map through
# LEGACY_TIER_MAPPING; an unlisted symbol defaults to ALPHA.
VALID_TIERS = ("FORTRESS", "ALPHA", "SPECULATIVE")
DEFAULT_TIER = "ALPHA"

LEGACY_TIER_MAPPING = {
    "CORE": "FORTRESS",
    "SATELLITE": "ALPHA",
    "ACTIVE": "ALPHA",
    "SECTOR": "ALPHA",
}


def _canonical_tier(tier) -> str:
    """Map a legacy tier name to its current name (v10.8.0, 2.3)."""
    t = str(tier or "").upper()
    return LEGACY_TIER_MAPPING.get(t, t)


def rebalance_threshold(tier: Any) -> float:
    """The drift threshold for a tier (legacy names mapped first)."""
    return REBALANCE_DRIFT_TIERS.get(_canonical_tier(tier), REBALANCE_DRIFT_THRESHOLD)


def rebalance_min_days(tier: Any) -> int:
    """The minimum days between rebalances for a tier (legacy names mapped first)."""
    return REBALANCE_FREQUENCY_DAYS.get(_canonical_tier(tier), 7)

# Per-tier behaviour. max_allocation is a hard cap (None = uncapped).
TIER_CONSTRAINTS = {
    "FORTRESS": {
        "rebalance_frequency": "QUARTERLY",
        "liquidity_score": "LOW",
        "tax_implications": "NEVER_SELL_TO_AVOID_TAXES",
        "max_allocation": None,
    },
    "ALPHA": {
        "rebalance_frequency": "WEEKLY_FRIDAY",
        "liquidity_score": "CALCULATED",
        "tax_implications": "CONSIDER_TAX_LOSS_HARVESTING",
        "max_allocation": 0.50,
    },
    "SPECULATIVE": {
        "rebalance_frequency": "WEEKLY_FRIDAY",
        "liquidity_score": "VARIABLE",
        "tax_implications": "HIGH_RISK_HIGH_REWARD",
        "max_allocation": 0.02,
    },
}


ALPHA_CONVICTION_HIGH = 75.0
ALPHA_CONVICTION_MEDIUM = 60.0

SPECULATIVE_TAKE_PROFIT = 1.00      # take profit at +100 percent
