"""
config.py — the app's fixed settings: fees, risk profiles, alert and decision
thresholds, the funnel, and the per-symbol target map. Pure constants.
"""

# ── Currency ──────────────────────────────────────────────────────────────────
BASE_CURRENCY  = "EUR"

# ── Trade Republic Execution Reality (Phase 4) ────────────────────────────────
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
