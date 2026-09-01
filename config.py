"""
config.py — All runtime settings. Values override via .env.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── Currency ──────────────────────────────────────────────────────────────────
BASE_CURRENCY  = "EUR"
CONVERT_TO_EUR = True

# ── FinBERT ───────────────────────────────────────────────────────────────────
FINBERT_MODEL         = "ProsusAI/finbert"
FINBERT_MAX_HEADLINES = 32
FINBERT_DEVICE        = -1    # -1 = CPU; 0 = first CUDA GPU

# ── NER ───────────────────────────────────────────────────────────────────────
NER_ENABLED     = True
NER_SPACY_MODEL = "en_core_web_sm"

# ── Data Sources ──────────────────────────────────────────────────────────────
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

# ── Async I/O ─────────────────────────────────────────────────────────────────
MAX_ASYNC_WORKERS = 10

# ── Scanning ──────────────────────────────────────────────────────────────────
HIST_PERIOD    = "5y"
TOP_PER_SECTOR = 3
TOP_GLOBAL     = 3

# ── Scoring Weights (must sum to 100) ─────────────────────────────────────────
WEIGHT_FUNDAMENTALS = 30
WEIGHT_STEWARDSHIP  = 30
WEIGHT_TECHNICAL    = 15
WEIGHT_SENTIMENT    = 25

# ── Position Sizing ───────────────────────────────────────────────────────────
KELLY_FRACTION    = 0.25
TARGET_VOLATILITY = 0.15
MAX_POSITION_PCT  = 0.10

# ── Backtest ──────────────────────────────────────────────────────────────────
BACKTEST_PERIOD_DAYS = 365
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

# ── Cash as Risk-Free Baseline (Phase 4) ──────────────────────────────────────
# TR pays 2.25% APY on uninvested cash. This is the REAL risk-free rate (R_f),
# not the theoretical US Treasury yield. Converted to daily in risk.py.
BROKER_CASH_APY = 0.0225

# ── Smart Balance Risk Buckets (Phase 4) ──────────────────────────────────────
# Hard inequality constraints for the cvxpy optimizer. Prevents 100% allocation
# into a handful of volatile tech stocks.
SAFETY_BUCKET_MIN = 0.10   # Cash & short-term bonds (2.25% risk-free)
CORE_BUCKET_MIN   = 0.40   # Broad ETFs via Sparplan (free execution)
ALPHA_BUCKET_MAX  = 0.50   # Active equities (1 EUR fee, high conviction)

# ── Signal Routing (Phase 4) ──────────────────────────────────────────────────
# Structural grade threshold for long-term hold -> route to Sparplan.
# Tactical grade threshold for immediate breakout -> route to Active Trade.
SPARPLAN_STRUCT_MIN = 75.0
ACTIVE_TACT_MIN     = 70.0

# ── Universe Graduation (Phase 4) ─────────────────────────────────────────────
WATCHLIST_LOOKBACK_DAYS = 5      # discovery.py: only fetch last 5 days
WATCHLIST_VOLUME_MULT   = 3.0    # volume > 3x 20-day avg -> graduate
WATCHLIST_52W_HIGH_DAYS = 252    # cross 52-week high -> graduate
ACTIVE_DEMOTE_MONTHS    = 6      # no signals for 6 months -> demote to watchlist

# ── Optional email reporting ──────────────────────────────────────────────────
SMTP_USER     = os.getenv("SMTP_USER",     "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
REPORT_TO     = os.getenv("REPORT_TO",     "")

# ── Strategy ──────────────────────────────────────────────────────────────────
STRATEGY_NAME = "High_Efficiency_Growth"

# ── Fast Filter / Screener Thresholds ─────────────────────────────────────────
FILTER_MAX_PE  = 25.0
FILTER_MIN_ROE = 0.20

# ── Fundamental Grade Thresholds ──────────────────────────────────────────────
STRUCT_MAX_PE  = 20.0
STRUCT_MAX_PEG = 1.5
STRUCT_MIN_ROE = 0.15

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
