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

# ── NER (Named Entity Recognition — semantic headline filter) ─────────────────
# Prevents false sentiment signals from irrelevant news
# (e.g., "Yellow Cake" culinary articles scored for YCA.L uranium company).
NER_ENABLED     = True
NER_SPACY_MODEL = "en_core_web_sm"  # pip install spacy && python -m spacy download en_core_web_sm

# ── Data Sources ──────────────────────────────────────────────────────────────
# Alpha Vantage free tier: 25 req/day. Set ALPHA_VANTAGE_API_KEY in .env to activate.
# When absent, yfinance is primary with exponential-backoff retries.
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

# ── Async I/O ─────────────────────────────────────────────────────────────────
MAX_ASYNC_WORKERS = 10   # ThreadPoolExecutor concurrency for yfinance fetches

# ── Scanning ──────────────────────────────────────────────────────────────────
HIST_PERIOD    = "5y"
TOP_PER_SECTOR = 3
TOP_GLOBAL     = 3

# ── Scoring Weights (must sum to 100) ─────────────────────────────────────────
# Technical weight reduced: oscillators generate short-term noise, not signal.
# Stewardship elevated: balance-sheet integrity is the non-negotiable foundation.
WEIGHT_FUNDAMENTALS = 30  # PE, PEG, ROE
WEIGHT_STEWARDSHIP  = 30  # D/E + ICR + Payout policy  (raised from 20)
WEIGHT_TECHNICAL    = 15  # Z-Score, sector-relative RSI (cut from 25)
WEIGHT_SENTIMENT    = 25  # NER-filtered FinBERT

# ── Position Sizing ───────────────────────────────────────────────────────────
# Final size = min(Kelly, TargetVol), hard-capped at MAX_POSITION_PCT.
# Quarter-Kelly guards against win-rate estimation errors.
KELLY_FRACTION    = 0.25   # Fractional multiplier on full Kelly
TARGET_VOLATILITY = 0.15   # 15% annualized target portfolio volatility
MAX_POSITION_PCT  = 0.10   # Hard cap: 10% of capital per position

# ── Backtest ──────────────────────────────────────────────────────────────────
BACKTEST_PERIOD_DAYS = 365
COMMISSION_SLIPPAGE  = 0.0015  # 15 bps per round-trip (liquid equities)

# Walk-Forward Optimization
WFO_IS_DAYS   = 365   # In-sample window: 1 year
WFO_OOS_DAYS  = 90    # Out-of-sample blind window: 3 months
WFO_STEP_DAYS = 90    # Roll step: 3 months

# ── Optional email reporting ──────────────────────────────────────────────────
SMTP_USER     = os.getenv("SMTP_USER",     "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
REPORT_TO     = os.getenv("REPORT_TO",     "")
