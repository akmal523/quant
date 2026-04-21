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
FINBERT_MAX_HEADLINES = 32  # Increased from 8
FINBERT_DEVICE        = -1    # -1 = CPU; 0 = first CUDA GPU

# ── Scanning ──────────────────────────────────────────────────────────────────
HIST_PERIOD    = "5y"
TOP_PER_SECTOR = 3
TOP_GLOBAL     = 3

# ── Backtest ──────────────────────────────────────────────────────────────────
BACKTEST_PERIOD_DAYS = 365

# ── Optional email reporting ──────────────────────────────────────────────────
SMTP_USER     = os.getenv("SMTP_USER",     "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
REPORT_TO     = os.getenv("REPORT_TO",     "")
