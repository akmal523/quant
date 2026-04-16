"""
config.py — centralised configuration for Trade Republic Quant-AI v4.5
All secrets are loaded from environment variables (see .env.example).
"""
import os

# ─────────────────────────────────────────────────────────────────────────────
# API / SMTP  (set in .env or export before running)
# ─────────────────────────────────────────────────────────────────────────────
#GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")

# Локальная модель сентимента (Hugging Face)
SENTIMENT_MODEL = "ProsusAI/finbert"

# Конфигурация SMTP (если используется отправка email)
SMTP_USER = "@gmail.com"
SMTP_PASSWORD = "your_app_password"
REPORT_TO = "recipient@example.com"


SMTP_HOST       = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT       = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER       = os.getenv("SMTP_USER", "")
SMTP_PASSWORD   = os.getenv("SMTP_PASSWORD", "")
REPORT_TO       = os.getenv("REPORT_TO", "")

# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
CONVERT_TO_EUR  = True          # append EUR equivalent to USD prices
PRICE_WARN_PCT  = 1.50          # flag if price deviates > 150 % from EMA50
#GEMINI_MODEL    = "gemini-2.5-flash"
BATCH_SIZE      = 5             # tickers per Gemini request
START_DATE      = "2021-01-01"  # macro history start
ATR_MULTIPLIER  = 2.5           # stop-loss = entry − ATR × multiplier
VERSION         = "4.5"

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR      = "outputs"
