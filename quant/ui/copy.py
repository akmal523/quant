"""
copy.py — Central user-facing copy + formatting (v10.5.1, spec 3).

Intent: ONE module holds every user-facing string and every number formatter.
Pages import from here so the vocabulary is consistent and the banned-token
test (spec 9) can scan a single source. No internal identifiers (run ids,
command names, file names, paths, column names, enum values, exit codes) may
appear in any string here (P2).

Invariants:
  - No emoji, no exclamation marks (P8).
  - Every number carries its unit inline (P10).
  - Pure module: no I/O, no imports from the pipeline.

Dependencies: datetime only.
"""
from __future__ import annotations

from datetime import date, datetime

# ── Page titles (P12) ─────────────────────────────────────────────────────────
PAGE_TODAY = "Today"
PAGE_PORTFOLIO = "Portfolio"
PAGE_EXPLORE = "Explore"
PAGE_SETTINGS = "Settings"

# ── Section titles ────────────────────────────────────────────────────────────
SEC_WHAT_TO_DO = "What to do today"
SEC_NEEDS_ATTENTION = "Needs attention first"
SEC_YOUR_PORTFOLIO = "Your portfolio"
SEC_PORTFOLIO_VALUE = "Portfolio value"
SEC_WHERE_MONEY = "Where your money is"
SEC_WHY_SCORES = "Why these scores"
SEC_NEWS = "News and filings"
SEC_HOW_TO_BUY = "How to buy"
SEC_DATA_STATUS = "Data status"
SEC_REVIEWS = "Reviews"
SEC_DIAGNOSTICS = "Diagnostics"
SEC_GLOSSARY = "What do these mean?"

# ── Buttons (P5: verb phrases describing the outcome) ─────────────────────────
BTN_REFRESH = "Refresh market data"
BTN_SAVE_AND_REVIEW = "Save and review"
BTN_SAVE_ONLY = "Save only"
BTN_VIEW_LOG = "View log"
BTN_REPAIR_REGISTRY = "Repair registry"
BTN_HOW_TO_BUY = "How to buy"
BTN_TRY_AGAIN = "Try again"
BTN_OPEN_TODAY = "Open Today"

# ── Empty and status states (exact strings, spec 3.2) ─────────────────────────
# P4 (v10.5.2): an empty state may describe a MISSING-DATA condition only. A
# computation that ran and failed is a Health item, never an empty state.
EMPTY_NOTHING_TO_DO = "Nothing to do today. The next review runs after the next market close."
EMPTY_NO_NEWS = "No recent news for {name}. Scores use price history and fundamentals only."
EMPTY_NO_MATCHES = "No instrument matches {query}."
EMPTY_REGIME = "Market trend: not enough history yet."
EMPTY_REGIME_ERROR = "Market trend: unavailable (see Health)."
EMPTY_NO_MARKET_DATA = "No market data yet. Press Refresh market data to start."
EMPTY_NO_REVIEWS = "No reviews yet. Save and review from Portfolio, or wait for the daily run."
STATUS_ALL_CURRENT = "All data current."
EMPTY_VALUE_CHART = "The value chart appears after your second review."

# ── Health items (a computation that ran and failed) ──────────────────────────
HEALTH_REGIME_FAILED = "Market trend could not be estimated. See log."

# ── Action cards (spec 3.2) ───────────────────────────────────────────────────
ACTION_ADD = ("Add about {amount} EUR to {symbol} ({name}). It sits {pct} percent below "
              "its {target} percent target. Suitable for your savings plan.")
ACTION_SELL = "Sell about {amount} EUR of {symbol}. It sits {pct} percent above its {target} percent target."
ACTION_BLOCKED = "ISIN missing for {symbol}."
ACTION_BLOCKED_MANUAL = ("ISIN missing for {symbol}. Not found automatically. Add a verified row "
                         "to data/isin_curated.csv (from your broker app or the fund factsheet), "
                         "then run Repair registry again.")

# ── Errors (spec 3.2) ─────────────────────────────────────────────────────────
ERROR_DB_BUSY = ("The database is busy because another Quant-AI session is open. "
                 "Close other tabs or terminals, then try again.")
ERROR_REFRESH_FAILED = "Refresh failed. Open View log for details, or try again."
ERROR_STALE_PRICES = "Prices are {n} days old. Refresh market data."
ERROR_ALREADY_RUNNING = "A review is already running in another tab."

# ── Helper texts (spec 3.2) ───────────────────────────────────────────────────
HELP_CASH_APY = ("Uninvested cash earns {apy} percent per year at Trade Republic "
                 "(from {date}).")
HELP_BROKER_VALUES = "Values come from your broker. The app never guesses them."
HELP_PROFILE_CONSERVATIVE = "More safety assets and cash, smaller bets."
HELP_PROFILE_BALANCED = "The default mix of core funds and tactical positions."
HELP_PROFILE_AGGRESSIVE = "Larger tactical positions, thinner cash buffer."
HELP_REVIEW_CADENCE = "Reviews run once a day after market close. Refresh manually anytime."
# A5: the add-holding affordance states what it does and what to do next.
PLACEHOLDER_ADD_HOLDING = "Type a name, symbol or ISIN to add a holding"
HELP_ADD_ROW = "Now fill value and profit or loss from your broker."
# A6: provenance caveat for auto-filled (yahoo) ISINs.
HELP_ISIN_YAHOO_CAVEAT = ("Auto-filled from market data. Confirm in your broker app "
                          "before ordering.")

# ── Glossary (spec 3.2) ───────────────────────────────────────────────────────
GLOSSARY = {
    "Quality score": "how sound the asset is fundamentally, 0 to 100.",
    "Trend score": "how strong the recent price trend is, 0 to 100.",
    "Overall score": "the blend the review uses to rank assets, 0 to 100.",
    "Market trend": "the estimated regime of the broad market, with confidence.",
}

# ── Internal-to-human dictionary (spec 3.1) ───────────────────────────────────
CLASS_WORDS = {
    "ETF": "Fund",
    "EQUITY": "Stock",
    "COMMODITY": "Commodity",
    "CASH": "Cash",
}
REGIME_WORDS = {
    "bull": "rising",
    "bear": "falling",
    "chop": "mixed",
}
STATUS_WORDS = {
    "WATCHLIST": "Watching",
    "ACTIVE": "Active",
    "CORE": "Core",
    "DELISTED": "Delisted",
}
# A3 (v10.5.2): the ONE status vocabulary. The holdings table and the action
# cards both read these words via status_for(), so two widgets on one screen can
# never disagree (P2 trust rule).
STATUS_ON_TRACK = "On track"
STATUS_ADD = "Add"
STATUS_TRIM = "Trim"
STATUS_BLOCKED = "Blocked"
STATUS_WAITING = "Waiting until {date}"
# Column-name -> human header (P2: no column names in the UI).
COLUMN_HEADERS = {
    "Symbol": "Instrument",
    "Avg_Entry_Price": "Average entry price",
    "Current_Value_EUR": "Current value (EUR)",
    "Broker_PnL_EUR": "Profit or loss (EUR)",
    "Current_Weight": "Share",
    "Target_Weight": "Target",
    "Drift": "Difference from target",
    "Signal": "Status",
}

# ── First-run guide (spec 8) ──────────────────────────────────────────────────
FIRST_RUN_STEPS = [
    "Open Portfolio and add your holdings.",
    "Set cash and risk profile.",
    "Press Save and review.",
]

# ── Formatting helpers (spec 3.3) ─────────────────────────────────────────────
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def fmt_eur(value: float | None) -> str:
    """Format EUR with two decimals and the unit after the number."""
    if value is None:
        return ""
    return f"{value:.2f} EUR"


def fmt_pct(fraction: float | None, decimals: int = 0) -> str:
    """Format a fraction as a percent. Integer by default (spec 3.3)."""
    if fraction is None:
        return ""
    return f"{fraction * 100:.{decimals}f}%"


def fmt_date(value: date | datetime | str | None) -> str:
    """Format a date as '13 Sep 2026'."""
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value
    if not isinstance(value, (date, datetime)):
        return ""
    return f"{value.day} {_MONTHS[value.month - 1]} {value.year}"


def fmt_time(value: datetime | str | None) -> str:
    """Format a time as '17:43'."""
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value
    return value.strftime("%H:%M")


def fmt_ts(value: datetime | str | None) -> str:
    """Format a timestamp as '13 Sep 2026, 17:43'."""
    if value is None:
        return ""
    return f"{fmt_date(value)}, {fmt_time(value)}"


def fmt_score(value: float | None) -> str:
    """Format a score as '67 / 100'."""
    if value is None:
        return ""
    return f"{value:.0f} / 100"


def class_word(instrument_class: str) -> str:
    """Map an instrument_class enum to its human word."""
    return CLASS_WORDS.get(str(instrument_class).upper(), "Asset")


def regime_word(regime: str) -> str:
    """Map a regime enum to its human word."""
    return REGIME_WORDS.get(str(regime).lower(), "mixed")


def status_word(universe_status: str) -> str:
    """Map a universe_status enum to its human word."""
    return STATUS_WORDS.get(str(universe_status).upper(), "Watching")


def status_for(
    recommendation: str,
    blocked: bool = False,
    cooldown_until: date | datetime | str | None = None,
) -> str:
    """Map an audit recommendation to the ONE canonical status word (A3).

    Intent: the holdings table and the action cards must read the same audit
    object, so a position 7.5 points over target during cooldown cannot read
    "On track" in one widget and "Waiting" in another.
    Invariants:
      - blocked -> Blocked (an ISIN blocker outranks any drift).
      - actionable (BUY/SELL) while in cooldown -> Waiting until {date}.
      - BUY -> Add, SELL -> Trim, otherwise -> On track.
    """
    if blocked:
        return STATUS_BLOCKED
    rec = str(recommendation or "")
    actionable = rec.startswith("BUY") or rec.startswith("SELL")
    if actionable and cooldown_until is not None:
        return STATUS_WAITING.format(date=fmt_date(cooldown_until))
    if rec.startswith("BUY"):
        return STATUS_ADD
    if rec.startswith("SELL"):
        return STATUS_TRIM
    return STATUS_ON_TRACK
