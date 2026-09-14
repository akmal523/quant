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
# H3.4: exact zero-match sentence (distinct from the empty-query helper).
EMPTY_NO_MATCHES = ("No instrument matches {query}. Try a company or fund name, "
                    "a symbol, an ISIN, or a theme such as gold or defence.")
EMPTY_REGIME = "Market trend: not enough history yet."
EMPTY_REGIME_ERROR = "Market trend: unavailable (see Health)."
EMPTY_NO_MARKET_DATA = "No market data yet. Press Refresh market data to start."
EMPTY_NO_REVIEWS = "No reviews yet. Save and review from Portfolio, or wait for the daily run."
STATUS_ALL_CURRENT = "All data current."
EMPTY_VALUE_CHART = "The value chart appears after your second review."

# ── Health items (a computation that ran and failed) ──────────────────────────
HEALTH_REGIME_FAILED = "Market trend could not be estimated. See log."
HEALTH_REVIEW_FAILED = "The last review failed. See log."

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
ERROR_RUNNING = "A review is already running."

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
STATUS_BELOW_MIN = "Below minimum order"
STATUS_NOT_REVIEWED = "Not reviewed yet"
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
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]


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


# ── v10.5.3 (spec v4): state machines, charts, feedback, calendar ─────────────

# Today (spec 2.1)
GUIDE_NO_REVIEW = "No review yet. Save and review from Portfolio to get your first advice."
HEADER_REVIEW = "Review of {date} close, prepared {prepared}."
MARKET_TREND = "Market trend: {label} ({confidence} confidence)."
MARKET_TREND_INSUFFICIENT = "Market trend: not enough history yet."
MARKET_TREND_FAILED = "Market trend: unavailable (see Health)."
NEEDS_ATTENTION_ISIN = "ISIN missing for {symbol}. Repair it in Settings."
LAST_REVIEW_FAILED = "The last review failed. See Settings for details."
FOOTNOTE_BELOW_MIN = ("{n} positions sit outside target; the moves are below the "
                      "{min} EUR minimum order size.")
FOOTNOTE_BELOW_MIN_ONE = ("1 position sits outside target; the move is below the "
                          "{min} EUR minimum order size.")
FOOTNOTE_COOLDOWN = "{n} positions are outside target and in their cooldown until {date}."
FOOTNOTE_COOLDOWN_ONE = "1 position is outside target and in its cooldown until {date}."

# Charts (spec 3)
CHART_BUILDING = "The value chart builds up after a few reviews."
CHART_SINCE = "{sign}{pct}% since {date} ({amount} EUR)"
LABEL_RANGE = "Range"
LABEL_VIEW = "View"
VALUE = "Value"
GROWTH = "Growth"
LABEL_BENCHMARK = "Compare to MSCI World (IWDA.AS)"
BENCHMARK_SYMBOL = "IWDA.AS"
CHART_NO_HISTORY = "No price history for {name} yet. Refresh market data in Settings."

# Explore (spec 2.2)
SCORES_NONE = "No scores for {name} yet. Scores appear after the next review."
# H3.6 (N1): freshness line when the shown scores are not from the latest attempt.
SCORES_AS_OF = "Scores as of {date}."
NEWS_CHECKING = "Checking news..."
HOW_TO_BUY_ISIN_MISSING = "ISIN missing for {name}. Repair it in Settings."
# H3.4: empty-query helper (distinct from the zero-match sentence).
SEARCH_HELPER = "Type a name, symbol, ISIN or theme to explore."
# H3.4: news display — cap the visible rows, expand the rest.
NEWS_EARLIER = "Earlier items ({n} more)"
# H3.4: one Diagnostics line when the sentiment model is absent.
SENTIMENT_UNAVAILABLE = "Sentiment model not available; news shown without sentiment."
# H3.5: discovery-universe loop closure (Explore zero-match + Portfolio add).
DISCOVERY_NOT_TRACKED = ("{label} is in the discovery universe but not tracked. "
                         "Add it in Portfolio to track it.")
NOT_TRACKED_LABEL = "{label} - not tracked yet"

# Portfolio (spec 2.3)
VALIDATION_UNIVERSE = ("{n} holdings are not in the universe yet; they will be added "
                       "on the next refresh.")
VALIDATION_UNIVERSE_ONE = "1 holding is not in the universe yet; it will be added on the next refresh."
OUTCOME_ACTIONS = "Review complete. {n} actions on Today."
OUTCOME_NOTHING = "Review complete. Nothing to do today."
SAVE_ONLY_DONE = "Saved. The next review will use these values."
LABEL_MATCHES = "Matches"
LABEL_ACCOUNT = "Account"

# Operation feedback (spec 4)
BTN_REFRESHING = "Refreshing market data..."
BTN_REVIEWING = "Reviewing..."
BTN_REPAIRING = "Repairing registry..."
OUTCOME_REFRESH = "Refreshed {n} instruments. Prices through {date} ({secs} s)."
OUTCOME_REFRESH_DONE = "Refresh complete."
VIEW_LOG = "View log"

# Calendar (spec 7)
SAVINGS_COUNTDOWN = ("Savings plan executes in {days} days ({date}); additions before "
                     "that date apply this month.")
SAVINGS_TODAY = "Savings plan executes today."
MARKETS_CLOSED = "Markets were closed; prices through {date} close."

# News outage (spec 6.3)
HEALTH_NEWS_OUTAGE = ("News source unreachable since {since}. Scores use price history "
                      "and fundamentals only.")
HEALTH_NAMES_MISSING = ("Display names missing for {n} instruments; metadata source "
                        "unreachable. Search by symbol or ISIN still works.")


def status_word(universe_status: str) -> str:
    """Map a universe_status enum to its human word."""
    return STATUS_WORDS.get(str(universe_status).upper(), "Watching")


def _parse_dt(value: date | datetime | str | None) -> datetime | None:
    """Parse ISO-8601 (with offset) or RFC-2822 into a datetime; None on failure.

    H3.6 (N2): the live news cache stores RFC-2822 ("Mon, 14 Sep 2026 14:41:24
    +0000"); the formatter must never return a raw timestamp.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day)
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        return parsedate_to_datetime(s)
    except Exception:  # noqa: BLE001
        return None


def fmt_weekday_date(value: date | datetime | str | None) -> str:
    """Format a date as 'Friday 11 Sep 2026'. Empty on bad input (H3.6, N2)."""
    dt = _parse_dt(value)
    if dt is None:
        return ""
    return f"{_WEEKDAYS[dt.weekday()]} {fmt_date(dt)}"


def fmt_weekday_ts(value: date | datetime | str | None) -> str:
    """Format a timestamp as 'Sunday 13 Sep 2026, 23:23' (H3.6, N1)."""
    dt = _parse_dt(value)
    if dt is None:
        return ""
    return f"{_WEEKDAYS[dt.weekday()]} {fmt_date(dt)}, {fmt_time(dt)}"


def fmt_review_ts(value: date | datetime | str | None) -> str:
    """Review timestamp (H3.7, L4): include a time only when the source has one.

    A date-only review_ts renders 'Sunday 13 Sep 2026' (never a bogus '00:00');
    a real timestamp renders 'Sunday 13 Sep 2026, 23:23'.
    """
    if isinstance(value, datetime):
        return fmt_weekday_ts(value)
    if isinstance(value, str) and ("T" in value or ":" in value):
        return fmt_weekday_ts(value)
    return fmt_weekday_date(value)


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
