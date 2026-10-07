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

from datetime import date, datetime, timedelta

# ── Page titles (P12) ─────────────────────────────────────────────────────────
# v10.7.0 nav order: Overview, Monthly decision, My holdings, Find investments,
# Settings.
PAGE_TODAY = "Overview"
PAGE_PORTFOLIO = "My holdings"
PAGE_EXPLORE = "Find investments"
PAGE_TAX = "Tax summary"
PAGE_SETTINGS = "Settings"

# ── Section titles ────────────────────────────────────────────────────────────
SEC_WHAT_TO_DO = "What to do today"
SEC_NEEDS_ATTENTION = "Needs attention first"
SEC_YOUR_PORTFOLIO = "Your portfolio"
SEC_PORTFOLIO_VALUE = "Portfolio value"
SEC_WHY_SCORES = "Why these scores"
SEC_NEWS = "News and filings"
SEC_HOW_TO_BUY = "How to buy"
SEC_DATA_STATUS = "Data status"
SEC_REVIEWS = "Reviews"
SEC_DIAGNOSTICS = "Diagnostics"
SEC_GLOSSARY = "What do these mean?"

# ── Buttons (P5: verb phrases describing the outcome) ─────────────────────────
BTN_REFRESH = "Refresh market data"
BTN_SAVE_AND_REVIEW = "Save and run review"
# v10.7.3 (Part 1.10): the button states what it does.
BTN_SAVE_AND_REVIEW_HELP = "Saves, then recomputes scores and advice now."
BTN_VIEW_LOG = "View log"
BTN_REPAIR_REGISTRY = "Repair registry"
BTN_HOW_TO_BUY = "How to buy"
BTN_TRY_AGAIN = "Try again"
BTN_OPEN_TODAY = "Open Overview"

# ── Empty and status states (exact strings, spec 3.2) ─────────────────────────
# P4 (v10.5.2): an empty state may describe a MISSING-DATA condition only. A
# computation that ran and failed is a Health item, never an empty state.
EMPTY_NOTHING_TO_DO = "Nothing to do today. The next review runs after the next market close."
# v10.8.0 (3.2): a computation that failed is a visible failure, never an
# empty success state.
COULD_NOT_CHECK = ("Could not check what to do: {reason}. "
                   "Try Refresh prices in Settings.")
EMPTY_NO_NEWS = "No recent news for {name}. Scores use price history and fundamentals only."
# H3.4: exact zero-match sentence (distinct from the empty-query helper).
EMPTY_NO_MATCHES = ("No instrument matches {query}. Try a company or fund name, "
                    "a symbol, an ISIN, or a theme such as gold or defence.")
EMPTY_REGIME = "Not enough market history yet."
EMPTY_REGIME_ERROR = "Market regime unavailable (see Health)."
EMPTY_NO_MARKET_DATA = "No market data yet. Press Refresh market data to start."
EMPTY_NO_REVIEWS = "No checks yet. Save in My holdings, or wait for the daily check."
STATUS_ALL_CURRENT = "All data current."
EMPTY_VALUE_CHART = "The value chart appears after your second review."

# ── Health items (a computation that ran and failed) ──────────────────────────
HEALTH_REGIME_FAILED = "Market trend could not be estimated. See log."
HEALTH_REVIEW_FAILED = "The last review failed. See log."

# ── Action cards (spec 3.2) ───────────────────────────────────────────────────
ACTION_ADD = ("Add about {amount} EUR to {symbol} ({name}). It sits {pct} percent below "
              "its {target} percent target. Suitable for your savings plan.")
ACTION_SELL = ("Sell about {amount} EUR of {symbol}. It sits {pct} percent "
               "above its {target} percent target.")
ACTION_BLOCKED = "ISIN missing for {symbol}."
ACTION_BLOCKED_MANUAL = ("ISIN missing for {symbol}. Not found automatically. Add a verified "
                         "row from your broker app or the fund factsheet, then press Repair "
                         "registry again.")

# ── v10.6.2 / v10.7.0: Three-tier dashboard, cash, tax-loss ──────────────────
# v10.7.0 naming dictionary (Section 11): plain words, invested pool only.
SEC_TIERS = "How your money is split"
SEC_EMERGENCY = "If you need cash now"
SEC_TAX_LOSS = "Losses you can use to lower tax"
TIER_FORTRESS = "Long-term (never sell)"
TIER_ALPHA = "Active (may sell)"
TIER_SPECULATIVE = "Small bets (high risk)"
HELP_TIER_FORTRESS = ("Never sold, to avoid capital gains tax. The only advice is "
                      "changing the monthly savings-plan amount.")
HELP_TIER_ALPHA = "May be sold when cash is needed or a position breaks."
HELP_TIER_SPECULATIVE = ("Hard cap 2 percent of invested. Stop-loss -50 percent, "
                         "take-profit +100 percent.")
EMPTY_TIER = "No holdings in this tier."
EMERGENCY_PROMPT = "How much cash do you need (EUR)?"
EMERGENCY_ORDER = "Sell in this order:"
EMERGENCY_NONE = "No Active holdings available for an emergency sale."
EMERGENCY_LINE = "{symbol} ({value} EUR) - tax {tax} EUR ({note})"
TAX_LOSS_HEADER = "Positions with an unrealized loss (losses you can use to lower tax):"
TAX_LOSS_NONE = "No positions with an unrealized loss."
TAX_LOSS_LINE = "{symbol}: {pnl} EUR unrealized loss"

# ── v10.6.3: Tier assignment alerts ───────────────────────────────────────────
TIER_UNCLASSIFIED_WARNING = "{n} assets have no tier assignment."
TIER_UNCLASSIFIED_LINE = "{symbol}: recommended tier {tier}. {reason}"
BTN_AUTO_ASSIGN_TIERS = "Auto-assign recommended tiers"
TIER_AUTO_ASSIGNED = "Tiers auto-assigned."
TIER_VALIDATION_HEADER = "Tier file issues:"
BTN_REPAIR_TIERS = "Repair tier file"
TIER_REPAIRED = "Tier file repaired."
TRADE_LIMIT_REACHED = ("Weekly Alpha trade limit reached ({n}/{max}). "
                       "Wait until next Friday.")
TRADE_LIMIT_ONE_LEFT = "Only 1 Alpha trade remaining this week. Use it wisely."

# ── v10.6.4: Auto-balance ─────────────────────────────────────────────────────
SEC_AUTOBALANCE = "How your money is split"
AUTOBALANCE_LINE = "{tier}: {value} EUR ({pct}), limit {limit}"
AUTOBALANCE_OK = "All tier allocations are within limits."
AUTOBALANCE_VIOLATION = "Tier allocation violations: {tiers}."
AUTOBALANCE_NONE = ("Violations detected but no suitable reassignments found. "
                    "Manual review required.")
AUTOBALANCE_SUGGESTIONS = "Suggested reassignments ({n}):"
AUTOBALANCE_SUGGESTION_TITLE = "Suggestion {i}: {symbol} ({source} to {target})"
AUTOBALANCE_MOVE = "Move: {source} to {target}"
AUTOBALANCE_VALUE = "Value: {value} EUR"
AUTOBALANCE_APPROVE = "Approve suggestion {i}"
BTN_APPLY_AUTOBALANCE = "Apply selected suggestions"
AUTOBALANCE_APPLIED = "Applied {n} reassignment(s)."
AUTOBALANCE_MANUAL = ("Tier changes do not execute trades. Buy or sell manually "
                      "in Trade Republic.")

# ── v10.6.3: Empty states and onboarding ──────────────────────────────────────
EMPTY_FORTRESS = ("No FORTRESS assets yet. FORTRESS is for eternal holdings you "
                  "never sell (tax-free accumulation). Recommended: broad ETFs via "
                  "a savings plan (URTH, SPY, VWRL). Add assets to data/tiers.csv "
                  "with tier FORTRESS.")
EMPTY_ALPHA = ("No ALPHA assets yet. ALPHA is for active trading (weekly "
               "rebalancing, sell when money is needed). Recommended: 5-15 tactical "
               "stocks (NVDA, TSM, AMD, AAPL, MSFT). Add assets to data/tiers.csv "
               "with tier ALPHA.")
EMPTY_SPECULATIVE = ("No SPECULATIVE assets yet. SPECULATIVE is for high-risk bets "
                     "(max 2 percent of the portfolio). Examples: penny stocks, meme "
                     "stocks, recent IPOs. These assets can go to zero; only invest "
                     "what you can afford to lose.")
ONBOARD_TITLE = "Welcome to Quant-AI"
ONBOARD_INTRO = "Set up your three-tier portfolio in three steps."
ONBOARD_STEP1 = "Step 1: add your first FORTRESS asset (an ETF you never sell)."
ONBOARD_STEP2 = "Step 2: add your first ALPHA asset (a stock you actively trade)."
ONBOARD_STEP3 = "Step 3: set your monthly savings-plan amount."
ONBOARD_SYMBOL_LABEL = "Symbol"
ONBOARD_ADD_FORTRESS = "Add to FORTRESS"
ONBOARD_ADD_ALPHA = "Add to ALPHA"
ONBOARD_SPARPLAN_LABEL = "Monthly savings plan (EUR)"
ONBOARD_COMPLETE = "Complete setup"
ONBOARD_DONE = "Setup complete. Add your holdings in My holdings, then save."
ONBOARD_SKIP = "Skip onboarding"

# ── Errors (spec 3.2) ─────────────────────────────────────────────────────────
ERROR_DB_BUSY = ("The database is busy because another Quant-AI session is open. "
                 "Close other tabs or terminals, then try again.")
# v10.7.2 (Part 1.3): the exact user-facing string when the daily job cannot
# acquire the database after retries. Stored here so the CLI, the marker, and
# the doctor all render one sentence.
DB_BUSY_RETRY = ("Database is busy: another quant process is writing. The run "
                 "will retry; if this persists, close other quant windows and "
                 "rerun.")
ERROR_REFRESH_FAILED = "Refresh failed. Open View log for details, or try again."
ERROR_STALE_PRICES = "Prices are {n} days old. Refresh market data."
ERROR_ALREADY_RUNNING = "A review is already running in another tab."
# H3.8 (M8): own-session copy is operation-agnostic (refresh or review).
ERROR_RUNNING = "An operation is already running."

# ── Helper texts (spec 3.2) ───────────────────────────────────────────────────
HELP_CASH_APY = ("Uninvested cash earns {apy} percent per year at Trade Republic "
                 "(from {date}).")
HELP_BROKER_VALUES = ("Values come from your broker. Between syncs we show estimates "
                      "from known shares and the latest price, labeled estimated.")
HELP_PROFILE_CONSERVATIVE = ("At least 50 percent of your invested money in "
                             "long-term assets, at most 40 percent in active positions.")
HELP_PROFILE_BALANCED = ("At least 40 percent of your invested money in long-term "
                         "assets, at most 50 percent in active positions.")
HELP_PROFILE_AGGRESSIVE = ("At least 30 percent of your invested money in long-term "
                           "assets, at most 65 percent in active positions.")
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
# never disagree (P2 trust rule). v10.7.0: statuses are verbs with EUR amounts
# (Section 11 dictionary).
STATUS_ON_TRACK = "OK, do nothing"
STATUS_ADD = "Buy more (regular buying is fine)"
STATUS_TRIM = "Sell part"
STATUS_BLOCKED = "Blocked"
STATUS_WAITING = "Cooldown until {date} (do nothing)"
STATUS_BELOW_MIN = "Below minimum order"
STATUS_NOT_REVIEWED = "Not reviewed yet"
# Column-name -> human header (P2: no column names in the UI).
COLUMN_HEADERS = {
    "Symbol": "Instrument",
    "Avg_Entry_Price": "Entry price per share",
    "Current_Value_EUR": "Value (EUR)",
    "Broker_PnL_EUR": "Profit (EUR)",
    "Current_Weight": "Now / should be (of invested)",
    "Target_Weight": "Target",
    "Drift": "Difference from target",
    "Signal": "Status",
}

# ── First-run guide (spec 8) ──────────────────────────────────────────────────
FIRST_RUN_STEPS = [
    "Open My holdings and add your positions.",
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


def fmt_eur_whole(value: float | None) -> str:
    """Format EUR as a whole number with the unit (v10.7.3, Part 1.12).

    Big plaques show whole EUR ("852 EUR"); cents live in the caption line.
    Tables keep two decimals via fmt_eur.
    """
    if value is None:
        return ""
    return f"{value:.0f} EUR"


def fmt_pct(fraction: float | None, decimals: int = 0) -> str:
    """Format a fraction as a percent. Integer by default (spec 3.3)."""
    if fraction is None:
        return ""
    return f"{fraction * 100:.{decimals}f}%"


def _is_missing(value) -> bool:
    """True for None or a NaN/NaT-like value (value != value)."""
    if value is None:
        return True
    try:
        return bool(value != value)
    except Exception:  # noqa: BLE001
        return False


def fmt_date(value: date | datetime | str | None) -> str:
    """Format a date as '13 Sep 2026'."""
    if _is_missing(value):
        return ""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            return value
    if not isinstance(value, date | datetime):
        return ""
    return f"{value.day} {_MONTHS[value.month - 1]} {value.year}"


def fmt_time(value: datetime | str | None) -> str:
    """Format a time as '17:43'."""
    if _is_missing(value):
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
GUIDE_NO_REVIEW = "No check yet. Save in My holdings to get your first advice."
HEADER_REVIEW = "Report as of {date} close, prepared {prepared}."
# H3.8 (M1): a legacy artifact with no close date renders only the prepared line.
# v10.7.3 (Part 1.1): "Review prepared ..." becomes "Report as of ...".
HEADER_REVIEW_PREPARED = "Report as of {prepared}."
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
# H3.8 (M3): Growth mode annotation is percent-only.
CHART_SINCE_PCT = "{sign}{pct}% since {date}"
LABEL_RANGE = "Range"
LABEL_VIEW = "View"
VALUE = "Value"
GROWTH = "Growth"
# v10.7.3 (Part 7.2): Growth is an explicit opt-in.
GROWTH_VIEW = "Growth view (normalized to 100)"
LABEL_BENCHMARK = "Compare to MSCI World (IWDA.AS)"
BENCHMARK_SYMBOL = "IWDA.AS"
CHART_NO_HISTORY = "No price history for {name} yet. Refresh market data in Settings."

# Explore (spec 2.2)
SCORES_NONE = "No scores for {name} yet. Scores appear after the next review."
# H3.6 (N1): freshness line when the shown scores are not from the latest attempt.
SCORES_AS_OF = "Scores as of {date}."
NEWS_CHECKING = "Checking news..."
HOW_TO_BUY_ISIN_MISSING = "ISIN missing for {name}. Repair it in Settings."
# F-series (bug 4): honest empty state when the instrument has no registry row
# yet (fresh install / untracked) — never show the bare ticker as a title.
EXPLORE_NO_DETAILS = ("No details for {symbol} yet. Refresh market data in "
                      "Settings to load its name and ISIN.")
# H3.4: empty-query helper (distinct from the zero-match sentence).
SEARCH_HELPER = "Type a name, symbol, ISIN or theme to explore."
# H3.4: news display — cap the visible rows, expand the rest.
NEWS_EARLIER = "Earlier items ({n} more)"
# H3.4: one Diagnostics line when the sentiment model is absent.
SENTIMENT_UNAVAILABLE = "Sentiment model not available; news shown without sentiment."

# ── v10.7.2 (Part 2): the news pillar's honest absent line + diagnostic ───────
# When the heavy FinBERT stack contributes nothing on this user's universe, the
# system says so in one honest line and runs lighter (no torch import).
NEWS_PILLAR_ABSENT = ("News pillar: no data for your assets. Tactical score uses "
                      "market regime and price momentum only.")
NEWS_DOCTOR_HEADER = "News pillar diagnostic"
NEWS_DOCTOR_LINE = ("{symbol}: {total} items in 30 days, {long} long enough, "
                    "{model} scored by the model, {default} defaulted ({reason}).")
NEWS_DOCTOR_NONE = "{symbol}: no news in the last 30 days."
NEWS_DOCTOR_STATUS = "News pillar: {status} ({model} model-scored items in 30 days)."
NEWS_DOCTOR_ENABLED = "News pillar forced active."
NEWS_DOCTOR_REASON_SHORT = "text too short"
NEWS_DOCTOR_REASON_UNAVAILABLE = "model unavailable"
NEWS_DOCTOR_REASON_NOT_INVOKED = "scorer not invoked"

# ── v10.7.2 (Part 3): backup ──────────────────────────────────────────────────
BACKUP_WARNING_SECRETS = ("Including data/notify.toml (contains your bot token). "
                          "Keep this archive private.")
BACKUP_RESTORE_HINT = ("To restore: stop quant processes, unpack the archive over "
                       "your project folder, then run quant doctor.")
BACKUP_DONE = "Backup written: {path} ({size} MB)."
BACKUP_MEMBERS = "Members: {members}"
BACKUP_LAST = "Last backup: {date}"
BACKUP_OLD = "Last backup {n} days ago. Run quant backup."
BACKUP_NEVER = "No backup yet. Run quant backup."

# ── v10.7.2 (Part 4): setup ───────────────────────────────────────────────────
SETUP_HEADER = "Setup status"
SETUP_STEP_LINE = "{n}. {title}: {status}"
SETUP_DONE = "done"
SETUP_TODO = "not done"
SETUP_MARKET = "Market data"
SETUP_TIERS = "Tiers"
SETUP_SCHEDULE = "Schedule"
SETUP_NOTIFY = "Notifications"
SETUP_BACKUP = "Backup"
SETUP_CHECKLIST = "First-week checklist"
SETUP_MARKET_DONE = "last refresh {date}"
SETUP_MARKET_TODO = "no market data yet; press Refresh market data"
SETUP_TIERS_DONE = "{n} tiers"
SETUP_TIERS_TODO = "missing"
SETUP_SCHEDULE_DONE = "installed"
SETUP_SCHEDULE_TODO = "not installed; run quant schedule"
SETUP_NOTIFY_DONE = "configured"
SETUP_NOTIFY_TODO = "not configured; run quant notify-setup"
SETUP_BACKUP_DONE = "last backup {date}"
SETUP_BACKUP_TODO = "no backup yet; run quant backup"
SETUP_CHECKLIST_HINT = "see docs/first_week.md"
SETUP_SKIP = "skip"
SETUP_ACTION = "do it now"
# Automation buttons (v10.8.0, Phase 4): install the timer and set up Telegram
# in the app, not only on the command line.
BTN_INSTALL_TIMER = "Install the daily timer"
BTN_INSTALLING_TIMER = "Installing the timer"
BTN_TELEGRAM_SETUP = "Set up Telegram"
TELEGRAM_TOKEN_LABEL = "Bot token"
TELEGRAM_CHAT_LABEL = "Chat id"
TELEGRAM_HELP = ("Create a bot with BotFather, then get your chat id from a "
                 "user-info bot. The token is stored locally and never logged.")
BTN_SAVE_TELEGRAM = "Save and send a test"
TELEGRAM_SAVED = "Telegram saved; test message sent."
TELEGRAM_SAVED_UNTESTED = "Telegram saved; the test message failed. Check the token and chat id."
# H3.5: discovery-universe loop closure (Explore zero-match + Portfolio add).
DISCOVERY_NOT_TRACKED = ("{label} is in the discovery universe but not tracked. "
                         "Add it in My holdings to track it.")
NOT_TRACKED_LABEL = "{label} - not tracked yet"

# Portfolio (spec 2.3)
VALIDATION_UNIVERSE = ("{n} holdings are not in the universe yet; they will be added "
                       "on the next refresh.")
VALIDATION_UNIVERSE_ONE = ("1 holding is not in the universe yet; it will be "
                           "added on the next refresh.")
OUTCOME_ACTIONS = "Check complete. {n} actions on Overview."
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


def _days_in_month(year: int, month: int) -> int:
    """Days in a month (leap-aware). Keeps copy.py datetime-only (no calendar)."""
    if month == 2:
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        return 29 if leap else 28
    return 30 if month in (4, 6, 9, 11) else 31


def savings_plan_line(today: date, day: int) -> str:
    """Exact Today sentence for the savings-plan day (R8). Pure, no I/O.

    Invariants: day in 1..31; a day beyond the month's length clamps to that
    month's last day. Same day -> SAVINGS_TODAY; otherwise the countdown
    sentence. The date carries its weekday (audit rule: every rendered date
    uses fmt_weekday_date).
    """
    if day == today.day:
        return SAVINGS_TODAY
    y, m = today.year, today.month
    if day > today.day:
        target = date(y, m, min(day, _days_in_month(y, m)))
    else:
        m += 1
        if m > 12:
            m, y = 1, y + 1
        target = date(y, m, min(day, _days_in_month(y, m)))
    # v10.8.0 (2.5): a broker executes on the next trading day, so a weekend
    # target is shown as the following Monday.
    while target.weekday() >= 5:
        target = target + timedelta(days=1)
    days = (target - today).days
    return SAVINGS_COUNTDOWN.format(days=days, date=fmt_weekday_date(target))

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
    drift_frac: float | None = None,
    threshold: float | None = None,
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
    # H3.8 (M4): an over-threshold drift is actionable even when the time gate
    # delays the order — so "On track" for a 7.5-point overshoot is impossible.
    if drift_frac is not None and threshold is not None and abs(drift_frac) >= threshold:
        actionable = True
    if actionable and cooldown_until is not None:
        return STATUS_WAITING.format(date=fmt_date(cooldown_until))
    if rec.startswith("BUY"):
        return STATUS_ADD
    if rec.startswith("SELL"):
        return STATUS_TRIM
    if actionable and drift_frac is not None:
        # Over threshold with no explicit rec: use the drift sign.
        return STATUS_ADD if drift_frac < 0 else STATUS_TRIM
    return STATUS_ON_TRACK


# ── v10.7.0: Naming dictionary (Section 11, mandatory) ────────────────────────
# Old -> new, exact strings. The forbidden-token test scans rendered pages and
# the briefing for the OLD tokens; these constants are the new vocabulary.

# Tier display words (company name first, symbol small in parentheses elsewhere).
TIER_WORDS = {
    "FORTRESS": TIER_FORTRESS,
    "ALPHA": TIER_ALPHA,
    "SPECULATIVE": TIER_SPECULATIVE,
}

# Section titles (v10.7.0 page redesign).
SEC_YOUR_MONEY = "Your money"
SEC_STEPS = "Your steps this week"
SEC_NOT_THIS_WEEK = "Not this week"
# v10.8.0 (Phase 1, redesign 3.2): the ONE decision list, three groups.
DECISION_GROUP_INPUT = "Needs your input"
DECISION_GROUP_RECOMMENDED = "Recommended"
DECISION_GROUP_OPTIONAL = "Optional"
DECISION_EMPTY = "Nothing needs your input right now."
SEC_YOUR_ASSETS = "Your assets"
SEC_SAVINGS_PLAN = "Savings plan"
SEC_MARKET = "Market"
SEC_MONTHLY = "Monthly decision"
SEC_AUTOMATION = "Automation"
SEC_REPORT_HISTORY = "Report history"
SEC_SYSTEM_CHECK = "System check"
SEC_BROKER_REFERENCE = "Broker reference"
SEC_ALERTS = "Alerts"

# Page titles (v10.7.0 nav order).
PAGE_OVERVIEW = "Overview"
PAGE_MONTHLY = "Monthly decision"
PAGE_HOLDINGS = "My holdings"
PAGE_FIND = "Find investments"

# Statuses as verbs with EUR amounts.
STATUS_SELL_PART_AMOUNT = "Sell part (about {amount} EUR)"

# Verdicts (Block D, plain words).
VERDICT_KEEP = "keep, top up"
VERDICT_NOTHING = "do nothing"
VERDICT_TOO_SMALL = "do not sell, position too small"

# Savings plan: first mention carries the German word.
SPARPLAN_FIRST = "savings plan (Sparplan)"
SPARPLAN = "savings plan"

# Market regime line (Section 10.1 Block E). v10.7.3 (Part 1.2): the line lives
# ONLY inside the Market expander and names both parts explicitly.
MARKET_REGIME_LINE = ("Market is {label}, {confidence} confidence. This affects only "
                      "the Active part; the Long-term part is untouched.")
MARKET_REGIME_BEAR = ("Market is falling, {confidence} confidence. This affects only "
                      "the Active part; the Long-term part is untouched. New active "
                      "money goes to cash until the regime recovers. Long-term "
                      "savings plan continues.")

# Report history / system check.
REPORT_AS_OF = "Report as of {date}"

# Money plaques (Section 10.1 Block A).
INVESTED_LINE = ("Invested: {amount} EUR. Profit {pnl} EUR ({pct} percent, "
                 "deposits excluded). As of {date}.")
OPERATIONAL_CASH_LINE = ("Operational cash: {amount} EUR, as of {date}. For daily "
                         "life, not for investing. Earns {apy} percent per year. "
                         "Dividends land here.")
INCOME_LINE = ("Income last 12 months: dividends {dividends} EUR, cash yield about "
               "{cash_yield} EUR (estimate).")
ESTIMATED_LABEL = "estimated, as of {date}"
# v10.7.3 (Part 3.4): a position recorded since the last CSV sync.
ESTIMATED_PENDING = "estimated, pending sync"
# v10.7.3 (Part 8.2): the Overview caption linking the provenance section.
WHERE_NUMBERS = ("Where each number comes from: invested estimate, broker "
                 "statement, flows, scores. See the first-week guide.")

# Performance line (Section 7.3): deposits do not count as profit.
PERFORMANCE_LINE = ("Since {date}: {change} EUR. Of that: you added {added} EUR, "
                    "market moved {market} EUR. Return: {ret} percent.")

# Split lines (Section 10.2).
SPLIT_LINE = "{tier}: {value} EUR, {pct} percent of invested. Rule: {rule}. {status}."
SPLIT_RULE_LONG_TERM = "at least {min} percent"
SPLIT_RULE_ACTIVE = "at most {max} percent"
SPLIT_RULE_BETS = "at most {max} percent"
SPLIT_OK = "OK."

# Steps and silence (Section 10.5).
NOTHING_TO_DO_WEEK = "Nothing to do this week."
# v10.7.3 (Part 1.9): the "Not this week" empty state never repeats the first phrase.
NOTHING_REJECTED = "No considered actions were rejected this week."
NOTHING_URGENT = "Nothing urgent this week."
MONITORING_GAP = ("Monitoring gap: no runs for {n} days; conditions evaluated on "
                  "the latest data.")
SYNC_REMINDER = ("Last broker sync was {n} days ago. Export a fresh CSV from Trade "
                 "Republic when convenient.")

# Alerts (Section 2, exact phrasing template).
ALERT_ACTION_HEADER = "ACTION FOR TOMORROW"
ALERT_ACTION_LINE = "{name}: {action} about {amount} EUR."
ALERT_ACTION_FOOTER = ("Place the order tonight or tomorrow; it executes at market "
                       "open.")
ALERT_REASON = "Reason: {reason}."
ALERT_FEE = "Fee: {fee} EUR {side}. Reply in app: done / declined with reason."
ALERT_DETAILS_IN_APP = "details in app"

# Advice self-scoring ledger (Section 5).
ADVICE_RECORD = ("Advice record, last 12 months: {n} actions, {correct} correct, "
                 "{wrong} wrong.")

# In-app alert banner (Section 4.4).
ALERT_BANNER_TITLE = "Open actions"
BTN_ALERT_DONE = "Done"
BTN_ALERT_DECLINED = "Declined (reason)"
ALERT_RESOLVED = "Action marked {status}."

# Monthly decision (Section 8.1).
MONTHLY_TITLE = "Monthly decision - {month}"
MONTHLY_STATUS_NOT_APPROVED = "Status: not approved yet for this month."
MONTHLY_STATUS_APPROVED = "Status: approved on {date}."
MONTHLY_BUDGET_LABEL = "Your savings-plan budget this month (EUR)"
MONTHLY_BUDGET_PREFILL = "(prefilled from {month})"
MONTHLY_SPLIT_HEADER = "The system splits it:"
# v10.7.3 (Part 1.5): no raw keys. The long/active/bet leg names the tier word and
# the route; the cash leg is its own sentence with no fee suffix.
MONTHLY_LEG_LINE = "{amount} EUR to {name} ({symbol}), {kind}, via savings plan. Fee {fee} EUR."
MONTHLY_CASH_LEG_LINE = "Keep {amount} EUR in cash."
MONTHLY_LEG_REASON = "Reason: {reason}"
MONTHLY_NEW_IDEAS = "New ideas this month (optional, you may ignore all):"
MONTHLY_CANDIDATE_LINE = "{name} - {detail}"
BTN_APPROVE = "Approve"
BTN_CHANGE_SPLIT = "Change split"
MONTHLY_APPROVED = "Plan approved. Execute it in Trade Republic."
MONTHLY_ACTUALS_HEADER = "Enter what you bought"
BTN_SAVE_ACTUALS = "Save actuals"
MONTHLY_ACTUALS_SAVED = "Actuals saved. The math stays honest."
MONTHLY_EXECUTE_STEP = "Execute the approved plan in Trade Republic"
MONTHLY_ENTER_ACTUALS_STEP = ("Enter what you bought (one line), so the math "
                              "stays honest.")
MONTHLY_MAKE_DECISION_STEP = "Make this month's decision"

# ── v10.7.1: One advice pipeline (Part 1) ─────────────────────────────────────
ADVICE_SELL_PART = "Sell part"
ADVICE_BUY = "Buy"
ADVICE_TOP_UP = "Top up savings plan"
ADVICE_KEEP = "Keep, do nothing"
ADVICE_TO_CASH = "Move new active money to cash"
ADVICE_FROM_CASH = "cash"
ADVICE_FROM_SAVINGS = "savings plan"
ADVICE_FROM_POSITION = "position"
# Rejected notes ("Not this week"): the system showing its work.
REJECT_TOO_SMALL = "position {value} EUR; the 2 EUR fee makes any sale pointless"
REJECT_COOLDOWN = "cooldown until {date}; sells wait, buys do not"
REJECT_BELOW_MIN = "the move is below the 25 EUR minimum after rounding"
REJECT_FORTRESS = "long-term assets are never sold"
# v10.7.4 (Part 3.1): a system-wide lockdown pauses non-emergency sells.
REJECT_LOCKDOWN = "system-wide lockdown; non-emergency sells are paused"
FORTRESS_LEG_SUGGESTION = ("Consider raising the savings-plan leg for {name} from "
                           "{current} to {target} EUR per month; it is {pct} percent "
                           "of invested vs {target_pct} percent target.")
# v10.7.3 (Part 4.2): the ONE top-up sentence, shared by quant run and Overview.
# {label} is the "Name (TICKER)" form (label_for), so the ticker is never doubled.
STEP_TOP_UP = ("Top up the savings plan for {label}: it is {pct} percent "
               "of invested vs {target} percent target.")
CASH_REGIME_LINE = ("New active money goes to cash at {apy} percent until the market "
                    "regime recovers.")
# v10.7.4 (R2-R5): classification-grid copy.
# R2: a FORTRESS holding far OVER target gets a plan-change note, never a sell.
FORTRESS_OVER_LEG = ("Consider lowering or pausing the savings-plan leg for {label}; "
                     "it is {pct} percent of invested vs {target} percent target.")
# R3: an ALPHA holding under target with MEDIUM conviction is a considered buy.
REJECT_MEDIUM_CONVICTION = "conviction MEDIUM, needs HIGH for a buy"
# R9: a HIGH-conviction buy suppressed by the bear regime (regime override).
REJECT_BEAR_REGIME = ("{name} buy considered (conviction HIGH); suppressed by bear "
                      "regime; new active money goes to cash.")
# R4: the untouchable law applies to ACTIVE buys too (savings-plan legs exempt).
REJECT_BUY_TOO_SMALL = ("position below 100 EUR; the 1 EUR fee makes small buys "
                        "inefficient")
REJECT_BUY_BELOW_MIN = "the move is below the 10 EUR minimum after rounding"
# R1/Part 2.1: the expected alpha must clear the fee.
REJECT_FEE_HURDLE = ("expected alpha {alpha} bps on {amount} EUR does not clear "
                     "the {fee} EUR fee")
# R5: SPECULATIVE advisories are notes, never forced sells.
SPEC_TAKE_PROFIT = "consider taking profit; bets are double-or-nothing by design"
SPEC_CAP_VIOLATION = ("bets are {pct} percent of invested, above the 2 percent cap; "
                      "consider trimming or reclassifying")
# My holdings (Part 2).
SEC_HOW_SPLIT = "How your money is split"
SEC_IF_CASH = "If you need cash now"
SEC_TAX_LOSSES = "Losses you can use to lower tax"
SEC_QUICK_EVENTS = "Record a buy, sell, or dividend"
SPLIT_RULE_LONG = "at least {min} percent"
SPLIT_RULE_ACTIVE = "at most {max} percent"
SPLIT_RULE_BETS = "at most {max} percent"
SPLIT_OK = "OK."
SPLIT_OVER = "Over the rule. See Your steps this week."
ENTRY_PRICE_LINE = ("Entry price per share: {entry} EUR. Current price per share: "
                    "{current} EUR ({estimated}).")
SHARES_LINE = "Shares: {shares}, from broker sync on {date}."
TIER_LINE = "Tier: {tier}."
WHY_VERDICT_LINE = "Why this verdict: {why}"
TAX_LOSS_LINE = "{name}: {pnl} EUR unrealized loss. Selling it can reduce tax on gains."
TAX_LOSS_NONE = "No positions with unrealized losses."
QUICK_EVENT_SAVED = ("Recorded: {type} {amount} EUR {name} on {date}. Performance "
                     "math updated.")
QUICK_EVENT_CASH_NOTE = ("Daily-life spending from operational cash is not recorded "
                         "and not asked for.")
SAVINGS_DAY_LABEL = "Savings plan execution day of month"
SAVINGS_DAY_CAPTION = ("Used to date planned savings-plan flows and the enter-actuals "
                       "reminder.")
# Find investments (Part 3).
SEC_CAND_LONG = "Candidates for long-term"
SEC_CAND_ACTIVE = "Candidates for active"
SEC_CAND_BETS = "Candidates for small bets"
CAND_LONG_REASON = "Would be a savings-plan asset. Fee 0 EUR on buys."
CAND_ACTIVE_REASON = "Active idea: strong recent trend, may be sold when cash is needed."
CAND_ACTIVE_LIMIT = "Counts against the active limit: {used} percent of {max} percent used."
CAND_BETS_LIMIT = "Counts against the bets limit: {used} percent of {max} percent used."
CAND_EMPTY = "No candidates for {section} this month."

# ── v10.7.6 (Part 1): Buffett quality filter ──────────────────────────────────
SEC_CAND_BUFFETT = "Buffett candidates"
BUFFETT_EMPTY = "No candidates meet Buffett quality standards this month."
BUFFETT_SCORE_LINE = "Buffett score: {score} of 100."
BUFFETT_MOAT_LINE = "Moat: {moat}."
BUFFETT_REASON_LINE = "{reason}"
BUFFETT_CHECK_MET = "{check}: met"
BUFFETT_CHECK_NOT_MET = "{check}: not met"
BUFFETT_LONG_REASON = "Would be a long-term holding. Fee 0 EUR on savings-plan buys."
BUFFETT_MOAT_WIDE = "wide"
BUFFETT_MOAT_NARROW = "narrow"
BUFFETT_MOAT_NONE = "none"
BUFFETT_QUALITY_LINE = "Buffett quality: {score} of 100. {reason}"
BUFFETT_MOAT_HOLDING_LINE = "Moat: {moat}."

# ── v10.7.3 (Part 1): UI truth pass — exact new strings ───────────────────────
# Broker statement (editable) expander on My holdings.
SEC_BROKER_STATEMENT = "Broker statement (editable)"
BROKER_STATEMENT_CAPTION = ("This is what your broker reported at the last sync. "
                            "Edit only after exporting fresh values from Trade "
                            "Republic.")
BROKER_STATEMENT_AS_OF = "broker statement, as of {date}"
# Reconciliation check (v10.8.0, redesign 3.1): the broker's own total vs the
# sum of the entered positions. Catches the class of error where a position is
# missing or mistyped (e.g. 1,075.94 shown vs 852.23 entered).
RECONCILE_TOTAL_LABEL = "Total shown in Trade Republic (EUR)"
RECONCILE_TOTAL_CAPTION = ("Optional. Enter the total your broker shows. We compare "
                           "it with the sum of the positions above.")
RECONCILE_MATCH = "The positions add up to the broker total."
RECONCILE_MISMATCH = ("The positions add up to {entered}, but your broker shows "
                      "{broker}. A difference of {diff} usually means a missing or "
                      "mistyped position.")
# Verdicts table scores caption.
SCORES_CAPTION = ("Structure and Tactics are scores from 0 to 100. Structure is "
                  "fundamental quality; Tactics is timing.")
# Emergency liquidity hint when the amount is zero.
EMERGENCY_HINT = "Enter an amount to see the order in which positions would be sold."
# Monthly decision: pre-approval actuals are ad-hoc buys.
MONTHLY_ADHOC_NOTE = ("Recorded as an ad-hoc buy outside the monthly plan. Approve a "
                      "plan or ignore; the math stays honest.")
# Monthly decision: post-approve impact line.
MONTHLY_PLAN_SAVED = ("Plan saved. Planned flows dated {date}. Overview steps will "
                      "track execution.")
# Account section (v10.8.0, Phase 2): moved from My holdings to Settings.
SEC_ACCOUNT = "Account"
BTN_SAVE_ACCOUNT = "Save account"
SAVE_ACCOUNT_DONE = "Account saved."
# Overview income line (Block A).
INCOME_LINE = ("Income last 12 months: dividends {dividends} EUR from flows, cash "
               "yield about {cash_yield} EUR at {apy} percent.")
INCOME_NONE = "No dividends recorded yet; record them in My holdings when they arrive."
# Actuals confirmation states the consequence.
ACTUALS_CONSEQUENCE = ("Actuals saved. Estimated value of {name} is now {value} EUR; "
                       "the broker statement will confirm it at the next CSV sync.")
# Sync reminder mentions pending positions explicitly.
SYNC_REMINDER_PENDING = ("Last broker sync was {n} days ago. You have positions "
                         "recorded since then; export a fresh CSV from Trade "
                         "Republic when convenient.")
# v10.7.4 (R7): the reminder lists the pending positions by name.
SYNC_REMINDER_PENDING_NAMES = ("Last broker sync was {n} days ago. Positions "
                               "recorded since then: {names}. Export a fresh CSV "
                               "from Trade Republic when convenient.")
# Find investments funnel transparency.
FUNNEL_LINE = ("This month: {entered} symbols entered the funnel, {survived} survived "
               "liquidity and trend, {conviction} meet the conviction bar.")
FUNNEL_NEAR_MISSES = "Near misses"
FUNNEL_NEAR_MISS_LINE = "{name} ({symbol}): {detail}"
FUNNEL_NEAR_MISS_TACTICS = "Tactics {score:.0f}, needs {needs:.0f}"
FUNNEL_NEAR_MISS_STRUCTURE = "Structure {score:.0f}, needs {needs:.0f}"
# Candidate card "View analysis" expander.
CAND_VIEW_ANALYSIS = "View analysis"
CAND_ANALYSIS_STRUCTURE = "Structure {score:.0f} of 100."
CAND_ANALYSIS_TACTICS = "Tactics {score:.0f} of 100."
CAND_ANALYSIS_CONVICTION = "Conviction {conviction}."
CAND_ANALYSIS_REASON = "{reason}"
# Monthly "New ideas this month" candidate line (identical to Find investments).
MONTHLY_CANDIDATE_CARD = "{name} ({symbol}): {detail}"

# ── v10.7.6 (Part 2): Tax summary page ────────────────────────────────────────
SEC_TAX_SUMMARY = "Tax summary {year}"
TAX_YEAR_LABEL = "Tax year"
TAX_FILING_LABEL = "Filing status"
TAX_FILING_HELP = ("Affects the tax-free allowance: 1000 EUR single, "
                   "2000 EUR married.")
TAX_REALIZED_GAINS = "Realized gains"
TAX_DIVIDENDS = "Dividends"
TAX_TOTAL_INCOME = "Total capital income"
TAX_ALLOWANCE = "Tax-free allowance"
TAX_ALLOWANCE_HELP = ("The German Sparerpauschbetrag: 1000 EUR single, "
                      "2000 EUR married.")
TAX_ALLOWANCE_USED = "Allowance used"
TAX_ALLOWANCE_REMAINING = "Allowance remaining"
TAX_TAXABLE_INCOME = "Taxable income"
TAX_TAXABLE_HELP = "Income above the tax-free allowance."
TAX_ESTIMATED_TAX = "Estimated tax (rough estimate)"
TAX_ESTIMATED_HELP = ("Rough estimate at 26.375 percent (Abgeltungssteuer plus "
                      "solidarity surcharge). Excludes the partial exemption for "
                      "equity funds, the advance lump sum on accumulating funds, "
                      "church tax, and any allowance used at another broker.")
# v10.8.0 (4): the partial exemption is not applied when the type is unknown.
TAX_PARTIAL_EXEMPTION_NOTE = ("The partial exemption for equity funds is not "
                              "applied: the instrument type is not in the registry.")
SEC_TAX_HARVEST = "Losses you can use to lower tax"
TAX_HARVEST_COVERED = ("Your gains are covered by the tax-free allowance. You "
                       "have {remaining} remaining. No harvesting needed.")
TAX_HARVEST_NONE = "No positions with unrealized losses to harvest."
TAX_HARVEST_INTRO = ("Consider selling these positions before year-end to offset "
                     "gains and reduce taxes:")
TAX_HARVEST_TITLE = "{name} - tax savings {savings}"
TAX_HARVEST_LOSS = "Unrealized loss: {loss}"
TAX_HARVEST_SAVINGS = "Tax savings: {savings}"
TAX_HARVEST_REASON = "{reason}"
TAX_HARVEST_NOTE = ("Selling this position reduces your taxable income by "
                    "{loss}.")
SEC_TAX_RECORD = "Record a trade"
# v10.8.0 (Phase 2): one transaction form. The Tax page links to it instead of
# duplicating the record-trade form.
TAX_RECORD_LINK = ("Record buys, sells, and dividends once, on My holdings. "
                   "They feed this summary automatically.")
BTN_GO_RECORD_TRADE = "Go to My holdings"
SEC_TAX_EXPORT = "Export"
BTN_EXPORT_TAX = "Export tax report CSV"

# Forbidden tokens (Section 11): the OLD vocabulary. The copy test fails if any
# of these appears in rendered pages or the briefing.
FORBIDDEN_TOKENS = (
    "Trim",
    "On track",
    "Waiting until",
    "Current_Value_EUR",
    "Broker_PnL_EUR",
    "Tier balance",
    "Emergency liquidity",
    "Tax-loss harvesting",
    "SECTOR",
    "SATELLITE",
    # v10.7.3 (Part 1): the surviving old strings.
    "Review prepared",
    "Market trend:",
    "The app never guesses them",
    "Broker registry",
    "long_term",
    "cash (cash), cash",
)


def tier_word(tier: str) -> str:
    """Map a tier enum to its plain display word (Section 11)."""
    return TIER_WORDS.get(str(tier).upper(), "Active (may sell)")


# v10.7.3 (Part 1.5): the monthly leg kind word (no raw keys in the UI).
_MONTHLY_LEG_KINDS = {
    "long_term": TIER_FORTRESS,
    "active": TIER_ALPHA,
    "bet": TIER_SPECULATIVE,
}


def monthly_leg_kind(kind: str) -> str:
    """Map an allocator leg kind to its plain tier word."""
    return _MONTHLY_LEG_KINDS.get(str(kind), TIER_ALPHA)


# v10.7.6 (Part 1): the plain word for a Buffett moat estimate.
_MOAT_WORDS = {"wide": BUFFETT_MOAT_WIDE, "narrow": BUFFETT_MOAT_NARROW}


def moat_word(moat: str | None) -> str:
    """Map a Buffett moat estimate to its plain display word."""
    return _MOAT_WORDS.get(str(moat), BUFFETT_MOAT_NONE)
