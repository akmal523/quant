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

# v10.8.7 (A1): the ONE class-word map lives in quant.ui.terminology. These are
# re-exported here for the existing `C.CLASS_LABELS` callers.
from quant.ui import palette as _palette
from quant.ui import terminology as _term

CLASS_SPECULATIVE = _term.CLASS_SPECULATIVE
CLASS_LABELS = _term.CLASS_LABELS

# ── Page titles (P12) ─────────────────────────────────────────────────────────
PAGE_SETTINGS = "Settings"

# ── Section titles ────────────────────────────────────────────────────────────
SEC_WHAT_TO_DO = "What to do today"

# ── Empty and status states (exact strings, spec 3.2) ─────────────────────────
# H3.4: exact zero-match sentence (distinct from the empty-query helper).
EMPTY_NO_MATCHES = ("No instrument matches {query}. Try a company or fund name, "
                    "a symbol, an ISIN, or a theme such as gold or defence.")

# ── Action cards (spec 3.2) ───────────────────────────────────────────────────
ACTION_BLOCKED_MANUAL = ("ISIN missing for {symbol}. Not found automatically. Add a verified "
                         "row from your broker app or the fund factsheet, then press Repair "
                         "registry again.")

# ── v10.6.2 / v10.7.0: Three-tier dashboard, cash, tax-loss ──────────────────
TIER_FORTRESS = "Long-term (never sell)"
TIER_ALPHA = "Active (may sell)"
# v10.8.7 (A1): the canonical word is "Speculative"; the descriptive form is
# "Speculative (high risk)". "Small bets" is banned by a lint test.
TIER_SPECULATIVE = "Speculative (high risk)"

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
ERROR_ALREADY_RUNNING = "A review is already running in another tab."
# H3.8 (M8): own-session copy is operation-agnostic (refresh or review).
ERROR_RUNNING = "An operation is already running."

# ── Helper texts (spec 3.2) ───────────────────────────────────────────────────
# A5: the add-holding affordance states what it does and what to do next.
PLACEHOLDER_ADD_HOLDING = "Type a name, symbol or ISIN to add a holding"
HELP_ADD_ROW = "Now fill value and profit or loss from your broker."

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
# Column-name -> human header (P2: no column names in the UI).
# v10.8.8 (R-P1): the editor grid + entry-check headers ARE the column contract
# (see quant/ui/columns.py). Every header is non-empty; money columns carry their
# currency. The editor grid is exactly Name | Instrument | Avg cost / share
# (paid ccy) | Value (EUR) | Profit (EUR) | Plan (EUR/m); no remove column.
COLUMN_HEADERS = {
    "Name": "Name",
    "Symbol": "Instrument",
    # v10.8.3: the fee-inclusive average cost basis, in plain English.
    "Avg_Entry_Price": "Avg cost / share (paid ccy)",
    "Current_Value_EUR": "Value (EUR)",
    "Broker_PnL_EUR": "Profit (EUR)",
    "Plan_EUR_month": "Plan (EUR/m)",
    "Remove": "Remove",
    "Current_Weight": "Now / should be (of invested)",
    "Target_Weight": "Target",
    "Drift": "Difference from target",
    "Signal": "Status",
}
# v10.8.8 (R-P1/A2): the read-only Entry-check table headers. The identity
# column is "Instrument" (the same broker ticker as the grid), so the two tables
# join visually on one key. This is the ONLY place live and typed meet.
ENTRY_CHECK_HEADERS = ("Instrument", "Shares (derived)", "Typed value (EUR)",
                       "Live value (EUR)", "Delta %")
ENTRY_CHECK_CAPTION = "Live values as of {date}; shares derived from your entries"
ENTRY_CHECK_EMPTY = "No holdings to check yet."

# v10.8.8 (R-P2): the Holdings Analysis drawer content contract. Fixed section
# order; each section has an explicit empty state (never silence or a dash).
DRAWER_PRICE = "Price"
DRAWER_SCORE = "Score"
DRAWER_NEWS = "News"
DRAWER_SEC = "SEC filings"
DRAWER_ADVICE = "Open advice"
DRAWER_NO_PRICE = "no price history yet - {reason}"
DRAWER_NO_NEWS = "no news fetched yet - {reason}"
DRAWER_NO_SEC = "no SEC filings fetched yet - {reason}"
DRAWER_NO_ADVICE = "no open advice"
DRAWER_ALERT_LINE = "{n}. {reason}"
# v10.8.8 (R-P2): the derived Holdings Analysis cells.
HOLDINGS_REASON_HEADER = "Reason"
HOLDINGS_ALERTS_HEADER = "Alerts"

# ── Formatting helpers (spec 3.3) ─────────────────────────────────────────────
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


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


# ── v10.8.8 (R-P3): the ONE delta badge ───────────────────────────────────────
# icon/color from the SIGN OF EUR; text with explicit rounding rules. "-0.0%" is
# unrepresentable. Every delta display in the product calls this helper.
_DELTA_UP = "\u25b2"     # up triangle
_DELTA_DOWN = "\u25bc"   # down triangle
_DELTA_FLAT = "\u2022"   # bullet


def delta_badge(eur, pct) -> tuple[str, str, str]:
    """The ONE delta badge: ``(icon, color, text)`` from the sign of ``eur`` (R-P3).

    icon/color: up/green, down/red, dot/neutral when ``|eur| < 0.5``.
    text: ``"<eur:+.0f> EUR (<pct:+.1f>%)"``. A non-zero value that rounds to zero
    is written as a signed "less than" instead: ``"-<1 EUR"`` / ``"+<0.1%"`` (the
    pct sign follows ``eur`` when ``eur != 0``). Exact zero is ``"+0 EUR (+0.0%)"``,
    so ``"-0 EUR"`` and ``"-0.0%"`` are unrepresentable.
    """
    e = 0.0 if _is_missing(eur) else float(eur)
    p = 0.0 if _is_missing(pct) else float(pct)
    if e >= 0.5:
        icon, color = _DELTA_UP, _palette.GAIN
    elif e <= -0.5:
        icon, color = _DELTA_DOWN, _palette.LOSS
    else:
        icon, color = _DELTA_FLAT, _palette.TEXT_MUTED
    whole = int(abs(e) + 0.5)  # half away from zero, matching the icon boundary
    if whole == 0:
        eur_txt = "+0" if e == 0 else ("-<1" if e < 0 else "+<1")
    else:
        eur_txt = f"{'-' if e < 0 else '+'}{whole}"
    if round(p, 1) == 0.0:
        sign = e if e != 0 else p
        pct_txt = "+0.0%" if sign == 0 else ("-<0.1%" if sign < 0 else "+<0.1%")
    else:
        pct_txt = f"{p:+.1f}%"
    return icon, color, f"{eur_txt} EUR ({pct_txt})"


def class_word(instrument_class: str) -> str:
    """Map an instrument_class enum to its human word."""
    return CLASS_WORDS.get(str(instrument_class).upper(), "Asset")


def regime_word(regime: str) -> str:
    """Map a regime enum to its human word."""
    return REGIME_WORDS.get(str(regime).lower(), "mixed")


# ── v10.5.3 (spec v4): state machines, charts, feedback, calendar ─────────────


# Charts (spec 3)
CHART_BUILDING = "The value chart builds up after a few reviews."
LABEL_RANGE = "Range"
LABEL_VIEW = "View"
VALUE = "Value"
# v10.7.3 (Part 7.2): Growth is an explicit opt-in.
GROWTH_VIEW = "Growth view (normalized to 100)"
LABEL_BENCHMARK = "Compare to MSCI World (IWDA.AS)"
BENCHMARK_SYMBOL = "IWDA.AS"


# ── v10.7.2 (Part 2): the news pillar's honest absent line + diagnostic ───────
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
BACKUP_RESTORE_HINT = ("To restore: close the app, unpack the archive into your "
                       "data folder (shown in Settings), then run the doctor.")
BACKUP_DONE = "Backup written: {path} ({size} MB)."
BACKUP_MEMBERS = "Members: {members}"
BACKUP_LAST = "Last backup: {date}"
# v10.8.10: point at the Settings button; the bare `quant` command is not on PATH
# after ./install.sh (the venv is never activated).
BACKUP_OLD = "Last backup {n} days ago. Use Back up now in Settings."
BACKUP_NEVER = "No backup yet. Use Back up now in Settings."

NOT_TRACKED_LABEL = "{label} - not tracked yet"


def _days_in_month(year: int, month: int) -> int:
    """Days in a month (leap-aware). Keeps copy.py datetime-only (no calendar)."""
    if month == 2:
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        return 29 if leap else 28
    return 30 if month in (4, 6, 9, 11) else 31


def status_word(universe_status: str) -> str:
    """Map a universe_status enum to its human word."""
    return STATUS_WORDS.get(str(universe_status).upper(), "Watching")


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

# v10.8.0 (Phase 1, redesign 3.2): the ONE decision list, three groups.
NEEDS_ATTENTION_ISIN = "ISIN missing for {symbol}. Repair it in Settings."
DECISION_GROUP_INPUT = "Needs your input"
DECISION_GROUP_RECOMMENDED = "Recommended"
DECISION_GROUP_OPTIONAL = "Optional"
SEC_MARKET = "Market"

# v10.8.2 (B2): the invest block on Portfolio. One amount, one cadence, a
# suggestion only; the app never controls the broker's plans.
SEC_INVEST = "How much do you want to invest?"
INVEST_AMOUNT_LABEL = "Amount (EUR)"
# v10.8.2 (B6): a buy that matches the saved plan is labeled a savings-plan buy.
DIFF_BOUGHT_SAVINGS = "Bought (savings plan)"
# v10.8.3: the timed ONE list (Today / This month, Buy / Sell, by class).
HORIZON_TODAY = "Today"
HORIZON_MONTH = "This month"
DIRECTION_BUY = "Buy"
DIRECTION_SELL = "Sell"
WHAT_TO_DO_EMPTY = "All classes are in balance. Prices are up to date."
# v10.8.3: the class-based invest block.
INVEST_CLASS_HEADER = "Class"
INVEST_NOW_HEADER = "Now"
INVEST_TARGET_HEADER = "Target"
# v10.8.9 (R-F1): a monetary column carries its currency in the header.
INVEST_SUGGESTED_HEADER = "Suggested (EUR)"
INVEST_AMOUNT_HEADER = "Amount (EUR)"
INVEST_HOW_HEADER = "How"
INVEST_STRATEGY_LABEL = "Strategy"
INVEST_NOTHING = "Nothing to buy: every class is already at or above its target."


# v10.8.7 (R-C1): the neutral remainder line. No accounting entity, no doubled
# currency word, correct singular/plural.
def invest_remainder_line(amount_eur) -> str:
    """The one neutral remainder line: '<X> EUR remain(s) at the broker (not tracked).'"""
    try:
        amount = float(amount_eur)
    except (TypeError, ValueError):
        amount = 0.0
    verb = "remains" if abs(amount - 1.0) < 0.005 else "remain"
    return f"{fmt_eur(amount)} {verb} at the broker (not tracked)."
COST_BASIS_HINT = ("The average you paid per share, including fees "
                   "(Trade Republic calls it Buy In).")
# v10.8.2: the news-sentiment stack is an optional extra.
MARKET_NEWS_OFF = ("News sentiment is off. Install the 'news' extra to enable it; "
                   "everything else works without it.")
SEC_ALERTS = "Alerts"


NOTHING_URGENT = "Nothing urgent this week."
MONITORING_GAP = ("Monitoring gap: no runs for {n} days; conditions evaluated on "
                  "the latest data.")

# Alerts (Section 2, exact phrasing template).
ALERT_ACTION_HEADER = "ACTION FOR TOMORROW"
ALERT_ACTION_LINE = "{name}: {action} about {amount} EUR."
ALERT_ACTION_FOOTER = ("Place the order tonight or tomorrow; it executes at market "
                       "open.")
ALERT_REASON = "Reason: {reason}."
ALERT_FEE = "Fee: {fee} EUR {side}. Reply in app: done / declined with reason."
ALERT_DETAILS_IN_APP = "details in app"


# ── v10.7.1: One advice pipeline (Part 1) ─────────────────────────────────────
ADVICE_SELL_PART = "Sell part"
ADVICE_BUY = "Buy"
ADVICE_TOP_UP = "Top up savings plan"
ADVICE_KEEP = "Keep, do nothing"
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
# v10.7.3 (Part 4.2): the ONE top-up sentence, shared by quant run and Overview.
# {label} is the "Name (TICKER)" form (label_for), so the ticker is never doubled.
STEP_TOP_UP = ("Top up the savings plan for {label}: it is {pct} percent "
               "of invested vs {target} percent target.")
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

# ── v10.7.3 (Part 1): UI truth pass — exact new strings ───────────────────────

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


# ── v10.8.4 (T4/T5/T6/T9): workflow repair copy ───────────────────────────────
# T4: the savings-plan reminder is a task, cleared when the flow lands or dismissed.
SAVINGS_PLAN_REMINDER = ("Your savings plan took ~{amount} EUR on {date}. Reflect "
                         "it in Update holdings, or mark as done.")
BTN_REMINDER_DONE = "Done / Nothing changed"
SEC_TASKS = "Tasks"
# T5: advice is a stateful object; a repeated item shows how long it has been open.
ADVICE_OPEN_SINCE = "open since {date}"
SEC_ADVICE_LOG = "Advice log"
ADVICE_LOG_EMPTY = "No advice recorded yet."
# T2: explicit empty state instead of a literal None (R2).
NOT_AVAILABLE = "Not available - {reason}."
# R3: freshness lines.
AS_OF_DATE = "as of {date}"
# v10.8.7 (R-C1): the cash row / class / target constants are removed. Uninvested
# cash is not an entity; there is no cash row, line, target or weight anywhere.

# v10.8.7 (T2/A3): the on-demand analysis hint (the first run of the day fetches).
RUN_ANALYSIS_HINT = "The first analysis of the day can take a few minutes."
# T7/T6: pipeline health.
SEC_PIPELINE_HEALTH = "Pipeline health"
BTN_RUN_ANALYSIS = "Run analysis"
# T6: the merged Research surface.
TAB_RESEARCH = "Research"
RESEARCH_SEARCH_LABEL = "Search"
RESEARCH_STAGE_FILTER = "Stage"
# v10.8.9 (R-F1): the Research funnel target set. Component detail (F-/T-/
# sentiment/stewardship) lives in the inspect drawer, not the table; the table
# keeps the composite + confidence. "Now / Target" is held-only ("-" for watched).
RESEARCH_HEADERS = ("Symbol", "Class", "Stage", "Score", "Confidence", "News",
                    "Now / Target")
# v10.8.6 (A1): one lazy per-tab drawer; a symbol is inspected only when picked.
INSPECT_SYMBOL = "Inspect symbol"
INSPECT_HOLDING = "Inspect holding"
INSPECT_PLACEHOLDER = "Choose a row..."
# T8: the invest-amount flow.
BTN_ADD_TO_LIST = "Add to today's list"
INVEST_PRESET_HINT = "Preset amounts"
STRATEGY_CONFIRM_LABEL = "This changes your class targets. Apply the new strategy?"
BTN_STRATEGY_APPLY = "Apply strategy"
