"""
render.py — Page renderers for the Quant-AI multipage app (v10.5.3, R2).

Intent: st.Page needs FILE-based pages so AppTest.switch_page can drive them, so
the four page renderers live here and the thin scripts under quant/pages/ call
them. dashboard.py is the navigation entry only.

State/View: render functions reflect the artifacts read via the five helpers in
quant.reporting.artifacts. All user-facing strings come from quant.ui.copy.

Dependencies: streamlit, pandas, plotly, quant.*, quant.ui.copy, quant.ui.runner.
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from quant import paths
import os
import pandas as pd
import streamlit as st

from quant import __version__
from quant.config import STALE_DATA_DAYS, RISK_PROFILES
from quant.data.database import read_only_connection
from quant.execution.taxonomy import (
    resolve_broker, classify_instrument, get_structure,
    INVERSE_STRUCTURE, LEVERAGED_STRUCTURE,
)
from quant.portfolio.account import load_account, save_account, AccountState
from quant.portfolio.editor import validate_positions, save_portfolio
from quant.data.news import load_news
from quant.portfolio.cash_rate import current_cash_apy, current_rate
from quant.reporting.artifacts import (
    latest_review, read_actions, read_history, read_regime, read_scores,
)
from quant.ui import copy as C
from quant.ui import runner
from quant.ui.search import label_for, load_index, search

_EDIT_COLS = ["Symbol", "Avg_Entry_Price", "Current_Value_EUR", "Broker_PnL_EUR"]
_COLUMN_CONFIG = {
    "Symbol": st.column_config.TextColumn(C.COLUMN_HEADERS["Symbol"]),
    "Avg_Entry_Price": st.column_config.NumberColumn(
        C.COLUMN_HEADERS["Avg_Entry_Price"], format="%.2f"),
    "Current_Value_EUR": st.column_config.NumberColumn(
        C.COLUMN_HEADERS["Current_Value_EUR"], format="%.2f"),
    "Broker_PnL_EUR": st.column_config.NumberColumn(
        C.COLUMN_HEADERS["Broker_PnL_EUR"], format="%.2f"),
}


# ── Read-only helpers (spec 4.1: short-lived, always closed) ──────────────────

def q(sql: str, params=None) -> pd.DataFrame:
    """Run a read-only query. Returns an empty frame on any failure."""
    try:
        with read_only_connection() as conn:
            return conn.execute(sql, params or []).df()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def load_portfolio() -> pd.DataFrame:
    try:
        from quant.portfolio.portfolio import load_portfolio as _lp
        return _lp(paths.DATA_PORTFOLIO)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def latest_bar_date() -> str:
    df = q("SELECT MAX(Date) AS d FROM market_history")
    if df.empty or df["d"].iloc[0] is None:
        return ""
    return str(df["d"].iloc[0])


# ── Shared renderers ──────────────────────────────────────────────────────────

def render_action_cards(holdings: list[dict]) -> None:
    """Render action cards from the canonical action list (spec 2.1 S2)."""
    cards = [h for h in holdings if h.get("action")]
    if not cards:
        st.info(C.EMPTY_NOTHING_TO_DO)
        return
    for a in cards:
        if a.get("blocked"):
            continue
        target = (a.get("target_weight") or "").rstrip("%") or "?"
        pct = abs(float(str(a.get("drift", "0")).rstrip("%") or 0))
        name = a.get("name") or a["symbol"]
        if a["action"] == "BUY MORE":
            st.write(C.ACTION_ADD.format(
                amount=f"{a['amount_eur']:.0f}", symbol=a["symbol"],
                name=name, pct=f"{pct:.0f}", target=target))
        else:
            st.write(C.ACTION_SELL.format(
                amount=f"{a['amount_eur']:.0f}", symbol=a["symbol"],
                pct=f"{pct:.0f}", target=target))


# ── Page: Today (P4) ──────────────────────────────────────────────────────────

def page_today() -> None:
    st.title(C.PAGE_TODAY)

    portfolio = load_portfolio()
    history = read_history()
    review = latest_review()
    has_review = bool(review)

    # S0 (spec 2.1): no review yet -> guidance card, no header, no trend line.
    if not has_review and history.empty:
        st.info(C.GUIDE_NO_REVIEW)
    elif not has_review:
        # S4: a prior review exists but the latest run produced no metrics.
        last_ts = history.iloc[-1]["review_ts"]
        st.write(C.HEADER_REVIEW.format(date=C.fmt_date(latest_bar_date()),
                                        prepared=C.fmt_ts(last_ts)))
        st.warning(C.LAST_REVIEW_FAILED)
    else:
        prepared = C.fmt_ts(review.get("review_ts"))
        bar = C.fmt_date(review.get("latest_bar") or latest_bar_date())
        st.write(C.HEADER_REVIEW.format(date=bar, prepared=prepared))
        reg = read_regime()
        if reg.get("state") == "estimated":
            st.write(C.MARKET_TREND.format(label=reg.get("label"),
                                           confidence=reg.get("confidence")))
        elif reg.get("state") == "failed":
            st.write(C.MARKET_TREND_FAILED)
        else:
            st.write(C.MARKET_TREND_INSUFFICIENT)

    holdings = read_actions()

    # 2. Portfolio value chart (spec 3.1).
    st.subheader(C.SEC_PORTFOLIO_VALUE)
    _render_value_chart(history, holdings)

    # 3. Where your money is (donut). Categorical blue/gray palette only;
    #    semantic colors never encode composition (A4). Percent labels only for
    #    slices >= 5 percent; every slice appears in the legend with name+percent.
    st.subheader(C.SEC_WHERE_MONEY)
    account = load_account()
    if not portfolio.empty:
        import plotly.graph_objects as go
        values = list(portfolio["Amount_EUR"])
        labels = list(portfolio["Symbol"])
        if account.cash_is_set and account.cash_eur:
            values.append(account.cash_eur)
            labels.append("Cash")
        total = sum(values) or 1.0
        pcts = [v / total * 100 for v in values]
        legend_labels = [f"{lbl} {p:.0f}%" for lbl, p in zip(labels, pcts)]
        slice_text = [f"{p:.0f}%" if p >= 5 else "" for p in pcts]
        palette = ["#1F3B73", "#3B5C99", "#5B83BF", "#8FA9CF", "#B8C4D9",
                   "#6B7280", "#8A8F99", "#A7ADB8"]
        colors = [palette[i % len(palette)] for i in range(len(values))]
        fig = go.Figure(go.Pie(
            labels=legend_labels, values=values, hole=0.55,
            text=slice_text, textinfo="text", sort=False,
            marker=dict(colors=colors)))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0),
                          showlegend=True, legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")

    # 4. Holdings table. Status comes from the audit actions so the table and the
    #    cards can never disagree (A3). S0 -> every row "Not reviewed yet".
    table_rows = []
    if holdings:
        for h in holdings:
            table_rows.append({
                "Holding": label_for(h.get("name") or h["symbol"], h["symbol"]),
                "Value": C.fmt_eur(h["value_eur"]),
                "Share vs target": f"{h['current_weight']} / {h['target_weight']}",
                "Status": h["status"],
            })
    elif not portfolio.empty:
        for sym in portfolio["Symbol"].astype(str):
            table_rows.append({"Holding": sym, "Value": "",
                               "Share vs target": "", "Status": C.STATUS_NOT_REVIEWED})
    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)

    # 5. What to do today (S0/S1: nothing, S2: cards).
    if has_review:
        st.subheader(C.SEC_WHAT_TO_DO)
        render_action_cards(holdings)
        _render_suppression_footnotes(holdings)

    # 6. Needs attention first (sentence only; Repair lives in Settings Health).
    blockers = [h for h in holdings if h["blocked"]]
    if blockers:
        st.subheader(C.SEC_NEEDS_ATTENTION)
        for b in blockers:
            st.warning(C.NEEDS_ATTENTION_ISIN.format(symbol=b["symbol"]))


# ── Page: Portfolio (P5) ──────────────────────────────────────────────────────

def page_portfolio() -> None:
    st.title(C.PAGE_PORTFOLIO)
    st.write(C.HELP_BROKER_VALUES)

    # Autocomplete add-row (A5): the input says what it does; selecting a match
    # appends an empty-value row and shows one helper line. No silent add.
    query = st.text_input(
        C.PLACEHOLDER_ADD_HOLDING, key="add_q",
        placeholder=C.PLACEHOLDER_ADD_HOLDING, label_visibility="collapsed",
    )
    if query:
        results = search(load_index(), query, 10)
        if results:
            labels = [r["label"] for r in results]
            sym_by_label = {r["label"]: r["symbol"] for r in results}

            def _add_selected() -> None:
                sym = sym_by_label.get(st.session_state.get("add_choice"))
                if not sym:
                    return
                st.session_state.setdefault("_extra", []).append({
                    "Symbol": sym, "Avg_Entry_Price": 0.0,
                    "Current_Value_EUR": 0.0, "Broker_PnL_EUR": 0.0,
                })
                st.session_state["_just_added"] = True

            st.selectbox("Matches", labels, key="add_choice", on_change=_add_selected)
            if st.session_state.get("_just_added"):
                st.caption(C.HELP_ADD_ROW)
        else:
            st.caption(C.EMPTY_NO_MATCHES.format(query=query))

    # Holdings editor.
    portfolio = load_portfolio()
    base = portfolio[_EDIT_COLS] if not portfolio.empty and \
        set(_EDIT_COLS).issubset(portfolio.columns) else pd.DataFrame(columns=_EDIT_COLS)
    extra = st.session_state.get("_extra", [])
    if extra:
        base = pd.concat([base, pd.DataFrame(extra)], ignore_index=True)
    edited = st.data_editor(
        base, num_rows="dynamic", column_config=_COLUMN_CONFIG,
        width="stretch", key="holdings",
    )

    # Account block.
    st.subheader("Account")
    account = load_account()
    cash_val = account.cash_eur if account.cash_is_set else 0.0
    cash = st.number_input("Cash (EUR)", min_value=0.0, value=float(cash_val), step=10.0)
    profile = st.radio(
        "Risk profile", list(RISK_PROFILES.keys()),
        index=list(RISK_PROFILES.keys()).index(account.risk_profile),
        captions=[C.HELP_PROFILE_CONSERVATIVE, C.HELP_PROFILE_BALANCED,
                  C.HELP_PROFILE_AGGRESSIVE],
        format_func=lambda p: p.capitalize(),
    )
    rate = current_rate()
    st.caption(C.HELP_CASH_APY.format(
        apy=f"{rate.apy * 100:g}", date=C.fmt_date(rate.effective_date)))

    # Buttons.
    registry = q("SELECT symbol FROM asset_registry")
    universe = set(registry["symbol"].astype(str)) if not registry.empty else set()

    def _save_inputs() -> None:
        cleaned, warnings, errors = validate_positions(edited, universe)
        if errors:
            for e in errors:
                st.error(e)
            return
        save_portfolio(cleaned, paths.DATA_PORTFOLIO)
        save_account(AccountState(account.base_currency, float(cash), profile, True))
        st.session_state["_extra"] = []
        st.session_state["_saved_unknown"] = len(warnings)

    _run_operation(C.BTN_SAVE_AND_REVIEW, C.BTN_REVIEWING, runner.SAVE_AND_REVIEW,
                   "save_review", primary=True, on_click=_save_inputs)
    # Open Today only after a successful review; advice is read from the run
    # artifact on Today, never from session state (fixes the dead-button symptom).
    if st.session_state.get("_review_ok"):
        if st.button(C.BTN_OPEN_TODAY, key="open_today"):
            st.switch_page("pages/today.py")
    if st.button(C.BTN_SAVE_ONLY, width="stretch"):
        _save_inputs()
        st.success(C.SAVE_ONLY_DONE)
    unknown = st.session_state.get("_saved_unknown")
    if unknown:
        tpl = C.VALIDATION_UNIVERSE_ONE if unknown == 1 else C.VALIDATION_UNIVERSE
        st.caption(tpl.format(n=unknown))

    # Broker registry (read-only, collapsed).
    with st.expander("Broker registry"):
        broker = pd.read_csv(paths.DATA_BROKER_REGISTRY) if os.path.exists(
            paths.DATA_BROKER_REGISTRY) else pd.DataFrame()
        # isin_source is internal provenance; never shown (decision memo 1.4).
        if "isin_source" in broker.columns:
            broker = broker.drop(columns=["isin_source"])
        st.dataframe(broker, width="stretch", hide_index=True)


def _render_outcome(res, command: str) -> None:
    """One outcome sentence; failure = plain sentence + collapsed View log (4.2/4.3)."""
    if res.status == "busy":
        st.warning(res.message)
        return
    if res.ok:
        if command == runner.SAVE_AND_REVIEW:
            n = len([h for h in read_actions() if h.get("action") and not h.get("blocked")])
            st.success(C.OUTCOME_ACTIONS.format(n=n) if n else C.OUTCOME_NOTHING)
        elif command == runner.REFRESH:
            instruments = q("SELECT COUNT(DISTINCT Symbol) AS n FROM market_history")
            n = int(instruments["n"].iloc[0]) if not instruments.empty else 0
            st.success(C.OUTCOME_REFRESH.format(
                n=n, date=C.fmt_date(latest_bar_date()), secs=f"{res.duration_s:.0f}"))
        else:
            st.success(res.message or C.OUTCOME_REFRESH_DONE)
        return
    st.error(res.message or C.ERROR_REFRESH_FAILED)
    if res.log_path:
        with st.expander(C.VIEW_LOG):
            st.code(_read_log(res.log_path) or "(no output)")


def _run_operation(trigger_label, running_label, command, key,
                   primary=False, on_click=None, runner_fn=None):
    """Verb-ing disabled button + dedicated progress container, cleared on completion.

    Two-phase: the trigger press sets _running and reruns; the running rerun shows
    the disabled verb-ing label, runs under the mutex/heartbeat, clears the
    progress container, and renders exactly one outcome line.
    """
    use = runner_fn or runner.run
    if st.session_state.get("_running"):
        st.button(running_label, key=key, disabled=True, width="stretch",
                  type="primary" if primary else "secondary")
        prog = st.empty()
        prog.progress(0)
        res = use() if runner_fn is not None else use(command)
        prog.empty()
        st.session_state["_running"] = False
        st.session_state["_review_ok"] = res.ok and command == runner.SAVE_AND_REVIEW
        _render_outcome(res, command)
        return res
    if st.button(trigger_label, key=key, width="stretch",
                 type="primary" if primary else "secondary"):
        if on_click:
            on_click()
        st.session_state["_running"] = True
        st.rerun()
    return None


def _read_log(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return "".join(f.readlines()[-50:])
    except Exception:  # noqa: BLE001
        return ""


# ── Page: Explore (P6) ────────────────────────────────────────────────────────

def page_explore() -> None:
    st.title(C.PAGE_EXPLORE)

    query = st.text_input("Search a name, symbol or ISIN", key="ex_q")
    options = [r["label"] for r in search(load_index(), query, 10)] if query else []
    if not options:
        st.caption("Type a name, symbol or ISIN to explore.")
        return
    choice = st.selectbox("Matches", options, key="ex_choice")
    symbol = next(r["symbol"] for r in search(load_index(), query, 10)
                  if r["label"] == choice)

    broker = resolve_broker(symbol)
    reg = q("SELECT COALESCE(display_name, name) AS nm, instrument_class, currency, "
            "isin FROM asset_registry WHERE symbol = ?", [symbol])
    name = str(reg["nm"].iloc[0]) if not reg.empty and reg["nm"].iloc[0] else symbol
    cls = str(reg["instrument_class"].iloc[0]) if not reg.empty else \
        classify_instrument(symbol)

    st.subheader(name)
    st.caption(symbol)
    _subtitle = " · ".join(
        p for p in (C.class_word(cls), broker.get("currency", ""), broker.get("isin", "")) if p
    )
    if _subtitle:
        st.caption(_subtitle)
    structure = get_structure(symbol)
    if structure in (INVERSE_STRUCTURE, LEVERAGED_STRUCTURE):
        st.warning("This product is leveraged or inverse. It can lose value quickly.")

    # Price chart (full width).
    market = q("SELECT Date, Close, Volume FROM market_history WHERE Symbol = ? "
               "ORDER BY Date ASC", [symbol])
    if market.empty:
        st.info(C.CHART_NO_HISTORY.format(name=name))
    else:
        import plotly.graph_objects as go
        close = market["Close"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=market["Date"], y=close, name="Price"))
        fig.add_trace(go.Scatter(x=market["Date"], y=close.rolling(200).mean(),
                                 name="200-day average"))
        band = 2.0 * close.pct_change().ewm(span=20).std() * close
        fig.add_trace(go.Scatter(x=market["Date"], y=close + band, name="Upper band",
                                 line=dict(color="rgba(0,0,0,0)")))
        fig.add_trace(go.Scatter(x=market["Date"], y=close - band, name="Lower band",
                                 fill="tonexty", line=dict(color="rgba(0,0,0,0)")))
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, width="stretch")

    # Why these scores.
    st.subheader(C.SEC_WHY_SCORES)
    review = latest_review()
    review_date = C.fmt_ts(review.get("review_ts")) if review.get("review_ts") else ""
    if review_date:
        st.caption(f"From the review of {review_date}")
    sc = read_scores(symbol)
    has_scores = sc.get("structural_grade") is not None
    if has_scores:
        for label, key in (("Quality score", "structural_grade"),
                           ("Trend score", "tactical_grade"),
                           ("Overall score", "active_score")):
            val = float(sc.get(key) or 0)
            st.write(f"{label}: {C.fmt_score(val)}")
            st.progress(min(max(val / 100.0, 0.0), 1.0))
        with st.expander(C.SEC_GLOSSARY):
            for term, text in C.GLOSSARY.items():
                st.write(f"{term}: {text}")
    else:
        st.info(C.SCORES_NONE.format(name=name))

    # News and filings (on-demand, 24 h cache; spinner on a cache miss).
    st.subheader(C.SEC_NEWS)
    _items = load_news(symbol)
    if not _items:
        st.info(C.EMPTY_NO_NEWS.format(name=name))
    else:
        for it in _items:
            senti = ("positive" if it.get("score", 0) > 0
                     else "negative" if it.get("score", 0) < 0 else "neutral")
            when = C.fmt_weekday_date(it.get("published_at")) or C.fmt_date(it.get("published_at"))
            st.write(f"{when} · {it.get('source', '')} · {it.get('headline', '')} · {senti}")

    # How to buy.
    st.subheader(C.SEC_HOW_TO_BUY)
    if broker.get("isin"):
        st.write(f"ISIN {broker['isin']}")
        if broker.get("isin_source") == "yahoo":
            st.caption(C.HELP_ISIN_YAHOO_CAVEAT)
        st.write(f"Route: {'savings plan' if cls in ('ETF', 'CASH') else 'one-off order'}")
    else:
        st.warning(C.HOW_TO_BUY_ISIN_MISSING.format(name=name))


# ── Page: Settings (P7) ───────────────────────────────────────────────────────

def page_settings() -> None:
    st.title(C.PAGE_SETTINGS)

    # Data status (A2): no placeholder sentence. The full sentence renders only
    # when all three facts exist; otherwise a genuine missing-data empty state.
    st.subheader(C.SEC_DATA_STATUS)
    instruments = q("SELECT COUNT(DISTINCT Symbol) AS n FROM market_history")
    m = int(instruments["n"].iloc[0]) if not instruments.empty else 0
    bar = latest_bar_date()
    history = read_history()
    if m > 0 and bar and not history.empty:
        refreshed = C.fmt_ts(history.iloc[-1]["review_ts"])
        st.write(f"{m} instruments, prices through {C.fmt_date(bar)}, "
                 f"refreshed {refreshed}.")
    elif m > 0 and bar:
        st.write(f"{m} instruments, prices through {C.fmt_date(bar)}.")
    else:
        st.info(C.EMPTY_NO_MARKET_DATA)
    _run_operation(C.BTN_REFRESH, C.BTN_REFRESHING, runner.REFRESH, "refresh")
    st.caption(C.HELP_REVIEW_CADENCE)

    # Reviews (last ten). The value-chart sentence belongs to Today only (A2).
    st.subheader(C.SEC_REVIEWS)
    if history.empty:
        st.info(C.EMPTY_NO_REVIEWS)
    else:
        for _, r in history.tail(10)[::-1].iterrows():
            st.write(f"{C.fmt_ts(r['review_ts'])} - {C.fmt_eur(r['value_eur'])} - "
                     f"{C.fmt_eur(r['pnl_eur'])}")

    # Health (problems only). The ONLY Repair registry button lives here (2.4).
    problems = _health_problems()
    st.subheader("Health")
    isin_missing = _isin_missing_holdings()
    if problems:
        for p in problems:
            st.warning(p)
    for sym in isin_missing:
        st.warning(C.ACTION_BLOCKED.format(symbol=sym))
        res = _run_operation(C.BTN_REPAIR_REGISTRY, C.BTN_REPAIRING, runner.REPAIR,
                             f"health_fix_{sym}", runner_fn=runner.run_repair)
        if res is not None and res.ok:
            st.rerun()
    if not problems and not isin_missing:
        st.success(C.STATUS_ALL_CURRENT)

    # Diagnostics (collapsed).
    with st.expander(C.SEC_DIAGNOSTICS):
        from quant.portfolio.cash_rate import as_dicts
        st.write(f"Version {__version__}")
        st.write(f"Cash rate: {current_cash_apy()*100:.2f} percent")
        st.write(f"Stale threshold: {STALE_DATA_DAYS} days")
        for row in as_dicts():
            st.caption(f"{row['effective_date']} · {row['apy']*100:.2f}% · {row['source_url']}")
        try:
            from quant.data.names import read_names_state

            st_ns = read_names_state()
            if st_ns:
                st.caption(f"Names: filled {st_ns.get('filled', 0)}, "
                           f"still missing {st_ns.get('still_missing', 0)}, "
                           f"skipped {st_ns.get('skipped_reason') or 'no'}")
        except Exception:  # noqa: BLE001
            pass


def _health_problems() -> list[str]:
    """Return plain-sentence problems only (empty when clean).

    A2: the stale rule fires only when data exists and its age >= threshold; no
    data never produces "Prices are 0 days old." A failed regime computation is
    a Health item, never masked as missing history.
    """
    problems: list[str] = []
    bar = latest_bar_date()
    if bar:
        try:
            from datetime import date
            days = (date.today() - date.fromisoformat(bar)).days
            if days >= STALE_DATA_DAYS:
                problems.append(C.ERROR_STALE_PRICES.format(n=days))
        except Exception:  # noqa: BLE001
            pass
    if read_regime().get("state") == "failed":
        problems.append(C.HEALTH_REGIME_FAILED)
    try:
        from quant.data.news import outage_message

        msg = outage_message()
        if msg:
            problems.append(msg)
    except Exception:  # noqa: BLE001
        pass
    try:
        from quant.data.names import read_names_state

        ns = read_names_state()
        if int(ns.get("still_missing", 0)) > 0 and ns.get("skipped_reason") == "metadata_unreachable":
            problems.append(C.HEALTH_NAMES_MISSING.format(n=int(ns["still_missing"])))
    except Exception:  # noqa: BLE001
        pass
    return problems


def _isin_missing_holdings() -> list[str]:
    """Holding symbols with no registry ISIN (the only place Repair lives)."""
    missing: list[str] = []
    portfolio = load_portfolio()
    if not portfolio.empty:
        for sym in portfolio["Symbol"].astype(str):
            if not resolve_broker(sym).get("isin"):
                missing.append(sym)
    return missing


def _render_suppression_footnotes(holdings: list[dict]) -> None:
    """S3: one footnote line per suppression kind present (spec 2.1)."""
    below = [h for h in holdings if h.get("suppressed") == "below_min"]
    if below:
        for val in sorted({h["min_trade_eur"] for h in below}):
            n = sum(1 for h in below if h["min_trade_eur"] == val)
            tpl = C.FOOTNOTE_BELOW_MIN_ONE if n == 1 else C.FOOTNOTE_BELOW_MIN
            st.caption(tpl.format(n=n, min=f"{val:.0f}"))
    cooldown = [h for h in holdings if str(h.get("status", "")).startswith("Waiting until")]
    if cooldown:
        for when in {h.get("cooldown_until") for h in cooldown}:
            n = sum(1 for h in cooldown if h.get("cooldown_until") == when)
            tpl = C.FOOTNOTE_COOLDOWN_ONE if n == 1 else C.FOOTNOTE_COOLDOWN
            st.caption(tpl.format(n=n, date=C.fmt_date(when)))


# ── Value chart helpers (spec 3.1; pure where possible) ───────────────────────
_RANGE_DAYS = {"1M": 31, "3M": 92, "1Y": 365, "Max": None}


def _filter_range(df: "pd.DataFrame", rng: str) -> "pd.DataFrame":
    """Return rows of df within the selected range (days back from the last)."""
    import pandas as _pd

    days = _RANGE_DAYS.get(rng or "Max")
    if not days or df.empty:
        return df
    ts = _pd.to_datetime(df["review_ts"], errors="coerce")
    cutoff = ts.max() - _pd.Timedelta(days=days)
    return df[ts >= cutoff]


def _rebase(values):
    """Rebase a numeric series to 100 at its first point (Growth mode)."""
    base = values.iloc[0] if hasattr(values, "iloc") else values[0]
    if not base:
        return values
    return values / base * 100.0


def _range_annotation(df: "pd.DataFrame") -> str:
    """`+4.2% since 1 Jun 2026 (34.80 EUR)` from the range endpoints."""
    if df.empty or len(df) < 2:
        return ""
    first, last = df.iloc[0], df.iloc[-1]
    if not first["value_eur"]:
        return ""
    pct = (last["value_eur"] / first["value_eur"] - 1.0) * 100.0
    abs_ = last["value_eur"] - first["value_eur"]
    sign = "+" if pct >= 0 else ""
    return C.CHART_SINCE.format(sign=sign, pct=f"{pct:.1f}",
                                date=C.fmt_date(first["review_ts"]), amount=f"{abs_:.2f}")


def _render_value_chart(history, holdings) -> None:
    """Today value chart: range selector, baseline, annotation, Value|Growth."""
    import plotly.graph_objects as go

    if history is None or len(history) < 3:
        st.info(C.CHART_BUILDING)
        return
    df = history.copy()
    df["review_ts"] = pd.to_datetime(df["review_ts"], errors="coerce")
    df = df.dropna(subset=["review_ts"]).sort_values("review_ts")
    rng = st.segmented_control(C.LABEL_RANGE, list(_RANGE_DAYS),
                               default="Max", key="val_range") or "Max"
    df = _filter_range(df, rng)
    if len(df) < 2:
        st.info(C.CHART_BUILDING)
        return
    mode = st.segmented_control(C.LABEL_VIEW, [C.VALUE, C.GROWTH],
                                default=C.VALUE, key="val_mode") or C.VALUE
    growth = mode == C.GROWTH

    fig = go.Figure()
    port = _rebase(df["value_eur"]) if growth else df["value_eur"]
    fig.add_trace(go.Scatter(
        x=df["review_ts"], y=port, mode="lines", name="Portfolio",
        line=dict(width=2, color="#1F3B73"),
        fill=None if growth else "tozeroy"))

    palette = ["#3B5C99", "#5B83BF", "#8FA9CF"]
    if growth:
        shown = [h for h in (holdings or []) if not h.get("blocked")][:3]
        for i, h in enumerate(shown):
            mh = q("SELECT Date AS d, Close FROM market_history WHERE Symbol = ? "
                   "ORDER BY Date ASC", [h["symbol"]])
            if mh.empty:
                continue
            mh["d"] = pd.to_datetime(mh["d"], errors="coerce")
            mh = mh.dropna(subset=["d"])
            mh = mh[mh["d"] >= df["review_ts"].min()]
            if len(mh) < 2:
                continue
            fig.add_trace(go.Scatter(x=mh["d"], y=_rebase(mh["Close"]),
                                     mode="lines", name=h["symbol"],
                                     line=dict(width=1.5, color=palette[i % 3])))
        if st.checkbox(C.LABEL_BENCHMARK, value=False, key="val_bench"):
            bench = q("SELECT Date AS d, Close FROM market_history WHERE Symbol = ? "
                      "ORDER BY Date ASC", [C.BENCHMARK_SYMBOL])
            if not bench.empty:
                bench["d"] = pd.to_datetime(bench["d"], errors="coerce")
                bench = bench.dropna(subset=["d"])
                bench = bench[bench["d"] >= df["review_ts"].min()]
                if len(bench) >= 2:
                    fig.add_trace(go.Scatter(x=bench["d"], y=_rebase(bench["Close"]),
                                             mode="lines", name=C.BENCHMARK_SYMBOL,
                                             line=dict(width=1.5, color="#8A8F99")))

    base = 100.0 if growth else float(df["value_eur"].iloc[0])
    fig.add_hline(y=base, line_dash="dot", line_color="#888")
    ann = _range_annotation(df)
    if ann:
        fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.98, text=ann,
                           showarrow=False, align="left", font=dict(size=12))
    fig.update_layout(height=280, margin=dict(l=8, r=8, t=8, b=8),
                      xaxis=dict(tickformat="%d %b"), showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                                  xanchor="left", x=0))
    st.plotly_chart(fig, width="stretch")
