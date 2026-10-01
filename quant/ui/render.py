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
import os
import threading
from datetime import date as _date
from datetime import datetime as _dt

import pandas as pd
import streamlit as st

from quant import __version__, paths
from quant.config import RISK_PROFILE_DESCRIPTIONS, RISK_PROFILES, STALE_DATA_DAYS
from quant.data.database import read_only_connection
from quant.data.news import load_news
from quant.execution.routing import holding_routes_to_savings_plan
from quant.execution.taxonomy import (
    INVERSE_STRUCTURE,
    LEVERAGED_STRUCTURE,
    get_structure,
    resolve_broker,
)
from quant.portfolio.account import AccountState, load_account, save_account
from quant.portfolio.cash_rate import current_cash_apy, current_rate
from quant.portfolio.editor import save_portfolio, validate_positions
from quant.reporting.artifacts import (
    latest_ok_review_ts,
    latest_review,
    read_actions,
    read_history,
    read_regime,
    read_scores,
    read_update_state,
)
from quant.ui import copy as C
from quant.ui import runner
from quant.ui.cards import explore_card_fields
from quant.ui.search import discovery_candidates, label_for, load_index, search

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

def q(sql: str, params: list | None = None) -> pd.DataFrame:
    """Run a read-only query. Returns an empty frame on any failure."""
    try:
        with read_only_connection() as conn:
            return conn.execute(sql, params or []).df()
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def load_portfolio() -> pd.DataFrame:
    """Load the broker-synced portfolio (empty frame on failure)."""
    try:
        from quant.portfolio.portfolio import load_portfolio as _lp
        return _lp(paths.DATA_PORTFOLIO)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def latest_bar_date() -> str:
    """Return the latest market bar date as a string (empty when absent)."""
    df = q("SELECT MAX(Date) AS d FROM market_history")
    if df.empty or df["d"].iloc[0] is None:
        return ""
    return str(df["d"].iloc[0])


def _markets_closed_line(today: _date | None = None) -> str:
    """R8: markets-closed freshness line.

    Intent: when today is non-trading (weekend) and the latest bar is the
    previous trading session (Friday), the Today header states the close date
    instead of implying stale data. Invariants: empty string when the bar is
    missing, unparseable, or today is a trading day.
    """
    bar = latest_bar_date()
    if not bar:
        return ""
    try:
        bd = _dt.fromisoformat(str(bar)).date()
    except ValueError:
        return ""
    today = today or _date.today()
    if today.weekday() >= 5 and bd < today and bd.weekday() == 4:
        return C.MARKETS_CLOSED.format(date=C.fmt_weekday_date(bd))
    return ""


def _any_savings_plan_holding(portfolio) -> bool:
    """R8: True when at least one holding routes to a savings plan."""
    if portfolio is None or portfolio.empty:
        return False
    for sym in portfolio["Symbol"].astype(str):
        broker = resolve_broker(sym) or {}
        if holding_routes_to_savings_plan(broker.get("instrument_class", ""),
                                          get_structure(sym)):
            return True
    return False


# ── Shared renderers ──────────────────────────────────────────────────────────

def render_action_cards(holdings: list[dict]) -> None:
    """Render action cards from the ONE advice pipeline (v10.7.1)."""
    cards = [h for h in holdings if h.get("action") and not h.get("blocked")]
    if not cards:
        st.info(C.EMPTY_NOTHING_TO_DO)
        return
    for a in cards:
        kind = a.get("kind")
        if not kind:
            # Legacy callers pass only the action word; map it.
            word = a.get("action", "")
            kind = "buy" if word == "BUY MORE" else ("sell_part" if word == "TRIM" else "keep")
        target = (a.get("target_weight") or "").rstrip("%") or "?"
        pct = abs(float(str(a.get("drift", "0")).rstrip("%") or 0))
        name = a.get("name") or a["symbol"]
        amount = a.get("amount_eur")
        if kind == "buy" and amount:
            st.write(C.ACTION_ADD.format(
                amount=f"{amount:.0f}", symbol=a["symbol"],
                name=name, pct=f"{pct:.0f}", target=target))
        elif kind == "sell_part" and amount:
            st.write(C.ACTION_SELL.format(
                amount=f"{amount:.0f}", symbol=a["symbol"],
                pct=f"{pct:.0f}", target=target))
        elif kind in ("change_savings_plan", "to_cash"):
            st.write(a.get("reason") or C.ADVICE_KEEP)


# ── v10.6.2: Three-tier dashboard, emergency liquidity, tax-loss ─────────────

def render_empty_state(tier: str) -> None:
    """Render a helpful empty state for a tier (v10.6.3)."""
    if tier == "FORTRESS":
        st.info(C.EMPTY_FORTRESS)
    elif tier == "ALPHA":
        st.info(C.EMPTY_ALPHA)
    elif tier == "SPECULATIVE":
        st.info(C.EMPTY_SPECULATIVE)
    else:
        st.info(C.EMPTY_TIER)


def _add_asset_to_tier(symbol: str, tier: str) -> None:
    """Append a symbol to data/tiers.csv with the given tier (v10.6.3)."""
    from quant.portfolio.tier_manager import load_tiers, save_tiers

    symbol = str(symbol).strip().upper()
    if not symbol:
        return
    tiers_df = load_tiers()
    if not tiers_df.empty and symbol in set(tiers_df["symbol"].astype(str)):
        return
    row = pd.DataFrame([{
        "symbol": symbol, "tier": tier,
        "last_updated": _date.today().isoformat(),
        "notes": "Added via onboarding",
    }])
    save_tiers(pd.concat([tiers_df, row], ignore_index=True))


def render_onboarding_wizard() -> None:
    """Three-step onboarding wizard for first-time users (v10.6.3)."""
    if st.session_state.get("onboarding_completed"):
        return
    portfolio = load_portfolio()
    if portfolio is not None and not portfolio.empty:
        return
    st.header(C.ONBOARD_TITLE)
    st.write(C.ONBOARD_INTRO)
    step = st.session_state.get("onboarding_step", 1)
    if step == 1:
        st.subheader(C.ONBOARD_STEP1)
        sym = st.text_input(C.ONBOARD_SYMBOL_LABEL, value="URTH", key="onboard_fortress")
        if st.button(C.ONBOARD_ADD_FORTRESS, key="onboard_add_fortress"):
            _add_asset_to_tier(sym, "FORTRESS")
            st.session_state["onboarding_step"] = 2
            st.rerun()
    elif step == 2:
        st.subheader(C.ONBOARD_STEP2)
        sym = st.text_input(C.ONBOARD_SYMBOL_LABEL, value="NVDA", key="onboard_alpha")
        if st.button(C.ONBOARD_ADD_ALPHA, key="onboard_add_alpha"):
            _add_asset_to_tier(sym, "ALPHA")
            st.session_state["onboarding_step"] = 3
            st.rerun()
    elif step == 3:
        st.subheader(C.ONBOARD_STEP3)
        st.number_input(C.ONBOARD_SPARPLAN_LABEL, min_value=25, max_value=1000,
                        value=150, key="onboard_sparplan")
        if st.button(C.ONBOARD_COMPLETE, key="onboard_complete"):
            st.session_state["onboarding_completed"] = True
            st.success(C.ONBOARD_DONE)
            st.rerun()
    if st.button(C.ONBOARD_SKIP, key="onboard_skip"):
        st.session_state["onboarding_completed"] = True
        st.rerun()


def render_tier_dashboard(portfolio: pd.DataFrame | None) -> None:
    """Render the three-tier dashboard (Fortress / Alpha / Speculative tabs)."""
    st.subheader(C.SEC_TIERS)
    tiers = ["FORTRESS", "ALPHA", "SPECULATIVE"]
    labels = [C.TIER_FORTRESS, C.TIER_ALPHA, C.TIER_SPECULATIVE]
    helps = [C.HELP_TIER_FORTRESS, C.HELP_TIER_ALPHA, C.HELP_TIER_SPECULATIVE]
    for tab, tier, help_text in zip(st.tabs(labels), tiers, helps):
        with tab:
            st.caption(help_text)
            if portfolio is None or portfolio.empty or "Tier" not in portfolio.columns:
                render_empty_state(tier)
                continue
            sub = portfolio[portfolio["Tier"] == tier]
            if sub.empty:
                render_empty_state(tier)
                continue
            cols = [c for c in ["Symbol", "Current_Value_EUR", "Broker_PnL_EUR"]
                    if c in sub.columns]
            st.dataframe(sub[cols], width="stretch", hide_index=True)


def render_emergency_liquidity(portfolio: pd.DataFrame | None) -> None:
    """Emergency liquidity calculator: amount input -> tier-aware sell order."""
    st.subheader(C.SEC_EMERGENCY)
    amount = st.number_input(C.EMERGENCY_PROMPT, min_value=0.0, value=0.0, step=100.0)
    if amount <= 0:
        return
    if portfolio is None or portfolio.empty or "Tier" not in portfolio.columns:
        st.info(C.EMERGENCY_NONE)
        return
    from quant.portfolio.risk import emergency_sell_plan
    plan = emergency_sell_plan(float(amount), portfolio)
    if plan["fortress_warning"]:
        st.warning(plan["fortress_warning"])
    if not plan["recommendations"]:
        st.info(C.EMERGENCY_NONE)
        return
    st.write(C.EMERGENCY_ORDER)
    for h in plan["recommendations"]:
        pnl = float(h.get("pnl_eur", 0) or 0)
        tax = pnl * 0.26375
        if pnl < 0:
            note = "loss, tax-loss harvest"
        elif pnl > 0:
            note = "profit, taxable"
        else:
            note = "no gain or loss"
        st.write(C.EMERGENCY_LINE.format(
            symbol=h["symbol"], value=f"{h['value_eur']:.0f}",
            tax=f"{tax:.2f}", note=note))


def render_tax_loss_alerts(portfolio: pd.DataFrame | None) -> None:
    """Highlight positions with an unrealized loss (tax-loss candidates)."""
    st.subheader(C.SEC_TAX_LOSS)
    if portfolio is None or portfolio.empty or "Broker_PnL_EUR" not in portfolio.columns:
        st.info(C.TAX_LOSS_NONE)
        return
    pnl = pd.to_numeric(portfolio["Broker_PnL_EUR"], errors="coerce")
    losers = portfolio[pnl < 0]
    if losers.empty:
        st.info(C.TAX_LOSS_NONE)
        return
    st.write(C.TAX_LOSS_HEADER)
    for _, r in losers.iterrows():
        st.write(C.TAX_LOSS_LINE.format(
            name=r.get("Name") or r.get("Symbol"),
            pnl=f"{float(r.get('Broker_PnL_EUR', 0)):.2f}"))


def render_autobalance_section() -> None:
    """Render the tier auto-balance section in the Portfolio page (v10.6.4)."""
    from quant.portfolio.autobalance import (
        analyze_tier_allocations,
        apply_rebalance_suggestions,
        suggest_rebalance,
    )
    from quant.portfolio.portfolio import load_portfolio
    from quant.portfolio.tier_manager import load_tiers, save_tiers

    portfolio_df = load_portfolio()
    tiers_df = load_tiers()
    analysis = analyze_tier_allocations(portfolio_df, tiers_df)

    st.subheader(C.SEC_AUTOBALANCE)
    for tier, alloc in analysis["allocations"].items():
        limit_str = f"{alloc['limit']:.0%}" if alloc["limit"] else "no limit"
        line = C.AUTOBALANCE_LINE.format(
            tier=tier, value=f"{alloc['value_eur']:.0f}",
            pct=f"{alloc['pct']:.1%}", limit=limit_str)
        if alloc["violated"]:
            st.error(line)
        else:
            st.write(line)

    if not analysis["violations"]:
        st.success(C.AUTOBALANCE_OK)
        return

    st.warning(C.AUTOBALANCE_VIOLATION.format(tiers=", ".join(analysis["violations"])))
    suggestions = suggest_rebalance(portfolio_df, tiers_df)
    if not suggestions:
        st.info(C.AUTOBALANCE_NONE)
        return

    st.write(C.AUTOBALANCE_SUGGESTIONS.format(n=len(suggestions)))
    approved: list[str] = []
    for i, s in enumerate(suggestions, 1):
        with st.expander(C.AUTOBALANCE_SUGGESTION_TITLE.format(
                i=i, symbol=s["symbol"], source=s["current_tier"],
                target=s["suggested_tier"])):
            st.write(C.AUTOBALANCE_MOVE.format(
                source=s["current_tier"], target=s["suggested_tier"]))
            st.write(C.AUTOBALANCE_VALUE.format(value=f"{s['value_eur']:.2f}"))
            st.write(s["reason"])
            if st.checkbox(C.AUTOBALANCE_APPROVE.format(i=i), key=f"approve_{i}"):
                approved.append(s["symbol"])
    if approved:
        if st.button(C.BTN_APPLY_AUTOBALANCE, key="apply_autobalance"):
            updated = apply_rebalance_suggestions(tiers_df, suggestions, approved)
            save_tiers(updated)
            st.success(C.AUTOBALANCE_APPLIED.format(n=len(approved)))
            st.rerun()
        st.caption(C.AUTOBALANCE_MANUAL)


def render_trade_limit_warning() -> None:
    """Show a warning when the weekly Alpha trade limit is reached (v10.6.3)."""
    from datetime import date as _d

    from quant.config import MAX_ALPHA_TRADES_PER_WEEK
    from quant.portfolio.behavioral_guardrails import track_weekly_trades

    trades = track_weekly_trades(_d.today().isoformat())
    remaining = max(0, MAX_ALPHA_TRADES_PER_WEEK - trades)
    if trades >= MAX_ALPHA_TRADES_PER_WEEK:
        st.error(C.TRADE_LIMIT_REACHED.format(n=trades, max=MAX_ALPHA_TRADES_PER_WEEK))
    elif remaining == 1:
        st.warning(C.TRADE_LIMIT_ONE_LEFT)


def render_tier_assignment_alerts() -> None:
    """Show alerts for unclassified assets with an auto-assign action (v10.6.3)."""
    from quant.portfolio.portfolio import load_portfolio
    from quant.portfolio.tier_manager import (
        auto_assign_tiers,
        detect_unclassified_assets,
        load_tiers,
        save_tiers,
    )

    portfolio_df = load_portfolio()
    tiers_df = load_tiers()
    unclassified = detect_unclassified_assets(portfolio_df, tiers_df)
    if not unclassified:
        return
    st.warning(C.TIER_UNCLASSIFIED_WARNING.format(n=len(unclassified)))
    for rec in unclassified:
        st.info(C.TIER_UNCLASSIFIED_LINE.format(
            symbol=rec["symbol"], tier=rec["recommended_tier"], reason=rec["reason"]))
    if st.button(C.BTN_AUTO_ASSIGN_TIERS, key="auto_assign_tiers"):
        updated = auto_assign_tiers(unclassified, tiers_df)
        save_tiers(updated)
        st.success(C.TIER_AUTO_ASSIGNED)
        st.rerun()


# ── Page: Today (P4) ──────────────────────────────────────────────────────────

def _resolve_alert(alert_id: int, status: str, reason: str | None) -> None:
    """Resolve an alert from the UI (write connection, short-lived)."""
    try:
        from quant.data.database import connect_with_retry
        from quant.engine import alerts as alerts_mod

        conn = connect_with_retry()
        try:
            alerts_mod.resolve_alert(conn, alert_id, status, reason)
        finally:
            conn.close()
        st.success(C.ALERT_RESOLVED.format(status=status))
    except Exception:  # noqa: BLE001
        st.warning("Could not resolve the action. Try again.")


def render_alerts_banner() -> None:
    """Red block listing open alerts until resolved (v10.7.0, Section 4.4)."""
    try:
        from quant.data.database import read_only_connection
        from quant.engine import alerts as alerts_mod

        with read_only_connection() as conn:
            open_now = alerts_mod.open_alerts(conn)
    except Exception:  # noqa: BLE001
        return
    if not open_now:
        return
    st.error(C.ALERT_BANNER_TITLE)
    for alert in open_now:
        st.write(alert.get("message", ""))
        cols = st.columns(2)
        if cols[0].button(C.BTN_ALERT_DONE, key=f"alert_done_{alert['id']}"):
            _resolve_alert(alert["id"], "done", None)
        if cols[1].button(C.BTN_ALERT_DECLINED, key=f"alert_decl_{alert['id']}"):
            _resolve_alert(alert["id"], "declined", "declined in app")


def _render_overview_money(portfolio, account) -> None:
    """Block A: the two money plaques (v10.7.0, Section 10.1)."""
    st.subheader(C.SEC_YOUR_MONEY)
    invested = 0.0
    pnl = 0.0
    if portfolio is not None and not portfolio.empty:
        if "Current_Value_EUR" in portfolio.columns:
            invested = float(portfolio["Current_Value_EUR"].sum())
        if "Broker_PnL_EUR" in portfolio.columns:
            pnl = float(portfolio["Broker_PnL_EUR"].sum())
    cost = invested - pnl
    pct = (pnl / cost * 100) if cost > 0 else 0.0
    cash = account.cash_eur if account.cash_is_set else 0.0
    cols = st.columns(2)
    cols[0].metric("Invested", C.fmt_eur(invested))
    cols[0].caption(C.INVESTED_LINE.format(
        amount=f"{invested:.0f}", pnl=f"{pnl:+.2f}", pct=f"{pct:+.1f}",
        date=C.fmt_date(_date.today())))
    cols[1].metric("Operational cash", C.fmt_eur(cash))
    cols[1].caption(C.OPERATIONAL_CASH_LINE.format(
        amount=f"{cash:.0f}", date=C.fmt_date(_date.today()), apy="2.5"))


def _render_overview_steps(holdings) -> None:
    """Block B: your steps this week + the Not this week block."""
    from quant.data.database import read_only_connection
    from quant.engine import alerts as alerts_mod
    from quant.engine import plans
    from quant.engine import steps as steps_mod

    st.subheader(C.SEC_STEPS)
    try:
        with read_only_connection() as conn:
            open_now = alerts_mod.open_alerts(conn)
            plan = plans.load_plan(conn, _month_key())
    except Exception:  # noqa: BLE001
        open_now, plan = [], None
    built = steps_mod.build_steps(_date.today(), open_now, plan, holdings)
    if built:
        for i, step in enumerate(built, 1):
            st.write(f"{i}. {step['what']}")
            if step.get("amount_eur"):
                st.caption(f"{C.fmt_eur(step['amount_eur'])}. {step['why']}")
    else:
        st.write(C.NOTHING_TO_DO_WEEK)
    st.write(C.SEC_NOT_THIS_WEEK)
    # v10.7.1: the "Not this week" block renders the advice pipeline's rejected
    # notes (the system showing its work), not a second computation.
    try:
        from quant.engine.advice import build_advice

        _advice, rejected = build_advice(
            _monthly_holdings(), open_alerts=open_now, plans=plan)
    except Exception:  # noqa: BLE001
        rejected = []
    if rejected:
        for note in rejected:
            st.caption(f"{note['symbol']}: {note['plain_reason']}")
    else:
        st.caption(C.NOTHING_TO_DO_WEEK)


def _render_overview_savings(plan) -> None:
    """Block C: the savings plan."""
    st.subheader(C.SEC_SAVINGS_PLAN)
    if plan:
        st.write(f"Budget: {plan['budget_eur']:.0f} EUR per month. This month: "
                 f"approved on {C.fmt_date(plan.get('approved_date'))}.")
    else:
        st.write("No plan approved yet for this month.")


def _render_market_expander(has_review: bool) -> None:
    """Block E: the market expander (regime + the honest news-pillar line).

    The regime line renders only when a review exists (S0 has no regime); the
    news-pillar line renders whenever the pillar is absent, independent of the
    review.
    """
    with st.expander(C.SEC_MARKET):
        if has_review:
            reg = read_regime()
            if reg.get("state") == "estimated":
                st.write(C.MARKET_TREND.format(label=reg.get("label"),
                                               confidence=reg.get("confidence")))
            elif reg.get("state") == "failed":
                st.write(C.MARKET_TREND_FAILED)
            else:
                st.write(C.MARKET_TREND_INSUFFICIENT)
        # v10.7.2 (Part 2.3): the exact absent line when the news pillar is off.
        try:
            from quant.engine import news_pillar

            if news_pillar.is_absent():
                st.write(C.NEWS_PILLAR_ABSENT)
        except Exception:  # noqa: BLE001
            pass


def page_today() -> None:
    """Render the Overview page."""
    st.title(C.PAGE_TODAY)
    render_alerts_banner()

    portfolio = load_portfolio()
    history = read_history()
    # H3.7 (L1): the header/tables/regime read the most recent SUCCESSFUL
    # review; the S4 card reads the latest review attempt of any status.
    attempt = latest_review()
    review = latest_review(ok_only=True)
    has_review = bool(review)

    if attempt.get("review_status") == "failed":
        # S4: the review ran and failed (single source: the run artifact).
        if not history.empty:
            last_ts = history.iloc[-1]["review_ts"]
            st.write(C.HEADER_REVIEW.format(date=C.fmt_date(latest_bar_date()),
                                            prepared=C.fmt_review_ts(last_ts)))
        st.warning(C.LAST_REVIEW_FAILED)
    elif not has_review:
        st.info(C.GUIDE_NO_REVIEW)
    else:
        prepared = C.fmt_review_ts(review.get("review_ts"))
        # H3.8 (M1): BOTH header fields come from the SAME review artifact; a
        # legacy artifact with no close date renders only the prepared line
        # (never a borrowed live close date).
        _bar = review.get("latest_bar")
        bar = C.fmt_date(_bar) if (_bar and str(_bar).strip().lower()
                                   not in ("unknown", "nan", "none")) else ""
        if bar:
            st.write(C.HEADER_REVIEW.format(date=bar, prepared=prepared))
        else:
            st.write(C.HEADER_REVIEW_PREPARED.format(prepared=prepared))
        reg = read_regime()
        if reg.get("state") == "estimated":
            st.write(C.MARKET_TREND.format(label=reg.get("label"),
                                           confidence=reg.get("confidence")))
        elif reg.get("state") == "failed":
            st.write(C.MARKET_TREND_FAILED)
        else:
            st.write(C.MARKET_TREND_INSUFFICIENT)

    # R8: markets-closed freshness line (header-level, independent of review).
    _closed = _markets_closed_line()
    if _closed:
        st.write(_closed)

    holdings = read_actions()

    # v10.7.0 Overview blocks A-C (Section 10.1).
    _account = load_account()
    _render_overview_money(portfolio, _account)
    _render_overview_steps(holdings)
    try:
        from quant.data.database import read_only_connection
        from quant.engine import plans as _plans

        with read_only_connection() as _conn:
            _plan = _plans.load_plan(_conn, _month_key())
    except Exception:  # noqa: BLE001
        _plan = None
    _render_overview_savings(_plan)
    _render_market_expander(has_review)

    # 2. Portfolio value chart (spec 3.1).
    st.subheader(C.SEC_PORTFOLIO_VALUE)
    _render_value_chart(history, holdings)

    # 3. How your invested money is split (donut). v10.7.0: the INVESTED pool
    #    only. Operational cash is never a slice (it is not an investment
    #    buffer). Categorical blue/gray palette only; semantic colors never
    #    encode composition (A4). Percent labels only for slices >= 5 percent.
    st.subheader(C.SEC_WHERE_MONEY)
    account = load_account()
    if not portfolio.empty:
        import plotly.graph_objects as go
        values = list(portfolio["Amount_EUR"])
        labels = list(portfolio["Symbol"])
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
        st.caption("of invested")

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
        # R8: savings-plan countdown under the actions block, only when a
        # holding actually routes to a savings plan.
        if account.savings_plan_day and _any_savings_plan_holding(portfolio):
            st.write(C.savings_plan_line(_date.today(), account.savings_plan_day))

    # 6. Needs attention first (sentence only; Repair lives in Settings Health).
    blockers = [h for h in holdings if h["blocked"]]
    if blockers:
        st.subheader(C.SEC_NEEDS_ATTENTION)
        for b in blockers:
            st.warning(C.NEEDS_ATTENTION_ISIN.format(symbol=b["symbol"]))


# ── Page: My holdings (v10.7.1, Section 10.2) ─────────────────────────────────

def _latest_close(symbol: str) -> float | None:
    """Latest close for a symbol from market_history; None when absent."""
    try:
        df = q("SELECT Close FROM market_history WHERE Symbol = ? "
               "ORDER BY Date DESC LIMIT 1", [symbol])
    except Exception:  # noqa: BLE001
        return None
    if df is None or df.empty:
        return None
    try:
        return float(df["Close"].iloc[0])
    except (TypeError, ValueError):
        return None


def _render_asset_chart(symbol: str) -> None:
    """Per-asset line chart: normalized to 100, 1M/3M/1Y/Max, no fill."""
    import plotly.graph_objects as go

    df = q("SELECT Date AS d, Close FROM market_history WHERE Symbol = ? "
           "ORDER BY Date ASC", [symbol])
    if df is None or df.empty:
        st.caption(C.CHART_NO_HISTORY.format(name=symbol))
        return
    df["d"] = pd.to_datetime(df["d"], errors="coerce")
    df = df.dropna(subset=["d"]).sort_values("d")
    rng = st.segmented_control(C.LABEL_RANGE, list(_RANGE_DAYS),
                               default="Max", key=f"asset_range_{symbol}") or "Max"
    df = _filter_range(df, rng)
    if len(df) < 2:
        st.caption(C.CHART_NO_HISTORY.format(name=symbol))
        return
    base = float(df["Close"].iloc[0]) or 1.0
    series = df["Close"] / base * 100.0
    fig = go.Figure(go.Scatter(x=df["d"], y=series, mode="lines",
                               line=dict(width=2, color="#1F3B73"), fill=None))
    fig.add_hline(y=100.0, line_dash="dot", line_color="#888")
    lo, hi = float(series.min()), float(series.max())
    pad = (hi - lo) * 0.1 or 1.0
    fig.update_layout(height=240, margin=dict(l=8, r=8, t=8, b=8),
                      yaxis=dict(range=[lo - pad, hi + pad]), showlegend=False)
    st.plotly_chart(fig, width="stretch")


def _render_holdings_table(portfolio, holdings) -> None:
    """Block 2: the holdings table (no per-share columns, B7)."""
    st.subheader(C.PAGE_HOLDINGS)
    by_sym = {h["symbol"]: h for h in holdings}
    rows = []
    if portfolio is not None and not portfolio.empty:
        for _, r in portfolio.iterrows():
            sym = str(r["Symbol"])
            h = by_sym.get(sym, {})
            scores = read_scores(sym)
            rows.append({
                "Name": label_for(r.get("Name") or sym, sym),
                "Value (EUR)": C.fmt_eur(r.get("Current_Value_EUR")),
                "Profit (EUR)": C.fmt_eur(r.get("Broker_PnL_EUR")),
                "Structure": scores.get("structural_grade") or "",
                "Tactics": scores.get("tactical_grade") or "",
                "Verdict": h.get("status", C.STATUS_NOT_REVIEWED),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_holding_expanders(portfolio, holdings) -> None:
    """Block 3: one expander per holding with the per-asset detail."""
    if portfolio is None or portfolio.empty:
        return
    by_sym = {h["symbol"]: h for h in holdings}
    for _, r in portfolio.iterrows():
        sym = str(r["Symbol"])
        name = label_for(r.get("Name") or sym, sym)
        with st.expander(name):
            _render_asset_chart(sym)
            entry = float(r.get("Avg_Entry_Price", 0) or 0)
            current = _latest_close(sym)
            st.write(C.ENTRY_PRICE_LINE.format(
                entry=f"{entry:.2f}",
                current=f"{current:.2f}" if current is not None else "not available yet",
                estimated=C.ESTIMATED_LABEL.format(date=C.fmt_date(_date.today()))))
            try:
                from quant.data.database import read_only_connection

                with read_only_connection() as conn:
                    row = conn.execute(
                        "SELECT shares, sync_date FROM holdings_meta WHERE symbol = ?",
                        [sym]).fetchone()
                if row:
                    st.write(C.SHARES_LINE.format(
                        shares=f"{float(row[0]):.3f}", date=C.fmt_date(row[1])))
            except Exception:  # noqa: BLE001
                pass
            h = by_sym.get(sym, {})
            st.write(C.TIER_LINE.format(tier=h.get("tier_word") or C.tier_word("ALPHA")))
            st.write(C.WHY_VERDICT_LINE.format(
                why=h.get("reason") or "within its target band."))


def _render_split_lines(portfolio) -> None:
    """Block 4: how your money is split (invested only)."""
    from quant.config import ACTIVE_MAX, BETS_MAX, LONG_TERM_MIN
    from quant.portfolio.tier_manager import load_tiers_safe, tier_map

    st.subheader(C.SEC_HOW_SPLIT)
    if portfolio is None or portfolio.empty:
        st.caption(C.EMPTY_TIER)
        return
    try:
        tiers_df, _ = load_tiers_safe()
        tmap = tier_map(tiers_df)
    except Exception:  # noqa: BLE001
        tmap = {}
    total = float(portfolio["Current_Value_EUR"].sum()) or 1.0
    buckets = {"FORTRESS": 0.0, "ALPHA": 0.0, "SPECULATIVE": 0.0}
    for _, r in portfolio.iterrows():
        tier = str(tmap.get(str(r["Symbol"]), "ALPHA")).upper()
        if tier in buckets:
            buckets[tier] += float(r.get("Current_Value_EUR", 0) or 0)
    rule_long = C.SPLIT_RULE_LONG.format(min=f"{LONG_TERM_MIN * 100:.0f}")
    rule_active = C.SPLIT_RULE_ACTIVE.format(max=f"{ACTIVE_MAX * 100:.0f}")
    rule_bets = C.SPLIT_RULE_BETS.format(max=f"{BETS_MAX * 100:.0f}")
    lines = [
        (C.TIER_FORTRESS, buckets["FORTRESS"], rule_long,
         buckets["FORTRESS"] / total >= LONG_TERM_MIN),
        (C.TIER_ALPHA, buckets["ALPHA"], rule_active,
         buckets["ALPHA"] / total <= ACTIVE_MAX),
        (C.TIER_SPECULATIVE, buckets["SPECULATIVE"], rule_bets,
         buckets["SPECULATIVE"] / total <= BETS_MAX),
    ]
    for label, value, rule, ok in lines:
        status = C.SPLIT_OK if ok else C.SPLIT_OVER
        st.write(f"{label}: {value:.0f} EUR, {value / total * 100:.0f} percent of "
                 f"invested. Rule: {rule}. {status}")


def _render_quick_events(portfolio) -> None:
    """Block 7: record a buy, sell, or dividend in five seconds."""
    st.subheader(C.SEC_QUICK_EVENTS)
    symbols = list(portfolio["Symbol"].astype(str)) if (
        portfolio is not None and not portfolio.empty) else []
    if not symbols:
        st.caption(C.EMPTY_TIER)
        return
    cols = st.columns(4)
    kind = cols[0].selectbox("Type", ["buy", "sell", "dividend"], key="qe_type")
    symbol = cols[1].selectbox("Symbol", symbols, key="qe_symbol")
    amount = cols[2].number_input("Amount (EUR)", min_value=0.0, value=0.0,
                                  step=10.0, key="qe_amount")
    when = cols[3].date_input("Date", value=_date.today(), key="qe_date")
    if st.button("Record", key="qe_save"):
        try:
            from quant.data.database import connect_with_retry
            from quant.engine import flows

            conn = connect_with_retry()
            try:
                flows.record_flow(conn, when, kind, amount, symbol)
            finally:
                conn.close()
            st.success(C.QUICK_EVENT_SAVED.format(
                type=kind, amount=f"{amount:.0f}", name=symbol,
                date=C.fmt_date(when)))
        except Exception:  # noqa: BLE001
            st.warning("Could not record the event. Try again.")
    st.caption(C.QUICK_EVENT_CASH_NOTE)


def page_portfolio() -> None:
    """Render the My holdings page (v10.7.1, Section 10.2)."""
    st.title(C.PAGE_PORTFOLIO)
    render_onboarding_wizard()
    st.write(C.HELP_BROKER_VALUES)

    # Autocomplete add-row (A5): the input says what it does; selecting a match
    # appends an empty-value row and shows one helper line. No silent add.
    query = st.text_input(
        C.PLACEHOLDER_ADD_HOLDING, key="add_q",
        placeholder=C.PLACEHOLDER_ADD_HOLDING, label_visibility="collapsed",
    )
    if query:
        results = search(load_index(), query, 10)
        # H3.5 (F5): also offer discovery-universe instruments (universe_master
        # rows not yet tracked); selecting one adds a held row -> enters W on
        # save, fetched at the next update.
        disc = discovery_candidates(query, 10)
        sym_by_label: dict[str, str] = {}
        labels: list[str] = []
        for r in results:
            sym_by_label[r["label"]] = r["symbol"]
            labels.append(r["label"])
        for r in disc:
            lbl = C.NOT_TRACKED_LABEL.format(label=r["label"])
            if lbl not in sym_by_label:
                sym_by_label[lbl] = r["symbol"]
                labels.append(lbl)
        if labels:
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

    # v10.7.1 My holdings blocks (Section 10.2).
    from quant.portfolio.portfolio import load_portfolio_with_tiers
    tiered = load_portfolio_with_tiers()
    holdings = read_actions()
    render_tier_assignment_alerts()
    render_trade_limit_warning()

    # Block 2 + 3: the table and the per-holding expanders.
    _render_holdings_table(portfolio, holdings)
    _render_holding_expanders(portfolio, holdings)

    # Block 4: how your money is split.
    _render_split_lines(portfolio)

    # Block 5: if you need cash now.
    render_emergency_liquidity(tiered)

    # Block 6: losses you can use to lower tax.
    render_tax_loss_alerts(tiered)

    # Block 7: quick events.
    _render_quick_events(portfolio)

    # Block 8: Account.
    st.subheader("Account")
    account = load_account()
    cash_val = account.cash_eur if account.cash_is_set else 0.0
    cash = st.number_input("Operational cash (EUR)", min_value=0.0,
                           value=float(cash_val), step=10.0)
    st.caption(C.OPERATIONAL_CASH_LINE.format(
        amount=f"{cash:.0f}", date=C.fmt_date(_date.today()), apy="2.5"))
    profile = st.radio(
        "Risk profile", list(RISK_PROFILES.keys()),
        index=list(RISK_PROFILES.keys()).index(account.risk_profile),
        captions=[RISK_PROFILE_DESCRIPTIONS[p] for p in RISK_PROFILES],
        format_func=lambda p: p.capitalize(),
    )
    savings_day = st.number_input(
        C.SAVINGS_DAY_LABEL, min_value=1, max_value=28,
        value=int(account.savings_plan_day or 1), step=1)
    st.caption(C.SAVINGS_DAY_CAPTION)
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
        save_account(AccountState(account.base_currency, float(cash), profile, True,
                                  int(savings_day)))
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
        stop = threading.Event()

        def _beat() -> None:
            while not stop.wait(_HEARTBEAT_INTERVAL):
                try:
                    runner._write_heartbeat()
                except Exception:  # noqa: BLE001
                    pass

        beat = threading.Thread(target=_beat, daemon=True)
        beat.start()
        try:
            res = use() if runner_fn is not None else use(command)
        finally:
            stop.set()
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

def _sentiment_available() -> bool:
    """True when the FinBERT stack (transformers + torch) is importable."""
    try:
        import importlib.util

        return (importlib.util.find_spec("transformers") is not None
                and importlib.util.find_spec("torch") is not None)
    except Exception:  # noqa: BLE001
        return False


def _candidate_list() -> dict:
    """Group funnel survivors into long-term / active / small-bets candidates.

    v10.7.1 (Section 10.4): the SAME list feeds Find investments and the Monthly
    decision "New ideas this month" section. Best-effort; never raises.
    """
    from quant.config import ACTIVE_TACT_MIN, SPARPLAN_STRUCT_MIN

    out = {"long": [], "active": [], "bets": []}
    try:
        surv = q("SELECT symbol FROM funnel_survivors ORDER BY score DESC LIMIT 30")
    except Exception:  # noqa: BLE001
        return out
    if surv is None or surv.empty:
        return out
    for sym in surv["symbol"].astype(str):
        sc = read_scores(sym)
        struct = sc.get("structural_grade")
        tact = sc.get("tactical_grade")
        name = label_for(sym, sym)
        if struct is not None and float(struct) >= SPARPLAN_STRUCT_MIN:
            out["long"].append({"symbol": sym, "name": name, "structure": struct})
        elif tact is not None and float(tact) >= ACTIVE_TACT_MIN:
            out["active"].append({"symbol": sym, "name": name,
                                  "structure": struct, "tactics": tact})
    return out


def _render_candidates() -> None:
    """Three grouped candidate sections (v10.7.1, Section 10.4)."""
    from quant.config import ACTIVE_MAX, BETS_MAX
    from quant.portfolio.tier_manager import load_tiers_safe, tier_map

    cands = _candidate_list()
    try:
        tiers_df, _ = load_tiers_safe()
        tmap = tier_map(tiers_df)
    except Exception:  # noqa: BLE001
        tmap = {}
    portfolio = load_portfolio()
    total = float(portfolio["Current_Value_EUR"].sum()) if (
        portfolio is not None and not portfolio.empty) else 0.0
    active_used = 0.0
    bets_used = 0.0
    if total > 0:
        for _, r in portfolio.iterrows():
            tier = str(tmap.get(str(r["Symbol"]), "ALPHA")).upper()
            value = float(r.get("Current_Value_EUR", 0) or 0)
            if tier == "ALPHA":
                active_used += value
            elif tier == "SPECULATIVE":
                bets_used += value

    st.subheader(C.SEC_CAND_LONG)
    if cands["long"]:
        for c in cands["long"][:3]:
            st.write(f"{c['name']} ({c['symbol']})")
            st.caption(f"Structure {float(c['structure']):.0f}. {C.CAND_LONG_REASON}")
    else:
        st.caption(C.CAND_EMPTY.format(section="long-term"))

    st.subheader(C.SEC_CAND_ACTIVE)
    if cands["active"]:
        for c in cands["active"][:3]:
            st.write(f"{c['name']} ({c['symbol']})")
            st.caption(f"Structure {float(c['structure'] or 0):.0f}, "
                       f"tactics {float(c['tactics'] or 0):.0f}. "
                       + C.CAND_ACTIVE_LIMIT.format(
                           used=f"{active_used / total * 100:.0f}" if total else "0",
                           max=f"{ACTIVE_MAX * 100:.0f}"))
    else:
        st.caption(C.CAND_EMPTY.format(section="active"))

    st.subheader(C.SEC_CAND_BETS)
    if cands["bets"]:
        for c in cands["bets"][:3]:
            st.write(f"{c['name']} ({c['symbol']})")
            st.caption(C.CAND_BETS_LIMIT.format(
                used=f"{bets_used / total * 100:.0f}" if total else "0",
                max=f"{BETS_MAX * 100:.0f}"))
    else:
        st.caption(C.CAND_EMPTY.format(section="small bets"))


def page_explore() -> None:
    """Render the Find investments page."""
    st.title(C.PAGE_EXPLORE)
    _render_candidates()

    query = st.text_input("Search a name, symbol or ISIN", key="ex_q")
    matches = search(load_index(), query, 10) if query else []
    if not query:
        st.caption(C.SEARCH_HELPER)
        return
    if not matches:
        st.info(C.EMPTY_NO_MATCHES.format(query=query))
        # H3.5 (F5): close the discovery loop — name a universe_master match.
        _disc = discovery_candidates(query, 1)
        if _disc:
            _d = _disc[0]
            st.caption(C.DISCOVERY_NOT_TRACKED.format(
                label=label_for(_d.get("display_name") or _d.get("name"), _d["symbol"])))
        return
    options = [r["label"] for r in matches]
    choice = st.selectbox("Matches", options, key="ex_choice")
    symbol = next(r["symbol"] for r in matches if r["label"] == choice)

    broker = resolve_broker(symbol)
    # H3.5 (F1-F3): title/class/subtitle come from ONE helper the doctor probes,
    # so the page and the probe can never diverge. Registry values win over CSV.
    card = explore_card_fields(symbol)
    name = card["name"]
    cls = card["class"]

    # F-series (bug 4): a bare-symbol fallback is not a name — render an honest
    # empty state instead of showing the ticker as a title.
    if card["has_details"]:
        st.subheader(name)
        st.caption(symbol)
        if card["subtitle"]:
            st.caption(card["subtitle"])
    else:
        st.info(C.EXPLORE_NO_DETAILS.format(symbol=symbol))
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
    # H3.8 (M9): ONE catalogue as-of line derived from the ok review; no variants
    # (no "From the review of" preamble).
    ok_review = latest_review(ok_only=True)
    ok_ts = ok_review.get("review_ts")
    if ok_ts:
        st.caption(C.SCORES_AS_OF.format(date=C.fmt_review_ts(ok_ts)))
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

    # News and filings (on-demand, 24 h cache). H3.4: weekday dates, cap at 5
    # visible rows with an "Earlier items" expander, and sentiment honesty.
    st.subheader(C.SEC_NEWS)
    _items = load_news(symbol)
    if not _items:
        st.info(C.EMPTY_NO_NEWS.format(name=name))
    else:
        _has_senti = _sentiment_available()
        _visible, _rest = _items[:5], _items[5:]

        def _row(it: dict) -> str:
            when = (C.fmt_weekday_date(it.get("published_at"))
                    or C.fmt_date(it.get("published_at")))
            line = f"{when} · {it.get('source', '')} · {it.get('headline', '')}"
            # H3.6 (N3): the word is shown only for a real model score, never a
            # silent default.
            if _has_senti and it.get("scorer") == "model":
                senti = ("positive" if it.get("score", 0) > 0
                         else "negative" if it.get("score", 0) < 0 else "neutral")
                line += f" · {senti}"
            return line

        for it in _visible:
            st.write(_row(it))
        if _rest:
            with st.expander(C.NEWS_EARLIER.format(n=len(_rest))):
                for it in _rest:
                    st.write(_row(it))
        if not _has_senti:
            st.caption(C.SENTIMENT_UNAVAILABLE)

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

def _dedupe_history_by_day(history):
    """v10.7.0 (B6): one report entry per trading day, keeping the latest."""
    if history is None or history.empty:
        return history
    df = history.copy()
    df["_day"] = pd.to_datetime(df["review_ts"], errors="coerce").dt.date
    df = df.dropna(subset=["_day"])
    if df.empty:
        return history
    df = df.sort_values("review_ts").groupby("_day", as_index=False).last()
    return df.drop(columns=["_day"])


def page_settings() -> None:
    """Render the Settings page."""
    st.title(C.PAGE_SETTINGS)

    # Data status (A2): no placeholder sentence. The full sentence renders only
    # when all three facts exist; otherwise a genuine missing-data empty state.
    st.subheader(C.SEC_DATA_STATUS)
    instruments = q("SELECT COUNT(DISTINCT Symbol) AS n FROM market_history")
    m = int(instruments["n"].iloc[0]) if not instruments.empty else 0
    bar = latest_bar_date()
    history = read_history()
    # H3.7 (L2): refreshed-at comes from the update artifact; count/date are live.
    us = read_update_state()
    if m > 0 and bar:
        if us.get("ts"):
            st.write(f"{m} instruments, prices through {C.fmt_date(bar)}, "
                     f"refreshed {C.fmt_ts(us.get('ts'))}.")
        else:
            st.write(f"{m} instruments, prices through {C.fmt_date(bar)}.")
    elif us.get("instruments") and us.get("ts"):
        # H3.8 (M6): a transient locked read must not erase state.
        st.write(f"{us['instruments']} instruments, prices through "
                 f"{C.fmt_date(us.get('prices_through'))}, "
                 f"refreshed {C.fmt_ts(us.get('ts'))}.")
    else:
        st.info(C.EMPTY_NO_MARKET_DATA)
    _run_operation(C.BTN_REFRESH, C.BTN_REFRESHING, runner.REFRESH, "refresh")
    st.caption(C.HELP_REVIEW_CADENCE)

    # Report history (last ten). The value-chart sentence belongs to Overview (A2).
    st.subheader(C.SEC_REPORT_HISTORY)
    # H3.7 (L3): drop rows newer than the last OK review (legacy phantom rows).
    _ok_cutoff = latest_ok_review_ts()
    if _ok_cutoff is not None and not history.empty:
        _ts = pd.to_datetime(history["review_ts"], errors="coerce")
        # H3.8 (M7): legacy NaT timestamps are kept; only rows newer than the
        # last ok review (phantoms) are hidden.
        history = history[(_ts <= _ok_cutoff) | _ts.isna()]
    # v10.7.0 (B6): one entry per trading day, keeping the latest per day.
    history = _dedupe_history_by_day(history)
    if history.empty:
        st.info(C.EMPTY_NO_REVIEWS)
    else:
        for _, r in history.tail(10)[::-1].iterrows():
            st.write(f"{C.fmt_ts(r['review_ts'])} - {C.fmt_eur(r['value_eur'])} - "
                     f"{C.fmt_eur(r['pnl_eur'])}")

    # Automation (v10.7.0, Section 10.6).
    st.subheader(C.SEC_AUTOMATION)
    try:
        from quant.engine import scheduler as _sched

        st.write(f"Scheduler: {_sched.status_line()}")
    except Exception:  # noqa: BLE001
        st.write("Scheduler: not installed; run quant schedule")
    try:
        from quant.engine import notify as _notify

        st.write(f"Notifications: {_notify.status_line()}")
    except Exception:  # noqa: BLE001
        st.write("Notifications: off; run quant notify-setup")
    try:
        _size_mb = os.path.getsize(paths.DB_FILE) / (1024 * 1024)
        st.write(f"Database size: {_size_mb:.2f} MB")
    except Exception:  # noqa: BLE001
        pass
    try:
        from quant.data.database import read_only_connection
        from quant.engine import alerts as _alerts
        from quant.engine import valuation as _valuation

        with read_only_connection() as _conn:
            st.write(_alerts.advice_record_line(_conn))
            _sync_days = _valuation.days_since_last_sync(_conn)
        st.write(f"Last broker sync: "
                 f"{_sync_days if _sync_days is not None else 'never'} days ago")
    except Exception:  # noqa: BLE001
        pass
    # v10.7.2 (Part 3.1): the backup line, gentle at most once per week.
    try:
        from quant.engine import backup as _backup

        _bline = _backup.ui_backup_line()
        if _bline:
            st.write(_bline)
    except Exception:  # noqa: BLE001
        pass

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
        _rev = latest_review()
        if _rev.get("review_status") == "failed":
            problems.append(_rev.get("error") or C.HEALTH_REVIEW_FAILED)
    except Exception:  # noqa: BLE001
        pass
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
    cooldown = [h for h in holdings if str(h.get("status", "")).startswith("Cooldown until")]
    if cooldown:
        for when in {h.get("cooldown_until") for h in cooldown}:
            n = sum(1 for h in cooldown if h.get("cooldown_until") == when)
            tpl = C.FOOTNOTE_COOLDOWN_ONE if n == 1 else C.FOOTNOTE_COOLDOWN
            st.caption(tpl.format(n=n, date=C.fmt_date(when)))


# ── Value chart helpers (spec 3.1; pure where possible) ───────────────────────
_RANGE_DAYS = {"1M": 31, "3M": 92, "1Y": 365, "Max": None}


def _filter_range(df: pd.DataFrame, rng: str) -> pd.DataFrame:
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


def _range_annotation(df: pd.DataFrame, growth: bool = False) -> str:
    """`+4.2% since 1 Jun 2026 (34.80 EUR)` from the range endpoints.

    H3.8 (M3): Growth mode is percent-only (no EUR parenthetical).
    """
    if df.empty or len(df) < 2:
        return ""
    first, last = df.iloc[0], df.iloc[-1]
    if not first["value_eur"]:
        return ""
    pct = (last["value_eur"] / first["value_eur"] - 1.0) * 100.0
    sign = "+" if pct >= 0 else ""
    if growth:
        return C.CHART_SINCE_PCT.format(sign=sign, pct=f"{pct:.1f}",
                                        date=C.fmt_date(first["review_ts"]))
    abs_ = last["value_eur"] - first["value_eur"]
    return C.CHART_SINCE.format(sign=sign, pct=f"{pct:.1f}",
                                date=C.fmt_date(first["review_ts"]), amount=f"{abs_:.2f}")


def _axis_tickformat(df) -> str:
    """H3.7 (L5): sub-3-day ranges label by hh:mm, else by day."""
    try:
        span = (df["review_ts"].max() - df["review_ts"].min()).days
    except Exception:  # noqa: BLE001
        return "%d %b"
    return "%H:%M" if span < 3 else "%d %b"


def _annotation_kwargs(text: str) -> dict:
    """H3.7 (L5): the in-plot annotation, in the top margin on a white box."""
    return dict(xref="paper", yref="paper", x=0.0, y=1.0, xanchor="left",
                yanchor="bottom", text=text, showarrow=False, align="left",
                font=dict(size=12, color="#1F2937"),
                bgcolor="rgba(255,255,255,0.85)")


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
    # v10.7.0 (B8): one clean line, no area fill.
    fig.add_trace(go.Scatter(
        x=df["review_ts"], y=port, mode="lines", name="Portfolio",
        line=dict(width=2, color="#1F3B73"), fill=None))

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
    ann = _range_annotation(df, growth)
    if ann:
        fig.add_annotation(**_annotation_kwargs(ann))     # H3.7 (L5)
    # v10.7.0 (B8): the axis never starts at zero; pad the visible range.
    vals = [float(v) for v in port if v is not None]
    yaxis = {}
    if vals:
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.1 or max(1.0, abs(hi) * 0.01)
        yaxis = dict(range=[lo - pad, hi + pad])
    fig.update_layout(height=280, margin=dict(l=8, r=8, t=30, b=8),
                      xaxis=dict(tickformat=_axis_tickformat(df)), yaxis=yaxis,
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                                  xanchor="left", x=0))
    st.plotly_chart(fig, width="stretch")


_HEARTBEAT_INTERVAL = 30.0


# ── v10.7.0: Monthly decision page (Section 8) ────────────────────────────────

def _month_key() -> str:
    """The current month key, e.g. '2026-11'."""
    return _date.today().strftime("%Y-%m")


def _month_name() -> str:
    """The current month name, e.g. 'November'."""
    return _date.today().strftime("%B")


def _monthly_holdings() -> list[dict]:
    """Build holding dicts for the allocator (value, tier, weights)."""
    from quant.config import LONG_TERM_MIN
    from quant.portfolio.portfolio import load_portfolio
    from quant.portfolio.tier_manager import load_tiers_safe, tier_map

    try:
        df = load_portfolio()
        tiers_df, _ = load_tiers_safe()
        tmap = tier_map(tiers_df)
    except Exception:  # noqa: BLE001
        return []
    if df is None or df.empty:
        return []
    total = float(df["Current_Value_EUR"].sum()) or 1.0
    fortress = [s for s in df["Symbol"].astype(str)
                if str(tmap.get(s, "ALPHA")).upper() == "FORTRESS"]
    n_fortress = max(1, len(fortress))
    out: list[dict] = []
    for _, row in df.iterrows():
        symbol = str(row["Symbol"])
        tier = str(tmap.get(symbol, "ALPHA")).upper()
        value = float(row.get("Current_Value_EUR", 0) or 0)
        target = (LONG_TERM_MIN / n_fortress) if tier == "FORTRESS" else 0.0
        out.append({
            "symbol": symbol, "name": symbol, "tier": tier, "value_eur": value,
            "current_weight": value / total, "target_weight": target,
            "conviction": 0.0,
        })
    return out


def _approve_plan(month: str, budget: float, legs: list[dict]) -> None:
    """Store the plan and write planned auto-flows (write connection)."""
    try:
        from quant.data.database import connect_with_retry
        from quant.engine import flows, plans

        conn = connect_with_retry()
        try:
            plans.save_plan(conn, month, budget, legs,
                            approved_date=_date.today(), execution_date=_date.today())
            buy_legs = [leg for leg in legs
                        if leg.get("kind") in ("long_term", "active", "bet")
                        and leg.get("symbol")]
            flows.write_planned_sparplan_flows(conn, month, buy_legs, _date.today())
        finally:
            conn.close()
        st.success(C.MONTHLY_APPROVED)
    except Exception:  # noqa: BLE001
        st.warning("Could not save the plan. Try again.")


def page_monthly() -> None:
    """Render the Monthly decision page (v10.7.0, Section 8)."""
    st.title(C.MONTHLY_TITLE.format(month=_month_name()))
    from quant.data.database import read_only_connection
    from quant.engine import allocator, plans

    month = _month_key()
    try:
        with read_only_connection() as conn:
            plan = plans.load_plan(conn, month)
    except Exception:  # noqa: BLE001
        plan = None

    if plan:
        st.write(C.MONTHLY_STATUS_APPROVED.format(
            date=C.fmt_date(plan.get("approved_date"))))
    else:
        st.write(C.MONTHLY_STATUS_NOT_APPROVED)

    holdings = _monthly_holdings()
    default_budget = float(plan["budget_eur"]) if plan else 200.0
    budget = st.number_input(C.MONTHLY_BUDGET_LABEL, min_value=0.0,
                             value=default_budget, step=10.0, key="monthly_budget")
    legs = allocator.allocate(budget, holdings, regime=None, candidates=[],
                              bets_enabled=False)

    st.write(C.MONTHLY_SPLIT_HEADER)
    for leg in legs:
        st.write(C.MONTHLY_LEG_LINE.format(
            amount=f"{leg['amount_eur']:.0f}", name=leg["name"],
            symbol=leg["symbol"] or "cash", kind=leg["kind"]))
        st.caption(C.MONTHLY_LEG_REASON.format(
            reason=leg["reason"], fee=f"{leg['fee_eur']:.0f}"))

    # New ideas this month (v10.7.1: live candidates + regime).
    st.write(C.MONTHLY_NEW_IDEAS)
    _cands = _candidate_list()
    _shown = False
    for c in _cands["long"][:1]:
        st.caption(C.MONTHLY_CANDIDATE_LINE.format(
            name=c["name"], detail=f"structure {float(c['structure']):.0f}, "
            f"long-term candidate."))
        _shown = True
    for c in _cands["active"][:1]:
        st.caption(C.MONTHLY_CANDIDATE_LINE.format(
            name=c["name"], detail=f"tactics {float(c['tactics'] or 0):.0f}, "
            f"active candidate."))
        _shown = True
    if not _shown:
        st.caption(C.CAND_EMPTY.format(section="new ideas"))

    cols = st.columns(2)
    if cols[0].button(C.BTN_APPROVE, key="monthly_approve"):
        _approve_plan(month, budget, legs)
    if cols[1].button(C.BTN_CHANGE_SPLIT, key="monthly_change"):
        st.session_state["_monthly_edit"] = True

    # Entering actuals (Section 8.4).
    st.subheader(C.MONTHLY_ACTUALS_HEADER)
    buy_legs = [leg for leg in legs
                if leg.get("kind") in ("long_term", "active", "bet") and leg.get("symbol")]
    actuals: list[dict] = []
    for leg in buy_legs:
        amount = st.number_input(
            f"{leg['name']} ({leg['symbol']})",
            min_value=0.0, value=float(leg["amount_eur"]), step=5.0,
            key=f"actual_{leg['symbol']}")
        actuals.append({"symbol": leg["symbol"], "amount_eur": amount,
                        "date": _date.today()})
    if st.button(C.BTN_SAVE_ACTUALS, key="monthly_save_actuals"):
        try:
            from quant.data.database import connect_with_retry
            from quant.engine import plans as plans_mod

            conn = connect_with_retry()
            try:
                recon = plans_mod.enter_actuals(conn, month, actuals)
            finally:
                conn.close()
            for row in recon:
                st.write(plans_mod.reconciliation_line(row))
            st.success(C.MONTHLY_ACTUALS_SAVED)
        except Exception:  # noqa: BLE001
            st.warning("Could not save actuals. Try again.")
