"""
pages.py — the five pages of the one workflow (v10.8.2, section 4).

Portfolio (home), Update holdings, History, Full analysis, Settings. Each page is
lean and reads the ONE sources: positions/actions, the value series, the decision
list, the ledger and the tax engine. No cash anywhere.
"""
from __future__ import annotations

from datetime import date as _date

import pandas as pd
import streamlit as st

from quant import paths
from quant.engine.decisions import actionable_items, format_item_line
from quant.ui import copy as C
from quant.ui import render as R


def _holdings() -> list[dict]:
    try:
        return R.read_actions()
    except Exception:  # noqa: BLE001
        return []


def _total_value(holdings: list[dict], portfolio) -> float:
    if holdings:
        return sum(float(h.get("value_eur") or 0) for h in holdings)
    if portfolio is not None and not portfolio.empty:
        return float(portfolio["Current_Value_EUR"].sum())
    return 0.0


def _range_change(series) -> tuple[float, float]:
    """Signed EUR and percent change over the series (flow-adjusted by design)."""
    if series is None or len(series) < 2:
        return 0.0, 0.0
    first = float(series.iloc[0]["value_eur"])
    last = float(series.iloc[-1]["value_eur"])
    if first == 0:
        return 0.0, 0.0
    return last - first, (last / first - 1.0) * 100.0


def _plan_day_reminder() -> None:
    """After the plan day, remind the user to check the savings plan (B6)."""
    from quant.portfolio.account import load_account

    day = load_account().savings_plan_day
    if not day:
        return
    if _date.today().day > int(day):
        st.info(C.PLAN_DAY_REMINDER.format(day=day))


def _current_plans(portfolio) -> dict:
    """The saved Plan_EUR_month per symbol (empty when the column is absent)."""
    if portfolio is None or portfolio.empty or "Plan_EUR_month" not in portfolio.columns:
        return {}
    out: dict[str, float] = {}
    for _, r in portfolio.iterrows():
        value = r.get("Plan_EUR_month")
        if value is not None and not pd.isna(value):
            out[str(r["Symbol"])] = float(value)
    return out


def _home_items(holdings: list[dict]) -> list[dict]:
    """The ONE list: one item per holding that needs action (never 'no action')."""
    items: list[dict] = []
    for h in holdings:
        status = str(h.get("status") or "")
        sym = str(h.get("symbol") or "")
        name = h.get("name") or sym
        label = R.label_for(name, sym)
        if status.startswith("Sell"):
            items.append({"group": C.DECISION_GROUP_RECOMMENDED, "verb": "sell_part",
                          "label": label, "amount_eur": h.get("amount_eur"),
                          "reason": "It is above its target.", "symbol": sym})
        elif status.startswith("Buy") or status.startswith("Add"):
            items.append({"group": C.DECISION_GROUP_RECOMMENDED, "verb": "buy",
                          "label": label, "amount_eur": h.get("amount_eur"),
                          "reason": "It is below its target.", "symbol": sym})
        elif h.get("blocked"):
            items.append({"group": C.DECISION_GROUP_INPUT, "verb": "blocked",
                          "label": label, "amount_eur": None,
                          "reason": C.NEEDS_ATTENTION_ISIN.format(symbol=sym),
                          "symbol": sym})
    return actionable_items(items)


def page_home() -> None:
    """Portfolio home (section 4.1): value header, chart, what to do, table."""
    st.title("Portfolio")
    holdings = _holdings()
    portfolio = R.load_portfolio()
    total = _total_value(holdings, portfolio)
    series = R.read_value_series()
    change_eur, change_pct = _range_change(series)

    st.metric("Total value", C.fmt_eur_whole(total),
              delta=f"{change_eur:+,.0f} EUR ({change_pct:+.1f}%)")
    st.caption(f"Prices as of {C.fmt_date(R.latest_bar_date())} close.")

    R._render_value_chart(R.read_history(), holdings)

    _plan_day_reminder()

    st.subheader(C.SEC_WHAT_TO_DO)
    items = _home_items(holdings)
    if items:
        for it in items:
            st.write(format_item_line(it))
            st.caption(it["reason"])
    else:
        st.write("Nothing to do. Next check after the next market close.")

    st.subheader("Holdings")
    rows = []
    for h in holdings:
        rows.append({
            "Holding": R.label_for(h.get("name") or h["symbol"], h["symbol"]),
            "Value": C.fmt_eur(h.get("value_eur") or 0),
            "Share vs target": f"{h.get('current_weight', '')} / {h.get('target_weight', '')}",
            "Status": h.get("status", ""),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("Add what you own in Update holdings.")

    _render_invest_block(holdings, portfolio)

    c1, c2 = st.columns(2)
    if c1.button("Update holdings", width="stretch"):
        st.switch_page("pages/update.py")
    if c2.button("Full analysis", width="stretch"):
        st.switch_page("pages/analysis.py")


def _render_invest_block(holdings: list[dict], portfolio) -> None:
    """The invest block (v10.8.2, B2): one amount, one cadence, a suggestion."""
    st.subheader(C.SEC_INVEST)
    amount = st.number_input(C.INVEST_AMOUNT_LABEL, min_value=0.0, value=0.0,
                             step=10.0, key="invest_amount")
    cadence = st.radio("Cadence", [C.INVEST_CADENCE_MONTHLY, C.INVEST_CADENCE_ONCE],
                       horizontal=True, key="invest_cadence",
                       label_visibility="collapsed")
    if amount <= 0:
        return
    from quant.engine.savings_plan import propose_once, propose_plans

    regime = R.read_regime().get("label")
    if cadence == C.INVEST_CADENCE_MONTHLY:
        out = propose_plans(amount, holdings, _current_plans(portfolio), regime)
        if out["already_fit"]:
            st.write(C.INVEST_ALREADY_FIT)
        rows = [{
            "Holding": R.label_for(r["name"], r["symbol"]),
            C.INVEST_PLAN_NOW: C.fmt_eur(r["current"]),
            C.INVEST_SUGGESTED: C.fmt_eur(r["suggested"]),
            C.INVEST_CHANGE: C.fmt_eur(r["change"]),
        } for r in out["rows"]]
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        orders = propose_once(amount, holdings, regime, min_order_eur=25.0)
        if orders:
            rows = [{"Holding": R.label_for(o["name"], o["symbol"]),
                     "Amount": C.fmt_eur(o["amount_eur"])} for o in orders]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.caption(C.INVEST_ONCE_NOTE)
        else:
            st.write(C.INVEST_ONCE_EMPTY)


def page_update() -> None:
    """Update holdings (section 4.2): the table, Review, Confirm, Cancel."""
    st.title("Update holdings")
    portfolio = R.load_portfolio()
    sync = R._last_sync_date()
    if sync:
        st.caption(f"Last saved {C.fmt_date(sync)}.")
        try:
            days = (_date.today() - _date.fromisoformat(str(sync)[:10])).days
            if days > 35:
                st.warning("These numbers may be out of date.")
        except Exception:  # noqa: BLE001
            pass
    else:
        st.caption("Add what you own. Name or ticker, then the three numbers from "
                   "Trade Republic.")

    edited = R._render_broker_editor(portfolio)

    if st.button("Review changes", type="primary", width="stretch"):
        from quant.engine.diff import diff_tables

        old = portfolio.to_dict("records") if not portfolio.empty else []
        new = edited.drop(columns=["Name"], errors="ignore").to_dict("records")
        result = diff_tables(old, new, plans=_current_plans(portfolio))
        st.session_state["_diff"] = result

    result = st.session_state.get("_diff")
    if result is not None:
        summary = result["summary"]
        if not summary["changed"]:
            st.write("No changes.")
        else:
            st.write(f"Bought {summary['bought_eur']:.0f} EUR, "
                     f"sold {summary['sold_eur']:.0f} EUR, "
                     f"realized {summary['realized_eur']:+.2f} EUR.")
            for ch in result["changes"]:
                label = (C.DIFF_BOUGHT_SAVINGS if ch.get("savings_plan")
                         else ch["kind"])
                st.caption(f"{label} {ch['symbol']}")
        c1, c2 = st.columns(2)
        if c1.button("Confirm and save", type="primary", width="stretch"):
            from quant.engine.confirm import confirm_save
            from quant.portfolio.editor import validate_positions

            cleaned, _w, errors = validate_positions(
                edited.drop(columns=["Name"], errors="ignore"))
            if errors:
                for e in errors:
                    st.error(e)
            else:
                res = confirm_save(cleaned, result)
                if res["ok"]:
                    st.session_state.pop("_diff", None)
                    st.success("Saved.")
                    st.switch_page("pages/portfolio.py")
                else:
                    st.error("Could not save. Nothing was written.")
        if c2.button("Cancel", width="stretch"):
            st.session_state.pop("_diff", None)
            st.rerun()

    with st.expander("Change type"):
        st.caption("Instrument type is assigned automatically; override it here.")


def page_history() -> None:
    """History (section 4.3): changes log + three yearly numbers + CSV."""
    st.title("History")
    st.caption("Dividends and interest are not tracked.")
    try:
        from quant.data.database import read_only_connection

        with read_only_connection() as conn:
            rows = conn.execute(
                "SELECT date, symbol, action, amount_eur, realized_pnl_eur "
                "FROM trades ORDER BY date DESC LIMIT 200").fetchall()
    except Exception:  # noqa: BLE001
        rows = []
    if rows:
        df = pd.DataFrame(rows, columns=["Date", "Name (TICKER)", "Bought or Sold",
                                         "Amount EUR", "Realized EUR"])
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button("Export this year (CSV)", df.to_csv(index=False),
                           file_name="history.csv", mime="text/csv")
    else:
        st.info("No changes recorded yet.")

    st.subheader("This year")
    try:
        from quant.portfolio.tax_accounting import calculate_yearly_tax_summary

        summary = calculate_yearly_tax_summary(_date.today().year)
        st.write(f"Realized gain or loss (estimated): "
                 f"{C.fmt_eur(summary.get('realized_gains', 0))}")
        st.write(f"Tax-free allowance remaining: "
                 f"{C.fmt_eur(summary.get('allowance_remaining', 0))}")
        st.write(f"Rough tax estimate: {C.fmt_eur(summary.get('estimated_tax', 0))}")
    except Exception:  # noqa: BLE001
        st.caption("Tax figures unavailable.")
    st.radio("Filing status", ["Single", "Married"], horizontal=True)


def page_analysis() -> None:
    """Full analysis (section 4.4): read-only tabs."""
    st.title("Full analysis")
    tabs = st.tabs(["Holdings analysis", "Universe scan", "Ideas", "Market",
                    "Run details"])
    with tabs[0]:
        holdings = _holdings()
        if holdings:
            st.dataframe(pd.DataFrame(holdings), width="stretch", hide_index=True)
        else:
            st.info("No holdings yet.")
    with tabs[1]:
        df = R.q("SELECT symbol, instrument_class, universe_status FROM asset_registry")
        st.dataframe(df, width="stretch", hide_index=True) if not df.empty else \
            st.info("No scan data yet.")
    with tabs[2]:
        df = R.q("SELECT symbol, score FROM funnel_survivors ORDER BY score DESC LIMIT 50")
        st.dataframe(df, width="stretch", hide_index=True) if not df.empty else \
            st.info("No ideas yet.")
    with tabs[3]:
        regime = R.read_regime()
        st.write(f"Market trend: {regime.get('label', 'unknown')} "
                 f"({regime.get('confidence', 'unknown')} confidence).")
        try:
            from quant.engine.news_pillar import model_available

            if not model_available():
                st.caption(C.MARKET_NEWS_OFF)
        except Exception:  # noqa: BLE001
            st.caption(C.MARKET_NEWS_OFF)
    with tabs[4]:
        st.write(f"Last price update: {C.fmt_date(R.latest_bar_date())}")
        st.write(f"Database size: {_db_size_mb():.2f} MB")


def _db_size_mb() -> float:
    import os

    try:
        return os.path.getsize(paths.DB_FILE) / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return 0.0


def page_settings() -> None:
    """Settings (section 4.5): at most seven controls."""
    st.title(C.PAGE_SETTINGS)
    st.caption(f"Your data folder: {paths.PROJECT_ROOT}")

    from quant.config import RISK_PROFILE_DESCRIPTIONS, RISK_PROFILES
    from quant.portfolio.account import load_account, write_account_fields

    account = load_account()

    def _save(fields: dict) -> None:
        write_account_fields(fields)

    st.radio("Risk profile", list(RISK_PROFILES.keys()),
             index=list(RISK_PROFILES.keys()).index(account.risk_profile),
             captions=[RISK_PROFILE_DESCRIPTIONS[p] for p in RISK_PROFILES],
             format_func=lambda p: p.capitalize(), key="s_profile",
             on_change=lambda: _save({"risk_profile": st.session_state["s_profile"]}))
    st.number_input("Monthly amount to invest (EUR)", min_value=0.0, value=0.0,
                    step=10.0, key="s_monthly",
                    help="0 or empty means no monthly plan.")
    st.number_input("My savings plans run on day", min_value=1, max_value=28,
                    value=int(account.savings_plan_day or 1), step=1, key="s_day",
                    on_change=lambda: _save(
                        {"savings_plan_day": int(st.session_state["s_day"])}))

    from quant.engine import notify as _notify

    st.write(f"Telegram: {_notify.status_line()}")
    with st.expander("Set up Telegram"):
        token = st.text_input("Bot token", type="password", key="s_tg_token")
        chat = st.text_input("Chat id", key="s_tg_chat")
        if st.button("Send test"):
            ok = _notify.send_test({"channel": "telegram",
                                    "telegram_bot_token": token,
                                    "telegram_chat_id": chat})
            st.success("Test sent." if ok else "Test failed.")

    st.write(f"Prices: last date {C.fmt_date(R.latest_bar_date())}.")
    if st.button("Refresh prices"):
        st.caption("Run `quant refresh` to fetch prices.")

    from quant import __version__

    st.write(f"Software: version {__version__}.")
    st.caption("Update with: quant upgrade")

    from quant.engine import backup as _backup

    st.write(f"Backups: {_backup.backup_dir()}")
    if st.button("Back up now"):
        res = _backup.create_backup()
        st.success("Backup written." if res.get("ok") else "Backup failed.")
