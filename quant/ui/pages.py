"""
pages.py — the five pages of the one workflow (v10.8.3).

Portfolio (home), Update holdings, History, Full analysis, Settings. Each page
reads the ONE live source: the holdings view (positions + tiers + targets), the
value series, the timed decision list, the ledger and the tax engine. No cash.
"""
from __future__ import annotations

from datetime import date as _date

import pandas as pd
import streamlit as st

from quant import paths
from quant.engine.decisions import format_item_line, timed_items
from quant.ui import copy as C
from quant.ui import render as R

# The instrument classes the user may pick, in plain words (no "cash").
_TYPE_OPTIONS = {"Stock": "EQUITY", "Fund (ETF)": "ETF", "Commodity": "COMMODITY"}


def _holdings() -> list[dict]:
    """The live holdings view (positions + tiers + targets). Never raises."""
    try:
        from quant.engine.holdings_view import holdings_view

        return holdings_view()
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


def _render_what_to_do(holdings: list[dict]) -> None:
    """The ONE list: Today / This month, Buy and Sell, tagged by class."""
    st.subheader(C.SEC_WHAT_TO_DO)
    from quant.engine.advice import build_advice

    regime = R.read_regime()
    advice, _rejected = build_advice(holdings=holdings, regime=regime)
    items = timed_items(holdings, advice)
    if not items:
        st.write(C.WHAT_TO_DO_EMPTY)
        return
    for horizon in (C.HORIZON_TODAY, C.HORIZON_MONTH):
        group = [it for it in items if it["horizon"] == horizon]
        if not group:
            continue
        st.markdown(f"**{horizon}**")
        for it in group:
            st.write(f"{it['direction']}: {format_item_line(it)}")
            cls = C.CLASS_LABELS.get(it["class"], it["class"])
            st.caption(f"{cls} — {it['reason']}")


def page_home() -> None:
    """Portfolio home: value header, chart, what to do, holdings, invest block."""
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
    _render_what_to_do(holdings)

    st.subheader("Holdings")
    rows = []
    for h in holdings:
        rows.append({
            "Holding": R.label_for(h.get("name") or h["symbol"], h["symbol"]),
            "Class": C.CLASS_LABELS.get(str(h.get("tier", "")).upper(), h.get("tier", "")),
            "Value": C.fmt_eur(h.get("value_eur") or 0),
            "Now / target": f"{float(h.get('current_weight') or 0) * 100:.0f}% / "
                            f"{float(h.get('target_weight') or 0) * 100:.0f}%",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("Add what you own in Update holdings.")

    _render_invest_block(holdings)

    c1, c2 = st.columns(2)
    if c1.button("Update holdings", width="stretch"):
        st.switch_page("pages/update.py")
    if c2.button("Full analysis", width="stretch"):
        st.switch_page("pages/analysis.py")


def _render_invest_block(holdings: list[dict]) -> None:
    """The class-based invest block (v10.8.3): one amount, a class-balanced plan."""
    st.subheader(C.SEC_INVEST)
    from quant.config import RISK_PROFILE_DESCRIPTIONS, RISK_PROFILES
    from quant.portfolio.account import load_account, write_account_fields

    account = load_account()
    profiles = list(RISK_PROFILES.keys())
    strategy = st.radio(
        C.INVEST_STRATEGY_LABEL, profiles,
        index=profiles.index(account.risk_profile) if account.risk_profile in profiles else 0,
        captions=[RISK_PROFILE_DESCRIPTIONS[p] for p in profiles],
        format_func=lambda p: p.capitalize(), horizontal=True, key="invest_strategy",
        on_change=lambda: write_account_fields(
            {"risk_profile": st.session_state["invest_strategy"]}))

    amount = st.number_input(C.INVEST_AMOUNT_LABEL, min_value=0.0, value=0.0,
                             step=10.0, key="invest_amount")
    if amount <= 0:
        return
    from quant.engine.invest_plan import propose_by_class

    plan = propose_by_class(amount, holdings, strategy)
    if not plan["orders"]:
        st.write(C.INVEST_NOTHING)
        return
    class_rows = [{
        C.INVEST_CLASS_HEADER: C.CLASS_LABELS.get(r["tier"], r["tier"]),
        C.INVEST_NOW_HEADER: f"{r['now_pct']:.0f}%",
        C.INVEST_TARGET_HEADER: f"{r['target_pct']:.0f}%",
        C.INVEST_SUGGESTED_HEADER: C.fmt_eur(r["suggested_eur"]),
    } for r in plan["classes"]]
    st.dataframe(pd.DataFrame(class_rows), width="stretch", hide_index=True)
    order_rows = [{
        "Holding": R.label_for(o["name"], o["symbol"]),
        "Amount": C.fmt_eur(o["amount_eur"]),
        C.INVEST_HOW_HEADER: o["how"],
    } for o in plan["orders"]]
    st.dataframe(pd.DataFrame(order_rows), width="stretch", hide_index=True)
    if plan["not_allocated"] > 0:
        st.caption(C.INVEST_NOT_ALLOCATED.format(amount=C.fmt_eur(plan["not_allocated"])))


def page_update() -> None:
    """Update holdings: the table, Review, Confirm, Cancel, Change type."""
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
        result = diff_tables(old, new)
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

    _render_change_type()


def _render_change_type() -> None:
    """Override the instrument type for one holding (v10.8.3)."""
    with st.expander("Change type"):
        st.caption("The type is assigned automatically; override it here.")
        holdings = _holdings()
        if not holdings:
            st.caption("Add a holding first.")
            return
        syms = [h["symbol"] for h in holdings]
        c1, c2 = st.columns(2)
        sym = c1.selectbox("Holding", syms, key="ct_sym")
        label = c2.selectbox("Type", list(_TYPE_OPTIONS.keys()), key="ct_cls")
        if st.button("Apply type", key="ct_apply"):
            from quant.data.registry_repair import set_instrument_class

            if set_instrument_class(sym, _TYPE_OPTIONS[label]):
                st.success(f"{sym} is now {label}.")
            else:
                st.error("Could not set the type.")


def _load_trades_rows() -> list[dict]:
    try:
        from quant.data.database import read_only_connection

        with read_only_connection() as conn:
            rows = conn.execute(
                "SELECT id, date, symbol, action, amount_eur, realized_pnl_eur "
                "FROM trades ORDER BY date DESC LIMIT 200").fetchall()
    except Exception:  # noqa: BLE001
        return []
    return [{"id": r[0], "date": r[1], "symbol": r[2], "action": r[3],
             "amount_eur": r[4], "realized_pnl_eur": r[5]} for r in rows]


def page_history() -> None:
    """History: an editable changes log + the yearly tax numbers + CSV."""
    st.title("History")
    st.caption("Dividends and interest are not tracked. Correct a row here if a "
               "derived number is off.")
    rows = _load_trades_rows()
    if rows:
        df = pd.DataFrame(rows)
        edited = st.data_editor(
            df, num_rows="dynamic", width="stretch", hide_index=True, key="hist_editor",
            column_config={
                "id": st.column_config.NumberColumn("id", disabled=True),
                "date": st.column_config.DateColumn("Date"),
                "symbol": st.column_config.TextColumn("Symbol"),
                "action": st.column_config.SelectboxColumn(
                    "Action", options=["buy", "sell", "dividend"]),
                "amount_eur": st.column_config.NumberColumn("Amount (EUR)", format="%.2f"),
                "realized_pnl_eur": st.column_config.NumberColumn(
                    "Realized (EUR)", format="%.2f"),
            })
        c1, c2 = st.columns(2)
        if c1.button("Save changes", type="primary", width="stretch"):
            from quant.engine.ledger import replace_trades

            records = []
            for r in edited.to_dict("records"):
                rid = r.get("id")
                if rid is None or (isinstance(rid, float) and rid != rid):
                    rid = None
                records.append({"id": rid, "date": r.get("date"),
                                "symbol": r.get("symbol"), "action": r.get("action"),
                                "amount_eur": r.get("amount_eur"),
                                "realized_pnl_eur": r.get("realized_pnl_eur")})
            replace_trades(records)
            st.success("Saved.")
            st.rerun()
        c2.download_button("Export (CSV)", df.to_csv(index=False),
                           file_name="history.csv", mime="text/csv")
    else:
        st.info("No changes recorded yet.")

    st.subheader("This year")
    filing = st.radio("Filing status", ["Single", "Married"], horizontal=True,
                      key="filing_status")
    try:
        from quant.portfolio.tax_accounting import calculate_yearly_tax_summary

        summary = calculate_yearly_tax_summary(
            _date.today().year,
            filing_status="married" if filing == "Married" else "single")
        st.write(f"Realized gain or loss (estimated): "
                 f"{C.fmt_eur(summary['realized_gains_eur'])}")
        st.write(f"Tax-free allowance remaining: "
                 f"{C.fmt_eur(summary['sparerpauschbetrag_remaining_eur'])}")
        st.write(f"Rough tax estimate: {C.fmt_eur(summary['estimated_tax_eur'])}")
    except Exception:  # noqa: BLE001
        st.caption("Tax figures unavailable.")


def page_analysis() -> None:
    """Full analysis: read-only tabs."""
    st.title("Full analysis")
    tabs = st.tabs(["Holdings analysis", "Universe scan", "Ideas", "Market",
                    "Run details"])
    with tabs[0]:
        holdings = _holdings()
        if holdings:
            rows = [{
                "Holding": R.label_for(h.get("name") or h["symbol"], h["symbol"]),
                "Class": C.CLASS_LABELS.get(str(h.get("tier", "")).upper(), h.get("tier", "")),
                "Value": C.fmt_eur(h.get("value_eur") or 0),
                "Profit": C.fmt_eur(h.get("profit_eur") or 0),
                "Now / target": f"{float(h.get('current_weight') or 0) * 100:.0f}% / "
                                f"{float(h.get('target_weight') or 0) * 100:.0f}%",
            } for h in holdings]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
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


def _run_cli(command: str) -> None:
    """Run a CLI command in a subprocess and report the result (v10.8.3)."""
    import subprocess
    import sys

    with st.spinner(f"Running {command}…"):
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "quant.cli", command],
                capture_output=True, text=True, timeout=900)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not run {command}: {e}")
            return
    if proc.returncode == 0:
        st.success(f"{command} finished.")
    else:
        st.error(f"{command} failed (exit {proc.returncode}).")
    tail = (proc.stdout or "")[-2000:]
    if tail:
        st.code(tail)


def page_settings() -> None:
    """Settings: the few controls that are not on the other pages."""
    st.title(C.PAGE_SETTINGS)
    st.caption(f"Your data folder: {paths.PROJECT_ROOT}")

    from quant.portfolio.account import load_account, write_account_fields

    account = load_account()
    st.number_input("My savings plans run on day", min_value=1, max_value=28,
                    value=int(account.savings_plan_day or 1), step=1, key="s_day",
                    on_change=lambda: write_account_fields(
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

    st.subheader("Actions")
    c1, c2, c3 = st.columns(3)
    if c1.button("Refresh prices", width="stretch"):
        _run_cli("refresh")
    if c2.button("Update software", width="stretch"):
        _run_cli("upgrade")
    if c3.button("Back up now", width="stretch"):
        from quant.engine import backup as _backup

        res = _backup.create_backup()
        st.success("Backup written." if res.get("ok") else "Backup failed.")

    from quant import __version__

    st.write(f"Software: version {__version__}.")
    st.write(f"Prices: last date {C.fmt_date(R.latest_bar_date())}.")
    from quant.engine import backup as _backup2

    st.caption(f"Backups: {_backup2.backup_dir()}")
