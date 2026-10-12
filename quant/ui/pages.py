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
from quant.engine.decisions import format_item_line, open_since_line, timed_items
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


def _totals() -> tuple[float | None, float]:
    """The Total value and cash from the valuation store (T3).

    Returns ``(total_or_None, cash)``; total includes cash. None when no store.
    """
    from quant.engine import mtm

    lt = mtm.latest_total()
    if not lt:
        return None, 0.0
    cash = float(lt.get("cash_eur") or 0)
    return float(lt.get("value_eur") or 0) + cash, cash


def _since_first_pct(series) -> float | None:
    if series is None or len(series) < 2:
        return None
    first = float(series.iloc[0]["value_eur"])
    last = float(series.iloc[-1]["value_eur"])
    if not first:
        return None
    return (last / first - 1.0) * 100.0


def _render_tasks() -> None:
    """Stateful task items in the action area (T4), not a floating banner."""
    from quant.data.database import write_connection
    from quant.engine import reminders

    try:
        with write_connection() as conn:
            rem = reminders.savings_plan_reminder(conn)
    except Exception:  # noqa: BLE001
        rem = None
    if not rem:
        return
    st.markdown(f"**{C.SEC_TASKS}**")
    st.write(rem["message"])
    if st.button(C.BTN_REMINDER_DONE, key="sav_done"):
        try:
            with write_connection() as conn:
                reminders.dismiss(conn)
            st.rerun()
        except Exception:  # noqa: BLE001
            st.error("Could not save. Nothing was written.")


def _invest_session_items() -> list[dict]:
    """The items pushed by 'Add to today's list' (T8), This-month horizon."""
    return list(st.session_state.get("_invest_items", []))


def _render_what_to_do(holdings: list[dict]) -> None:
    """The ONE list: Today / This month, Buy and Sell, stateful and paired."""
    st.subheader(C.SEC_WHAT_TO_DO)
    from quant.data.database import write_connection
    from quant.engine import advice_state
    from quant.engine.advice import build_advice

    regime = R.read_regime()
    advice, _rejected = build_advice(holdings=holdings, regime=regime)
    items = timed_items(holdings, advice) + _invest_session_items()
    # T5: persist state (open_since / status) so a repeated item shows its age
    # and a finished one stops repeating.
    try:
        with write_connection() as conn:
            items = advice_state.sync(conn, items)
    except Exception:  # noqa: BLE001
        pass
    if not items:
        st.write(C.WHAT_TO_DO_EMPTY)
        return
    for horizon in (C.HORIZON_TODAY, C.HORIZON_MONTH):
        group = [it for it in items if it["horizon"] == horizon]
        if not group:
            continue
        st.markdown(f"**{horizon}**")
        for it in group:
            line = format_item_line(it)
            if str(it.get("verb")) == "sell_part" or it.get("direction") == C.DIRECTION_SELL:
                from quant.engine.pairing import paired_line

                line = paired_line({**it, "label": it.get("label") or it.get("symbol")},
                                   holdings=holdings)
            st.write(line)
            cls = C.CLASS_LABELS.get(it["class"], it["class"])
            caption = f"{cls} — {it['reason']}"
            since = open_since_line(it)
            if since:
                caption += f" ({since})"
            st.caption(caption)
            if it.get("id"):
                if st.button("Mark done", key=f"adv_done_{it['id']}"):
                    try:
                        with write_connection() as conn:
                            advice_state.mark_done(conn, it["id"])
                        st.rerun()
                    except Exception:  # noqa: BLE001
                        st.error("Could not save. Nothing was written.")


def page_home() -> None:
    """Portfolio home: value header, chart, tasks, what to do, holdings, invest."""
    st.title("Portfolio")
    holdings = _holdings()
    portfolio = R.load_portfolio()
    total, cash = _totals()
    if total is None:
        total = _total_value(holdings, portfolio)
    day_eur, day_pct, _day = _day_delta()
    series = R.read_value_series()
    first_pct = _since_first_pct(series)

    st.metric("Total value", C.fmt_eur_whole(total),
              delta=f"vs previous close: {day_eur:+,.0f} EUR ({day_pct:+.1f}%)")
    when = R.latest_bar_date()
    st.caption(f"Prices {C.AS_OF_DATE.format(date=C.fmt_date(when))}.")
    if first_pct is not None:
        st.caption(f"since first purchase: {first_pct:+.1f}%")

    R._render_value_chart(R.read_history(), holdings)

    _render_tasks()
    _render_what_to_do(holdings)

    st.subheader("Holdings")
    rows = []
    for h in holdings:
        rows.append({
            "Holding": R.label_for(h.get("name") or h["symbol"], h["symbol"]),
            "Class": C.CLASS_LABELS.get(str(h.get("tier", "")).upper(), h.get("tier", "")),
            "Value (EUR)": C.fmt_eur(h.get("value_eur") or 0),
            "Now / target": f"{float(h.get('current_weight') or 0) * 100:.0f}% / "
                            f"{float(h.get('target_weight') or 0) * 100:.0f}%",
        })
    # T3: cash is a first-class row (class Cash, target from strategy).
    cash_target = _cash_target()
    rows.append({
        "Holding": C.CLASS_CASH,
        "Class": C.CLASS_CASH,
        "Value (EUR)": C.fmt_eur(cash),
        "Now / target": C.CASH_TARGET_HEADER.format(pct=f"{cash_target * 100:.0f}"),
    })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if not holdings:
        st.info("Add what you own in Update holdings.")

    _render_invest_block(holdings)


def _day_delta() -> tuple[float, float, object]:
    from quant.engine import mtm

    return mtm.day_delta()


def _cash_target() -> float:
    """The strategy's cash target (T3); a sensible default when unset."""
    try:
        from quant.config import RISK_PROFILES
        from quant.portfolio.account import load_account

        profile = load_account().risk_profile
        # Risk profiles are invested-only 3-tuples; cash target is a calm default.
        _ = RISK_PROFILES.get(profile)
    except Exception:  # noqa: BLE001
        pass
    return 0.10


def _render_invest_block(holdings: list[dict]) -> None:
    """The class-based invest block (T8): amount + presets, a class plan, add."""
    st.subheader(C.SEC_INVEST)
    from quant.config import RISK_PROFILE_DESCRIPTIONS, RISK_PROFILES
    from quant.portfolio.account import load_account, write_account_fields

    account = load_account()
    profiles = list(RISK_PROFILES.keys())
    current = account.risk_profile if account.risk_profile in profiles else profiles[0]
    strategy = st.radio(
        C.INVEST_STRATEGY_LABEL, profiles, index=profiles.index(current),
        captions=[RISK_PROFILE_DESCRIPTIONS[p] for p in profiles],
        format_func=lambda p: p.capitalize(), horizontal=True, key="invest_strategy")
    if strategy != current:
        # T8: a strategy change requires an inline confirm before applying.
        st.caption(C.STRATEGY_CONFIRM_LABEL)
        if st.button(C.BTN_STRATEGY_APPLY, key="invest_apply_strategy"):
            write_account_fields({"risk_profile": strategy})
            st.rerun()
        return

    st.caption(C.INVEST_PRESET_HINT)
    p1, p2, p3 = st.columns(3)
    for col, val in ((p1, 50.0), (p2, 100.0), (p3, 500.0)):
        if col.button(C.fmt_eur_whole(val), key=f"inv_preset_{int(val)}",
                      width="stretch"):
            st.session_state["invest_amount"] = val
            st.rerun()
    amount = st.number_input(C.INVEST_AMOUNT_LABEL, min_value=0.0, value=0.0,
                             step=10.0, key="invest_amount")
    if amount <= 0:
        return
    from quant.engine.invest_plan import propose_by_class

    plan = propose_by_class(amount, holdings, strategy)
    class_rows = [{
        C.INVEST_CLASS_HEADER: C.CLASS_LABELS.get(r["tier"], r["tier"]),
        C.INVEST_NOW_HEADER: f"{r['now_pct']:.0f}%",
        C.INVEST_TARGET_HEADER: f"{r['target_pct']:.0f}%",
        C.INVEST_SUGGESTED_HEADER: C.fmt_eur(r["suggested_eur"]),
    } for r in plan["classes"]]
    st.dataframe(pd.DataFrame(class_rows), width="stretch", hide_index=True)
    st.caption(f"Suggested sums to {C.fmt_eur(plan['total'])} of {C.fmt_eur(amount)}.")
    if not plan["orders"]:
        st.write(C.INVEST_NOTHING)
        return
    order_rows = [{
        "Holding": R.label_for(o["name"], o["symbol"]),
        "Amount": C.fmt_eur(o["amount_eur"]),
        C.INVEST_HOW_HEADER: o["how"],
    } for o in plan["orders"]]
    st.dataframe(pd.DataFrame(order_rows), width="stretch", hide_index=True)
    if plan["not_allocated"] > 0:
        st.caption(C.INVEST_NOT_ALLOCATED.format(amount=C.fmt_eur(plan["not_allocated"])))
    if st.button(C.BTN_ADD_TO_LIST, key="invest_add_list", width="stretch"):
        st.session_state["_invest_items"] = [{
            "verb": "buy", "symbol": o["symbol"],
            "label": R.label_for(o["name"], o["symbol"]),
            "amount_eur": o["amount_eur"], "reason": o["how"],
            "horizon": C.HORIZON_MONTH, "class": o["tier"]} for o in plan["orders"]]
        st.success("Added to Today's list under This month.")
        st.rerun()


def page_update() -> None:
    """Update holdings: broker table + live read-only columns (T2)."""
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

    live = {h["symbol"]: h for h in _holdings()}
    when = R.latest_bar_date()
    edited = R._render_broker_editor(portfolio, live=live)
    if live:
        st.caption(f"Live values {C.AS_OF_DATE.format(date=C.fmt_date(when))}. "
                   "Broker cells stay editable until you confirm.")

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
                    # T2: invalidate the editor widget so it re-syncs from the DB.
                    st.session_state.pop("holdings", None)
                    st.success("Saved.")
                    st.switch_page("pages/portfolio.py")
                else:
                    st.error("Could not save. Nothing was written.")
        if c2.button("Cancel", width="stretch"):
            st.session_state.pop("_diff", None)
            st.rerun()

    _render_change_type()


def _render_change_type() -> None:
    """Override the instrument type for one holding (T2: renamed expander)."""
    with st.expander("Instrument type (Stock / Fund / Commodity)"):
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
        # v10.8.4 (R2): a blank realized figure is a blank cell, never a NaN.
        df["realized_pnl_eur"] = df["realized_pnl_eur"].apply(
            lambda v: "" if v is None or v != v else v)
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

    # T5: the advice log (the stateful record of what was recommended).
    st.subheader(C.SEC_ADVICE_LOG)
    try:
        from quant.engine import advice_state

        log = advice_state.history()
    except Exception:  # noqa: BLE001
        log = []
    if not log:
        st.caption(C.ADVICE_LOG_EMPTY)
    else:
        rows = [{
            "Date": C.fmt_date(r.get("open_since") or r.get("created_at")),
            "Action": str(r.get("verb") or ""),
            "Instrument": R.label_for(_display(r.get("symbol") or ""),
                                      r.get("symbol") or ""),
            "Amount (EUR)": C.fmt_eur(r.get("amount_eur") or 0),
            "Status": str(r.get("status") or ""),
        } for r in log]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _safe_score(value) -> str:
    """A score string or an explicit em dash (never a literal None, R2)."""
    try:
        if value is None:
            return "-"
        return f"{float(value):.0f} / 100"
    except (TypeError, ValueError):
        return "-"


def _research_rows(holdings: list[dict]) -> list[dict]:
    """The merged Research funnel rows (T6): held + watchlist + idea + rejected."""
    from quant.engine import manifest as M
    from quant.reporting.artifacts import read_scores

    held = {str(h.get("symbol")): h for h in holdings or []}
    reg = R.q("SELECT symbol, instrument_class, universe_status FROM asset_registry")
    ideas = R.q("SELECT symbol, score FROM funnel_survivors")
    idea_score = ({str(r["symbol"]): float(r["score"] or 0)
                   for _, r in ideas.iterrows()} if not ideas.empty else {})
    nlp_ok = M.stage_status("nlp") == "ok"

    def _row(sym, instrument_class, universe_status, tier_word=""):
        s = read_scores(sym) or {}
        comp = s.get("active_score")
        if comp is None:
            comp = s.get("structural_grade")
        conf = "high" if (nlp_ok and comp is not None) else "low"
        if sym in held:
            stage = "Held"
            h = held[sym]
            wt = (f"{float(h.get('current_weight') or 0) * 100:.0f}% / "
                  f"{float(h.get('target_weight') or 0) * 100:.0f}%")
            reason = f"held; {tier_word or 'class set'}."
        elif sym in idea_score:
            stage = "Idea"
            wt = "-"
            reason = f"funnel idea, score {idea_score[sym]:.0f}."
        elif str(universe_status).upper() == "WATCHLIST":
            stage = "Watchlist"
            wt = "-"
            reason = "on the watchlist."
        else:
            stage = "Rejected"
            wt = "-"
            reason = "in the universe; not shortlisted this run."
        news = R.q("SELECT COUNT(*) AS n FROM nlp_evidence WHERE symbol = ?", [sym])
        n = int(news["n"].iloc[0]) if not news.empty else 0
        if comp is None:
            reason = M.empty_state_reason("scoring") or "not scored yet."
        return {"Symbol": R.label_for(_display(sym), sym),
                "Class": C.class_word(instrument_class or "EQUITY"),
                "Stage": stage, "Score": _safe_score(comp), "Confidence": conf,
                "News": str(n), "Weight now / target": wt, "Reason": reason,
                "_sym": sym}

    out = []
    seen = set()
    for sym, h in held.items():
        out.append(_row(sym, "EQUITY", "ACTIVE",
                        C.CLASS_LABELS.get(str(h.get("tier", "")).upper(), "")))
        seen.add(sym)
    if not reg.empty:
        for _, r in reg.iterrows():
            sym = str(r["symbol"])
            if sym in seen:
                continue
            out.append(_row(sym, r.get("instrument_class"), r.get("universe_status")))
            seen.add(sym)
    return out


def _display(sym: str) -> str:
    try:
        from quant.data.names import display_name

        return display_name(sym)
    except Exception:  # noqa: BLE001
        return sym


def _pipeline_table() -> None:
    """The Pipeline health table (T6/T7): every stage with its status."""
    from quant.engine import manifest as M

    rows = M.read_manifest()
    if not rows:
        st.info(C.NOT_AVAILABLE.format(reason="no analysis run yet") + " "
                f"[{C.BTN_RUN_ANALYSIS}]")
        return
    table = []
    for r in rows:
        last = r.get("finished_at")
        table.append({
            "Stage": M.stage_word(r["stage"]),
            "Status": M.status_word(r["status"]),
            "Last run": C.fmt_ts(last) if last else "-",
            "Coverage": r.get("coverage") or "-",
            "Error": r.get("error") or "-",
        })
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)


def page_analysis() -> None:
    """Full analysis: the Research funnel, Market, Pipeline health (T6)."""
    st.title("Full analysis")
    tabs = st.tabs(["Holdings analysis", C.TAB_RESEARCH, C.SEC_MARKET,
                    C.SEC_PIPELINE_HEALTH])
    with tabs[0]:
        holdings = _holdings()
        if holdings:
            rows = [{
                "Holding": R.label_for(h.get("name") or h["symbol"], h["symbol"]),
                "Class": C.CLASS_LABELS.get(str(h.get("tier", "")).upper(), h.get("tier", "")),
                "Value (EUR)": C.fmt_eur(h.get("value_eur") or 0),
                "Profit (EUR)": C.fmt_eur(h.get("profit_eur") or 0),
                "Now / target": f"{float(h.get('current_weight') or 0) * 100:.0f}% / "
                                f"{float(h.get('target_weight') or 0) * 100:.0f}%",
            } for h in holdings]
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.info("No holdings yet.")
    with tabs[1]:
        _render_research()
    with tabs[2]:
        _render_market()
    with tabs[3]:
        _pipeline_table()
        if st.button(C.BTN_RUN_ANALYSIS, key="run_analysis_health",
                     width="stretch"):
            _run_cli("run")


def _render_research() -> None:
    """The merged Research funnel (T6): sortable table + row detail."""
    from quant.engine import manifest as M

    rows = _research_rows(_holdings())
    if not rows:
        reason = M.empty_state_reason("scoring") or "no instruments yet"
        st.info(C.NOT_AVAILABLE.format(reason=reason))
        if st.button(C.BTN_RUN_ANALYSIS, key="run_analysis_research",
                     width="stretch"):
            _run_cli("run")
        return
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in rows])
    if len(df) > 10:
        query = st.text_input(C.RESEARCH_SEARCH_LABEL, key="research_search")
        stages = st.multiselect(C.RESEARCH_STAGE_FILTER,
                                sorted(df["Stage"].unique()), key="research_stage")
        if query:
            df = df[df["Symbol"].str.contains(query, case=False, na=False)]
        if stages:
            df = df[df["Stage"].isin(stages)]
    st.dataframe(df, width="stretch", hide_index=True)
    st.caption(f"coverage {_manifest_coverage()}")
    for r in rows:
        sym = r["_sym"]
        with st.expander(f"{r['Symbol']} - {r['Stage']}"):
            _render_instrument_detail(sym, r)


def _manifest_coverage() -> str:
    from quant.engine import manifest as M

    rows = {x["stage"]: x for x in M.read_manifest()}
    scoring = rows.get("scoring", {})
    return scoring.get("coverage") or "-"


def _render_instrument_detail(sym: str, row: dict) -> None:
    """Row drawer: price chart, component bars, news, SEC mentions (T6)."""
    hist = R.q("SELECT Date, Close FROM market_history WHERE Symbol = ? "
               "ORDER BY Date", [sym])
    if not hist.empty:
        import plotly.graph_objects as go

        hist["d"] = pd.to_datetime(hist["Date"], errors="coerce")
        hist = hist.dropna(subset=["d"])
        if len(hist) >= 2:
            fig = go.Figure(go.Scatter(x=hist["d"], y=hist["Close"], mode="lines"))
            fig.update_layout(height=180, margin=dict(l=8, r=8, t=8, b=8),
                              showlegend=False)
            st.plotly_chart(fig, width="stretch")
    st.write(f"Score: {row['Score']} ({row['Confidence']} confidence)")
    comps = R.q("SELECT structural_grade, tactical_grade, active_score "
                "FROM asset_registry WHERE symbol = ?", [sym])
    if not comps.empty:
        c = comps.iloc[0]
        st.write(f"Fundamental {_safe_score(c.get('structural_grade'))} | "
                 f"Technical {_safe_score(c.get('tactical_grade'))}")
    news = R.q("SELECT title, score FROM nlp_evidence WHERE symbol = ? "
               "ORDER BY published_at DESC LIMIT 5", [sym])
    if not news.empty:
        for _, n in news.iterrows():
            st.caption(f"{n.get('title') or '-'}")
    sec = R.q("SELECT COUNT(*) AS n FROM nlp_evidence WHERE symbol = ? "
              "AND lower(coalesce(source,'')) = 'sec'", [sym])
    n = int(sec["n"].iloc[0]) if not sec.empty else 0
    st.caption(f"SEC filing mentions: {n}.")


def _render_market() -> None:
    """Market tab (T6): regime + confidence + what it changes, or reason + run."""
    from quant.engine import manifest as M

    regime = R.read_regime()
    label = regime.get("label")
    conf = regime.get("confidence")
    if label and conf:
        st.write(f"Market trend: {label} ({conf} confidence).")
        st.caption("This changes only the Active part today; the Long-term part is "
                   "untouched.")
    else:
        reason = M.empty_state_reason("regime") or "not enough history yet"
        st.info(C.NOT_AVAILABLE.format(reason=reason) + f" [{C.BTN_RUN_ANALYSIS}]")
        if st.button(C.BTN_RUN_ANALYSIS, key="run_analysis_market",
                     width="stretch"):
            _run_cli("run")
    try:
        from quant.engine.news_pillar import model_available

        if not model_available():
            st.caption(C.MARKET_NEWS_OFF)
    except Exception:  # noqa: BLE001
        st.caption(C.MARKET_NEWS_OFF)


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
