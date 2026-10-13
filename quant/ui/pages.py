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
# v10.8.5 (T4): derived, read-only editor columns never enter the diff/validation.
_DERIVED_COLS = ["Name", "Shares", "Check"]


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


def _total_value_store() -> float | None:
    """The Total value (invested only) from the valuation store (T3; v10.8.7 R-C1).

    None when no store row exists. Cash is not an entity, so there is no cash term.
    """
    from quant.engine import mtm

    lt = mtm.latest_total()
    if not lt:
        return None
    return float(lt.get("value_eur") or 0)


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


def _idea_rows() -> list[dict]:
    """Idea candidates (T2): watched non-held symbols with composite >= IDEA_FLOOR.

    v10.8.7 (T2): ideas come from the ``scores`` store, not the funnel cache, so a
    watched symbol with a computed composite at or above the floor becomes an Idea
    and flows into Research, swap candidates, the invest plan and This month.
    """
    from quant.config import IDEA_FLOOR
    from quant.reporting.artifacts import read_score_row

    held = {str(h.get("symbol")) for h in _holdings()}
    df = R.q("SELECT symbol FROM asset_registry")
    if df.empty:
        return []
    out: list[dict] = []
    for _, r in df.iterrows():
        sym = str(r["symbol"])
        if sym in held:
            continue
        row = read_score_row(sym) or {}
        comp = row.get("composite")
        if comp is None or float(comp) < IDEA_FLOOR:
            continue
        out.append({"symbol": sym, "name": _display(sym), "score": float(comp)})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out[:50]


def _scores_map() -> dict:
    """The scores store as ``{symbol: {active_score, structural_grade, composite}}``.

    v10.8.7 (T4): the decision taxonomy is fed the real scores, so a Threat or a
    Swap can actually fire.
    """
    from quant.reporting.artifacts import read_score_row

    df = R.q("SELECT symbol FROM asset_registry")
    if df.empty:
        return {}
    out: dict = {}
    for _, r in df.iterrows():
        sym = str(r["symbol"])
        row = read_score_row(sym) or {}
        if row:
            out[sym] = {"active_score": row.get("composite"),
                        "structural_grade": row.get("structural_grade"),
                        "composite": row.get("composite")}
    return out


def _taxonomy_ui_items(holdings: list[dict], regime) -> list[dict]:
    """The widened verb classes as ONE-list items (T3/T4).

    Rebalance stays with build_advice (which already yields the drift buy/sell);
    the new classes are appended here, fed the real scores and ideas.
    """
    from quant.engine import action_taxonomy as T

    taxo = T.build(holdings, scores=_scores_map(), ideas=_idea_rows(),
                   regime=regime)
    directions = {T.VERB_SWAP: C.DIRECTION_SELL, T.VERB_THREAT: "Review",
                  T.VERB_PLAN: C.DIRECTION_BUY}
    out: list[dict] = []
    for t in taxo:
        if t["verb"] == T.VERB_REBALANCE:
            continue
        out.append({
            "horizon": t["horizon"], "class": t["class"],
            "direction": directions.get(t["verb"], C.DIRECTION_BUY),
            "verb": t["verb"], "label": t["label"], "amount_eur": t["amount_eur"],
            "reason": t["reason"], "symbol": t["symbol"],
            "evidence": t.get("evidence"),
            "target_symbol": t.get("target_symbol"),
            "target_name": t.get("target_name"),
        })
    return out


def _render_what_to_do(holdings: list[dict]) -> None:
    """The ONE list: Today / This month, five verb classes, stateful and paired."""
    st.subheader(C.SEC_WHAT_TO_DO)
    from quant.data.database import write_connection
    from quant.engine import advice_state
    from quant.engine.advice import build_advice

    regime = R.read_regime()
    advice, _rejected = build_advice(holdings=holdings, regime=regime)
    items = (timed_items(holdings, advice) + _invest_session_items()
             + _taxonomy_ui_items(holdings, regime))
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
            line = _item_line(it, holdings)
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


def _item_line(it: dict, holdings: list[dict]) -> str:
    """The single line for one item (T3/T5/D8): swap pairs, sells pair, else verb."""
    verb = str(it.get("verb") or "")
    if verb == "swap":
        tgt = it.get("target_name") or it.get("target_symbol") or "an idea"
        amount = float(it.get("amount_eur") or 0)
        return f"Sell ~{amount:.0f} EUR of {it.get('label')} -> buy {tgt}"
    if verb == "sell_part" or it.get("direction") == C.DIRECTION_SELL:
        from quant.engine.pairing import paired_line

        return paired_line({**it, "label": it.get("label") or it.get("symbol")},
                           holdings=holdings)
    return format_item_line(it)


def page_home() -> None:
    """Portfolio home: value header, chart, tasks, what to do, holdings, invest."""
    st.title("Portfolio")
    holdings = _holdings()
    portfolio = R.load_portfolio()
    total = _total_value_store()
    if total is None:
        total = _total_value(holdings, portfolio)
    day_eur, day_pct, _day = _day_delta()
    series = R.read_value_series()
    first_pct = _since_first_pct(series)

    st.metric("Total value", C.fmt_eur_whole(total),
              delta=f"vs previous close: {day_eur:+,.0f} EUR ({day_pct:+.1f}%)")
    # v10.8.5 (D7): the "as of" date is the valuation store's latest date, so the
    # caption and the chart tail can never disagree.
    when = _as_of_date()
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
    # v10.8.7 (R-C1): no cash row. Uninvested cash is not an entity.
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if not holdings:
        st.info("Add what you own in Update holdings.")

    _render_invest_block(holdings)


def _day_delta() -> tuple[float, float, object]:
    from quant.engine import mtm

    return mtm.day_delta()


def _as_of_date():
    """The valuation store's latest date, else the latest market bar (D7)."""
    from quant.engine import mtm

    return mtm.latest_price_date() or R.latest_bar_date()


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

    plan = propose_by_class(amount, holdings, strategy,
                            candidates=_idea_rows(),
                            open_advice=_open_advice_items(holdings))
    class_rows = [{
        C.INVEST_CLASS_HEADER: C.CLASS_LABELS.get(r["tier"], r["tier"]),
        C.INVEST_NOW_HEADER: f"{r['now_pct']:.0f}%",
        C.INVEST_TARGET_HEADER: f"{r['target_pct']:.0f}%",
        C.INVEST_SUGGESTED_HEADER: C.fmt_eur(r["suggested_eur"]),
    } for r in plan["classes"]]
    st.dataframe(pd.DataFrame(class_rows), width="stretch", hide_index=True)
    # T3: the one-off sum is separate from the savings-plan EUR/m increase.
    st.caption(f"Suggested sums to {C.fmt_eur(plan['total'])} of {C.fmt_eur(amount)}.")
    if plan["savings_plan_eur_month"] > 0:
        st.caption(f"+{C.fmt_eur(plan['savings_plan_eur_month'])} EUR/m savings plan.")
    for ex in plan["excluded"]:
        st.caption(f"{ex['name']}: {ex['reason']}")
    if not plan["orders"] and not plan["savings_plan"]:
        st.write(C.INVEST_NOTHING)
        return
    order_rows = [{
        "Holding": R.label_for(o["name"], o["symbol"]),
        "Amount": C.fmt_eur(o["amount_eur"]),
        C.INVEST_HOW_HEADER: o["how"],
    } for o in (plan["orders"] + plan["savings_plan"])]
    st.dataframe(pd.DataFrame(order_rows), width="stretch", hide_index=True)
    if plan["not_allocated"] > 0:
        st.caption(C.invest_remainder_line(plan["not_allocated"]))
    if st.button(C.BTN_ADD_TO_LIST, key="invest_add_list", width="stretch"):
        st.session_state["_invest_items"] = [{
            "verb": "buy", "symbol": o["symbol"],
            "label": R.label_for(o["name"], o["symbol"]),
            "amount_eur": o["amount_eur"], "reason": o["how"],
            "horizon": C.HORIZON_MONTH, "class": o["tier"]} for o in plan["orders"]]
        st.success("Added to Today's list under This month.")
        st.rerun()


def _open_advice_items(holdings: list[dict]) -> list[dict]:
    """The open advice set (verb + symbol) for the invest plan (T3).

    An instrument with an open trim/sell gets no buy suggestion until it clears.
    """
    from quant.engine import action_taxonomy as T
    from quant.engine.advice import build_advice

    regime = R.read_regime()
    advice, _rejected = build_advice(holdings=holdings, regime=regime)
    items = [{"verb": a.get("kind"), "symbol": a.get("symbol")} for a in advice]
    items += [{"verb": t["verb"], "symbol": t["symbol"]}
              for t in T.build(holdings, scores={}, ideas=_idea_rows(),
                               regime=regime)]
    return items


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
        new = edited.drop(columns=_DERIVED_COLS, errors="ignore").to_dict("records")
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
                edited.drop(columns=_DERIVED_COLS, errors="ignore"))
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


def _short_score(value) -> str:
    """A one-token score for the Components cell ('72' or '-')."""
    return _safe_score(value).split(" ")[0]


def _news_tone(score) -> str:
    """The plain tone word for a sentiment score (neutral on a missing value)."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "neutral"
    if s > 0.15:
        return "positive"
    if s < -0.15:
        return "negative"
    return "neutral"


def _research_rows(holdings: list[dict]) -> list[dict]:
    """The merged Research funnel rows (T6/T1 v10.8.5; v10.8.7 T2).

    A computed score renders as a NUMBER plus a confidence chip, never a dash
    (R-C2). A watched non-held symbol with composite >= IDEA_FLOOR is an Idea.
    """
    from quant.config import IDEA_FLOOR
    from quant.engine import manifest as M
    from quant.reporting.artifacts import read_score_row

    held = {str(h.get("symbol")): h for h in holdings or []}
    reg = R.q("SELECT symbol, instrument_class, universe_status FROM asset_registry")

    def _row(sym, instrument_class, universe_status, tier_word=""):
        row = read_score_row(sym) or {}
        comp = row.get("composite")
        conf = str(row.get("confidence") or "low")
        fund = row.get("structural_grade")
        tech = row.get("tactical_grade")
        components = (f"F {_short_score(fund)} / T {_short_score(tech)} / "
                      f"conf {conf}")
        is_idea = (sym not in held and comp is not None
                   and float(comp) >= IDEA_FLOOR)
        if sym in held:
            stage = "Held"
            h = held[sym]
            wt = (f"{float(h.get('current_weight') or 0) * 100:.0f}% / "
                  f"{float(h.get('target_weight') or 0) * 100:.0f}%")
            reason = f"held; {tier_word or 'class set'}."
        elif is_idea:
            stage = "Idea"
            wt = "-"
            reason = (f"composite {float(comp):.0f} at or above the "
                      f"{IDEA_FLOOR:.0f} floor.")
        elif str(universe_status).upper() == "WATCHLIST":
            stage = "Watchlist"
            wt = "-"
            reason = "on the watchlist."
        else:
            stage = "Rejected"
            wt = "-"
            reason = "in the universe; not shortlisted this run."
        news_rows = R.q("SELECT score FROM nlp_evidence WHERE symbol = ? "
                        "ORDER BY published_at DESC LIMIT 5", [sym])
        n = 0 if news_rows.empty else len(news_rows)
        tone = "no news" if n == 0 else _news_tone(news_rows["score"].iloc[0])
        news = f"{n} ({tone})"
        if comp is None:
            reason = M.empty_state_reason("scoring") or "not scored yet."
        return {"Symbol": R.label_for(_display(sym), sym),
                "Class": C.class_word(instrument_class or "EQUITY"),
                "Stage": stage, "Score": _safe_score(comp),
                "Components": components, "News": news,
                "Weight now / target": wt, "Reason": reason, "_sym": sym}

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


def _open_alert_count(sym: str) -> int:
    """Open alerts for a symbol (T5). Never raises."""
    try:
        from quant.data.database import read_only_connection
        from quant.engine import alerts as alerts_mod

        with read_only_connection() as conn:
            return sum(1 for a in alerts_mod.open_alerts(conn)
                       if str(a.get("symbol")) == sym)
    except Exception:  # noqa: BLE001
        return 0


def _render_holdings_analysis() -> None:
    """Per-holding drill-down (v10.8.5, T5): table + row drawer."""
    from quant.reporting.artifacts import read_scores

    holdings = _holdings()
    if not holdings:
        st.info("No holdings yet.")
        return
    rows = []
    for h in holdings:
        s = read_scores(h["symbol"]) or {}
        score = s.get("active_score")
        if score is None:
            score = s.get("structural_grade")
        cur = float(h.get("current_weight") or 0)
        tgt = float(h.get("target_weight") or 0)
        rows.append({
            "Symbol": R.label_for(h.get("name") or h["symbol"], h["symbol"]),
            "Class": C.CLASS_LABELS.get(str(h.get("tier", "")).upper(), h.get("tier", "")),
            "Now / Target": f"{cur * 100:.0f}% / {tgt * 100:.0f}%",
            "Drift (pp)": f"{(cur - tgt) * 100:+.1f}",
            "Score": _safe_score(score),
            "Alerts": str(_open_alert_count(h["symbol"])),
            "Reason": "held",
            "_sym": h["symbol"],
        })
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in rows])
    st.dataframe(df, width="stretch", hide_index=True)
    st.caption(f"coverage {_manifest_coverage()}")
    by_sym = {r["_sym"]: {"Score": r["Score"], "Components": ""} for r in rows}
    _inspect_drawer([r["_sym"] for r in rows], by_sym, prefix="ha",
                    key="ha_inspect", label=C.INSPECT_HOLDING)


def page_analysis() -> None:
    """Full analysis: the Research funnel, Market, Pipeline health (T6)."""
    st.title("Full analysis")
    tabs = st.tabs(["Holdings analysis", C.TAB_RESEARCH, C.SEC_MARKET,
                    C.SEC_PIPELINE_HEALTH])
    # v10.8.6 (A2): an error boundary per tab. A failing tab renders its own
    # error card and the other three still render; one bug never kills the page.
    with tabs[0]:
        _guarded("Holdings analysis", _render_holdings_analysis)
    with tabs[1]:
        _guarded(C.TAB_RESEARCH, _render_research)
    with tabs[2]:
        _guarded(C.SEC_MARKET, _render_market)
    with tabs[3]:
        _guarded(C.SEC_PIPELINE_HEALTH, _render_pipeline)


def _guarded(label: str, render: object, *args: object) -> None:
    """Render a tab body, catching any failure into an in-tab error card (A2)."""
    try:
        render(*args)  # type: ignore[operator]
    except Exception as e:  # noqa: BLE001
        st.error(f"{label} tab failed: {e}")
        st.caption("See Pipeline health for details.")


def _render_pipeline() -> None:
    """Pipeline health tab body: the stage table + a Run analysis button."""
    _pipeline_table()
    st.caption(C.RUN_ANALYSIS_HINT)
    if st.button(C.BTN_RUN_ANALYSIS, key="run_analysis_health", width="stretch"):
        _run_cli("run")


def _render_research() -> None:
    """The merged Research funnel (T6; v10.8.5, D1/D2): table + lazy drawers."""
    from quant.engine import manifest as M

    rows = _research_rows(_holdings())
    if not rows:
        reason = M.empty_state_reason("scoring") or "no instruments yet"
        st.info(C.NOT_AVAILABLE.format(reason=reason) + f" [{C.BTN_RUN_ANALYSIS}]")
        if st.button(C.BTN_RUN_ANALYSIS, key="run_analysis_research",
                     width="stretch"):
            _run_cli("run")
        return
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in rows])
    if len(df) > 10:
        c1, c2 = st.columns(2)
        query = c1.text_input(C.RESEARCH_SEARCH_LABEL, key="research_search")
        stages = c2.multiselect(C.RESEARCH_STAGE_FILTER,
                                sorted(df["Stage"].unique()), key="research_stage")
        if query:
            df = df[df["Symbol"].str.contains(query, case=False, na=False)]
        if stages:
            df = df[df["Stage"].isin(stages)]
    st.dataframe(df, width="stretch", hide_index=True)
    st.caption(f"coverage {_manifest_coverage()}")
    # v10.8.5 (D1): the funnel table above is the content; the per-row drawer
    # below is optional. Its chart carries a unique key so identical figures can
    # never collide on Streamlit's auto-generated element id.
    by_sym = {r["_sym"]: r for r in rows}
    _inspect_drawer([r["_sym"] for r in rows], by_sym, prefix="rs",
                    key="rs_inspect", label=C.INSPECT_SYMBOL)


def _manifest_coverage() -> str:
    from quant.engine import manifest as M

    rows = {x["stage"]: x for x in M.read_manifest()}
    scoring = rows.get("scoring", {})
    return scoring.get("coverage") or "-"


def _inspect_drawer(symbols: list[str], rows_by_sym: dict, prefix: str,
                    key: str, label: str) -> None:
    """One lazy per-tab symbol drawer (v10.8.6, A1).

    Renders the detail for the selected symbol only; the select's keys and the
    drawer's widget keys are scoped by ``prefix`` ("ha"/"rs"), so the same
    symbol inspected in two tabs can never collide on a Streamlit key.
    """
    if not symbols:
        return
    choice = st.selectbox(label, symbols, index=None,
                          placeholder=C.INSPECT_PLACEHOLDER, key=key,
                          format_func=_display)
    if choice is None:
        return
    _render_instrument_detail(choice, rows_by_sym[choice], prefix=prefix)


def _render_instrument_detail(sym: str, row: dict, prefix: str = "rs") -> None:
    """Row drawer: price chart, score components, headlines, SEC (T6/T5).

    ``prefix`` scopes every widget key to the tab context ("ha"/"rs"), so the
    same symbol inspected in two tabs can never collide on a page-global key
    (v10.8.6, A1).
    """
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
            st.plotly_chart(fig, width="stretch", key=f"{prefix}_chart_{sym}")
    st.write(f"Score {row['Score']} | {row['Components']}")
    comps = R.q("SELECT structural_grade, tactical_grade, active_score "
                "FROM asset_registry WHERE symbol = ?", [sym])
    if not comps.empty:
        c = comps.iloc[0]
        m1, m2, m3 = st.columns(3)
        m1.metric("Fundamental", _short_score(c.get("structural_grade")))
        m2.metric("Technical", _short_score(c.get("tactical_grade")))
        m3.metric("Overall", _short_score(c.get("active_score")))
    news = R.q("SELECT title, score FROM nlp_evidence WHERE symbol = ? "
               "ORDER BY published_at DESC LIMIT 5", [sym])
    if news.empty:
        st.caption(C.RESEARCH_NO_NEWS)
    else:
        for _, n in news.iterrows():
            st.caption(f"{n.get('title') or '-'} ({_news_tone(n.get('score'))})")
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
    """Run a CLI command in a subprocess and report the result (v10.8.3).

    v10.8.7 (T2/A3): the run is wrapped in ``st.status`` with the first-analysis
    hint, so the user sees progress while the chain fetches and computes.
    """
    import subprocess
    import sys

    with st.status(f"Running {command}…", expanded=True) as status:
        st.write(C.RUN_ANALYSIS_HINT)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "quant.cli", command],
                capture_output=True, text=True, timeout=900)
        except Exception as e:  # noqa: BLE001
            status.update(label=f"Could not run {command}", state="error")
            st.error(f"Could not run {command}: {e}")
            return
        if proc.returncode == 0:
            status.update(label=f"{command} finished.", state="complete")
        else:
            status.update(label=f"{command} failed (exit {proc.returncode}).",
                          state="error")
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
