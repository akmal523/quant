"""
telegram_bot.py — update holdings and log actions from Telegram (v10.9.0).

Intent: the digest arrives on the phone, next to Trade Republic; the answer
should go back the same way. The bot reads commands from the ONE configured
chat and turns them into the actions the app already has:

  /portfolio                    the holdings, value, profit, now vs target
  /set SYM VALUE PROFIT [COST]  type the three broker numbers for one holding
  /buy SYM EUR [PRICE]          add a buy to a holding (price: latest close)
  /sell SYM EUR|all [PRICE]     take a sell off a holding
  /confirm  /cancel             save or drop the pending change
  /todo                         the ONE list, numbered
  /done N                       Mark done item N of the last /todo
  /history                      the latest trades and done actions

A table change is only a proposal until /confirm: the bot shows the same
Review diff as Update holdings, and /confirm runs Confirm and save (the table,
the snapshot, the ledger, the valuation store, the ONE list). /done runs Mark
done. No new writer exists (R-C4).

Invariants:
  - Messages from any chat other than ``telegram_chat_id`` are ignored.
  - :func:`handle` never raises; every reply is plain text.
  - The update offset is persisted, so a restart never replays a /confirm.
  - The bot token never appears in any log or reply.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from typing import Any

from quant import paths

STATE_NAME = "telegram_bot_state.json"
POLL_TIMEOUT = 30          # seconds; Telegram long polling
ERROR_BACKOFF = 15         # seconds after a failed poll
HISTORY_ROWS = 10

HELP = (
    "Quant-AI commands:\n"
    "/portfolio - holdings, value, profit, now vs target\n"
    "/set SYMBOL VALUE PROFIT [COST] - the broker numbers for one holding "
    "(COST = average cost per share; needed for a new holding)\n"
    "/buy SYMBOL EUR [PRICE] - add a buy (price per share in EUR; default the "
    "latest close)\n"
    "/sell SYMBOL EUR|all [PRICE] - record a sell\n"
    "/confirm - save the pending change\n"
    "/cancel - drop the pending change\n"
    "/todo - what to do, numbered\n"
    "/done N - mark item N of the last /todo done\n"
    "/history - latest trades and done actions\n"
    "Decimals may use a comma: 12,5"
)
NO_PENDING = "Nothing to confirm. Change a holding first with /set, /buy or /sell."
CANCELLED = "Change dropped. Nothing was written."
SAVED = "Saved. {n} change(s) recorded."
SAVE_FAILED = "Could not save. Nothing was written."
NO_CHANGES = "No changes against the saved table."
PENDING_HINT = "Send /confirm to save or /cancel to drop it."
UNKNOWN = "Unknown command. Send /help for the list."
NO_HOLDINGS = "No holdings saved yet."
NOT_HELD = "{sym} is not in your holdings."
NO_PRICE = "No price for {sym} yet. Add the price per share: {example}"
BAD_NUMBER = "Not a number: {text}"
USAGE = "Usage: {usage}"
TODO_EMPTY = "Nothing to do."
TODO_FIRST = "Send /todo first, then /done N with a number from that list."
TODO_RANGE = "No item {n} in the last list (1 to {last})."
DONE = "Marked done: {line}"
DONE_FAILED = "Could not save. Nothing was written."
HISTORY_EMPTY = "No trades or done actions yet."


# ── Telegram API ──────────────────────────────────────────────────────────────

def _api(token: str, method: str, params: dict, timeout: float) -> dict | None:
    """One Bot API call. Returns the decoded body, or None. Never raises."""
    url = f"https://api.telegram.org/bot{token}/{method}"
    data = urllib.parse.urlencode(params).encode()
    try:
        request = urllib.request.Request(url, data=data)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return body if isinstance(body, dict) else None
    except Exception:  # noqa: BLE001
        # Never include the token (it is in the URL) in a message.
        return None


def _state_path() -> str:
    return os.path.join(str(paths.OUTPUTS_DIR), STATE_NAME)


def _read_offset() -> int:
    try:
        with open(_state_path(), encoding="utf-8") as f:
            return int(json.load(f).get("offset") or 0)
    except Exception:  # noqa: BLE001
        return 0


def _write_offset(offset: int) -> None:
    try:
        os.makedirs(str(paths.OUTPUTS_DIR), exist_ok=True)
        with open(_state_path(), "w", encoding="utf-8") as f:
            json.dump({"offset": int(offset)}, f)
    except Exception:  # noqa: BLE001
        pass


# ── Parsing ───────────────────────────────────────────────────────────────────

def _num(text: str) -> float:
    """A typed number; a comma is a decimal separator (Trade Republic, DE)."""
    try:
        return float(str(text).replace("EUR", "").replace(",", ".").strip())
    except ValueError:
        raise ValueError(BAD_NUMBER.format(text=text)) from None


def _fmt(v: Any) -> str:
    return f"{float(v or 0):,.2f} EUR"


# ── The holdings table ────────────────────────────────────────────────────────

_TABLE_COLS = ["Symbol", "Avg_Entry_Price", "Current_Value_EUR", "Broker_PnL_EUR",
               "Plan_EUR_month"]


def _saved_rows() -> list[dict]:
    """The saved holdings table as broker rows (the editor's columns only)."""
    from quant.portfolio.portfolio import load_portfolio

    df = load_portfolio(paths.DATA_PORTFOLIO)
    if df is None or df.empty:
        return []
    rows = []
    for r in df.to_dict("records"):
        row = {c: r.get(c) for c in _TABLE_COLS if c in r}
        plan = row.get("Plan_EUR_month")
        if plan is None or plan != plan:   # NaN / NA: no plan
            row["Plan_EUR_month"] = None
        rows.append(row)
    return rows


def _find(rows: list[dict], symbol: str) -> dict | None:
    want = symbol.strip().upper()
    for r in rows:
        if str(r.get("Symbol") or "").strip().upper() == want:
            return r
    return None


def _live_price(symbol: str) -> float | None:
    from quant.data.currency import price_in_eur

    try:
        p = price_in_eur(symbol)
        return float(p) if p else None
    except Exception:  # noqa: BLE001
        return None


def _invested(row: dict) -> float:
    return float(row.get("Current_Value_EUR") or 0) - float(row.get("Broker_PnL_EUR") or 0)


def _apply_set(rows: list[dict], args: list[str]) -> list[dict]:
    usage = "/set SYMBOL VALUE PROFIT [COST]"
    if len(args) not in (3, 4):
        raise ValueError(USAGE.format(usage=usage))
    sym, value, profit = args[0], _num(args[1]), _num(args[2])
    rows = [dict(r) for r in rows]
    row = _find(rows, sym)
    if row is None:
        if len(args) < 4:
            raise ValueError(USAGE.format(usage="/set SYMBOL VALUE PROFIT COST "
                                                "(COST is needed for a new holding)"))
        row = {"Symbol": sym.upper(), "Plan_EUR_month": None}
        rows.append(row)
    if len(args) == 4:
        row["Avg_Entry_Price"] = _num(args[3])
    row["Current_Value_EUR"] = value
    row["Broker_PnL_EUR"] = profit
    return rows


def _apply_buy(rows: list[dict], args: list[str]) -> list[dict]:
    usage = "/buy SYMBOL EUR [PRICE]"
    if len(args) not in (2, 3):
        raise ValueError(USAGE.format(usage=usage))
    sym, amount = args[0], _num(args[1])
    if amount <= 0:
        raise ValueError(USAGE.format(usage=usage))
    rows = [dict(r) for r in rows]
    row = _find(rows, sym)
    price = _num(args[2]) if len(args) == 3 else _live_price(
        row["Symbol"] if row else sym.upper())
    if not price or price <= 0:
        raise ValueError(NO_PRICE.format(sym=sym.upper(),
                                         example=f"/buy {sym.upper()} {args[1]} 123,45"))
    if row is None:
        rows.append({"Symbol": sym.upper(), "Avg_Entry_Price": price,
                     "Current_Value_EUR": amount, "Broker_PnL_EUR": 0.0,
                     "Plan_EUR_month": None})
        return rows
    entry = float(row.get("Avg_Entry_Price") or 0)
    invested = _invested(row)
    shares = invested / entry if entry > 0 else 0.0
    new_invested = invested + amount
    new_shares = shares + amount / price
    # The profit is unchanged by a buy: value and invested both grow by the amount.
    row["Avg_Entry_Price"] = new_invested / new_shares
    row["Current_Value_EUR"] = float(row.get("Current_Value_EUR") or 0) + amount
    return rows


def _apply_sell(rows: list[dict], args: list[str]) -> list[dict]:
    usage = "/sell SYMBOL EUR|all [PRICE]"
    if len(args) not in (2, 3):
        raise ValueError(USAGE.format(usage=usage))
    sym = args[0]
    rows = [dict(r) for r in rows]
    row = _find(rows, sym)
    if row is None:
        raise ValueError(NOT_HELD.format(sym=sym.upper()))
    if args[1].lower() == "all":
        return [r for r in rows if r is not row]
    amount = _num(args[1])
    if amount <= 0:
        raise ValueError(USAGE.format(usage=usage))
    price = _num(args[2]) if len(args) == 3 else _live_price(str(row["Symbol"]))
    if not price or price <= 0:
        raise ValueError(NO_PRICE.format(sym=str(row["Symbol"]),
                                         example=f"/sell {row['Symbol']} {args[1]} 123,45"))
    entry = float(row.get("Avg_Entry_Price") or 0)
    invested = _invested(row)
    shares = invested / entry if entry > 0 else 0.0
    sold = amount / price
    if sold >= shares * (1 - 1e-6):
        return [r for r in rows if r is not row]
    value = max(0.0, float(row.get("Current_Value_EUR") or 0) - amount)
    # The average cost per share is unchanged by a sell.
    row["Current_Value_EUR"] = value
    row["Broker_PnL_EUR"] = value - (invested - sold * entry)
    return rows


def _diff_text(diff: dict) -> str:
    from quant.ui import copy as C

    summary = diff["summary"]
    if not summary["changed"]:
        return NO_CHANGES
    lines = [f"Review: bought {summary['bought_eur']:.0f} EUR, "
             f"sold {summary['sold_eur']:.0f} EUR, "
             f"realized {summary['realized_eur']:+.2f} EUR."]
    for ch in diff["changes"]:
        label = (C.DIFF_BOUGHT_SAVINGS if ch.get("savings_plan")
                 else f"Split {ch['ratio']:g}:1" if ch["kind"] == "Split"
                 else ch["kind"])
        amount = ch.get("amount_eur")
        lines.append(f"- {label} {ch['symbol']}"
                     + (f": {_fmt(amount)}" if amount else ""))
    lines.append(PENDING_HINT)
    return "\n".join(lines)


def _clean(rows: list[dict]):
    import pandas as pd

    from quant.portfolio.editor import validate_positions

    df = pd.DataFrame(rows, columns=_TABLE_COLS)
    if df["Plan_EUR_month"].isna().all():
        df = df.drop(columns=["Plan_EUR_month"])
    cleaned, _warnings, errors = validate_positions(df)
    return cleaned, errors


# ── Session ───────────────────────────────────────────────────────────────────

class Session:
    """What the bot remembers between two messages of one process."""

    def __init__(self) -> None:
        self.pending: dict | None = None   # {"rows": [...], "diff": {...}}
        self.todo: list[dict] = []         # the items of the last /todo


def _propose(session: Session, rows: list[dict]) -> str:
    from quant.engine.diff import diff_tables

    old = _saved_rows()
    _cleaned, errors = _clean(rows)
    if errors:
        session.pending = None
        return "\n".join(errors)
    diff = diff_tables(old, rows, prices={
        str(r["Symbol"]): p for r in old if (p := _live_price(str(r["Symbol"])))})
    session.pending = {"rows": rows, "diff": diff} if diff["summary"]["changed"] else None
    return _diff_text(diff)


def _cmd_confirm(session: Session) -> str:
    from quant.engine.confirm import confirm_save

    if session.pending is None:
        return NO_PENDING
    cleaned, errors = _clean(session.pending["rows"])
    if errors:
        return "\n".join(errors)
    res = confirm_save(cleaned, session.pending["diff"])
    session.pending = None
    if not res.get("ok"):
        return SAVE_FAILED
    return SAVED.format(n=res.get("changes", 0))


def _cmd_portfolio() -> str:
    from quant.engine.holdings_view import holdings_view

    holdings = holdings_view()
    if not holdings:
        return NO_HOLDINGS
    total = sum(float(h.get("value_eur") or 0) for h in holdings)
    profit = sum(float(h.get("profit_eur") or 0) for h in holdings)
    lines = [f"Portfolio {_fmt(total)}, profit {profit:+,.2f} EUR"]
    for h in sorted(holdings, key=lambda x: -float(x.get("value_eur") or 0)):
        tgt = h.get("target_weight")
        lines.append(
            f"- {h.get('name') or h['symbol']} [{h['symbol']}]: "
            f"{_fmt(h.get('value_eur'))}, profit {float(h.get('profit_eur') or 0):+,.2f} EUR, "
            f"{float(h.get('current_weight') or 0) * 100:.0f}%"
            + (f" of target {float(tgt) * 100:.0f}%" if tgt is not None else ""))
    return "\n".join(lines)


def _current_items() -> list[dict]:
    """The ONE list exactly as the Portfolio page shows it (read-only)."""
    from quant.engine import advice_state
    from quant.engine.holdings_view import holdings_view
    from quant.engine.today_list import build_items
    from quant.reporting.artifacts import read_regime

    items = build_items(holdings_view(), regime=read_regime())
    return advice_state.annotate(items)


def _cmd_todo(session: Session) -> str:
    from quant.engine.decisions import format_item_line

    session.todo = _current_items()
    if not session.todo:
        return TODO_EMPTY
    lines = []
    for n, it in enumerate(session.todo, 1):
        lines.append(f"{n}. [{it.get('horizon')}] {format_item_line(it)}"
                     + (f": {it['reason']}" if it.get("reason") else ""))
    lines.append("Send /done N after you placed the order.")
    return "\n".join(lines)


def _cmd_done(session: Session, args: list[str]) -> str:
    from quant.data.database import write_connection
    from quant.engine import advice_state
    from quant.engine.decisions import format_item_line

    if len(args) != 1:
        return USAGE.format(usage="/done N")
    if not session.todo:
        return TODO_FIRST
    try:
        n = int(args[0])
    except ValueError:
        return BAD_NUMBER.format(text=args[0])
    if not 1 <= n <= len(session.todo):
        return TODO_RANGE.format(n=n, last=len(session.todo))
    item = session.todo[n - 1]
    try:
        with write_connection() as conn:
            ok = advice_state.mark_item_done(conn, item)
    except Exception:  # noqa: BLE001
        ok = False
    if not ok:
        return DONE_FAILED
    session.todo = []   # the numbers no longer match the list
    return DONE.format(line=format_item_line(item))


def _cmd_history() -> str:
    from quant.engine.advice_state import STATUS_DONE, _query
    from quant.ui import copy as C

    trades = _query(None, "SELECT date, action, symbol, amount_eur FROM trades "
                          "ORDER BY date DESC LIMIT ?", [HISTORY_ROWS])
    done = _query(None, "SELECT done_at, verb, symbol, amount_eur FROM advice_log "
                        "WHERE status = ? ORDER BY done_at DESC LIMIT ?",
                  [STATUS_DONE, HISTORY_ROWS])
    if not trades and not done:
        return HISTORY_EMPTY
    lines = []
    if trades:
        lines.append("Trades:")
        lines += [f"- {C.fmt_date(d)} {a} {s} {_fmt(amt)}" for d, a, s, amt in trades]
    if done:
        lines.append("Done:")
        lines += [f"- {C.fmt_date(d)} {str(v).replace('_', ' ')} {s}"
                  + (f" {_fmt(amt)}" if amt else "") for d, v, s, amt in done]
    return "\n".join(lines)


def handle(text: str, session: Session) -> str:
    """The reply to one message. Never raises."""
    parts = str(text or "").strip().split()
    if not parts:
        return UNKNOWN
    cmd = parts[0].split("@", 1)[0].lower()   # "/todo@MyBot" in a group
    args = parts[1:]
    try:
        if cmd in ("/start", "/help"):
            return HELP
        if cmd == "/portfolio":
            return _cmd_portfolio()
        if cmd in ("/set", "/buy", "/sell"):
            apply = {"/set": _apply_set, "/buy": _apply_buy, "/sell": _apply_sell}[cmd]
            base = session.pending["rows"] if session.pending else _saved_rows()
            return _propose(session, apply(base, args))
        if cmd == "/confirm":
            return _cmd_confirm(session)
        if cmd == "/cancel":
            session.pending = None
            return CANCELLED
        if cmd == "/todo":
            return _cmd_todo(session)
        if cmd == "/done":
            return _cmd_done(session, args)
        if cmd == "/history":
            return _cmd_history()
        return UNKNOWN
    except ValueError as e:
        return str(e)
    except Exception:  # noqa: BLE001
        return SAVE_FAILED if cmd == "/confirm" else "Something went wrong. Try again."


# ── The polling loop ──────────────────────────────────────────────────────────

def poll_once(cfg: dict, session: Session, offset: int,
              api: Any = None) -> int:
    """Fetch and answer one batch of updates. Returns the next offset."""
    api = api or _api
    token = str(cfg.get("telegram_bot_token") or "")
    chat_id = str(cfg.get("telegram_chat_id") or "")
    body = api(token, "getUpdates", {"offset": offset, "timeout": POLL_TIMEOUT,
                                     "allowed_updates": '["message"]'},
               POLL_TIMEOUT + 10)
    if not body or not body.get("ok"):
        raise ConnectionError("getUpdates failed")
    for update in body.get("result") or []:
        offset = max(offset, int(update.get("update_id", 0)) + 1)
        _write_offset(offset)   # before acting: a crash never replays a command
        message = update.get("message") or {}
        if str((message.get("chat") or {}).get("id")) != chat_id:
            continue
        text = message.get("text")
        if not text:
            continue
        api(token, "sendMessage", {"chat_id": chat_id, "text": handle(text, session)}, 10)
    return offset


def run(config: dict | None = None) -> int:
    """Answer commands until interrupted. Returns an exit code."""
    from quant.engine import notify

    cfg = config if config is not None else notify.load_config()
    if (str(cfg.get("channel", "none")).lower() != "telegram"
            or not cfg.get("telegram_bot_token") or not cfg.get("telegram_chat_id")):
        print("Telegram is not set up; run quant notify-setup first.")
        return 2
    session = Session()
    offset = _read_offset()
    print("Telegram bot running. Send /help to your bot. Ctrl+C stops it.")
    try:
        while True:
            try:
                offset = poll_once(cfg, session, offset)
            except Exception:  # noqa: BLE001
                time.sleep(ERROR_BACKOFF)
    except KeyboardInterrupt:
        return 0


__all__ = ["Session", "handle", "poll_once", "run", "HELP"]
