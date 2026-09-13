"""
web.py — Static Published Briefing (v10.5.0, spec 5).

Intent: render a single, minimalist, read-only HTML page from the latest run
artifacts and write it to outputs/run_<ts>/web/index.html plus data.json. The
page is deployed to GitHub Pages by a scheduled Action; no paid server, no
ephemeral filesystem, no secrets.

Invariants:
  - publish() never raises on missing artifacts; it renders empty states.
  - Every number carries a source line (R3); no charts in v1 (spec 5.1).
  - outputs/run_latest is a symlink (copy fallback) to the published run.
  - Pure-ish: reads artifacts, writes index.html + data.json.

Dependencies: quant.reporting.actions, quant.portfolio.account, quant.paths.
"""
from __future__ import annotations

import html
import json
import os
import shutil

import pandas as pd

from quant import paths
from quant import __version__
from quant.reporting.actions import build_actions
from quant.reporting.artifacts import latest_run
from quant.portfolio.account import load_account

_CSS = """
:root { color-scheme: light dark; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       max-width: 880px; margin: 0 auto; padding: 24px; line-height: 1.5; }
h1 { font-size: 1.5rem; } h2 { font-size: 1.1rem; margin-top: 1.6rem; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid #8884; }
.muted { color: #888; font-size: 0.85rem; }
.blocked { color: #b00; }
"""


def _esc(v) -> str:
    return html.escape(str(v))


def _read_audit(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "portfolio_audit.csv")
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _read_metrics(run_dir: str) -> dict:
    path = os.path.join(run_dir, "metrics.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def build_data(run_dir: str) -> dict:
    """Assemble the data.json payload from run artifacts."""
    audit = _read_audit(run_dir)
    account = load_account()
    actions = build_actions(audit)
    metrics = _read_metrics(run_dir)

    total_value = 0.0
    pnl = 0.0
    if not audit.empty:
        if "Value_EUR" in audit:
            total_value = float(audit["Value_EUR"].sum())
        if "Real_PnL_EUR" in audit:
            pnl = float(audit["Real_PnL_EUR"].sum())

    holdings = []
    if not audit.empty:
        for _, r in audit.iterrows():
            holdings.append({
                "symbol": str(r.get("Symbol", "")),
                "tier": str(r.get("Tier", "")),
                "weight": str(r.get("Current_Weight", "")),
                "target": str(r.get("Target_Weight", "")),
                "drift": str(r.get("Drift", "")),
                "status": str(r.get("Signal", "")),
            })

    watchlist = []
    if not audit.empty and "Active_Score" in audit:
        top = audit.sort_values("Active_Score", ascending=False).head(5)
        for _, r in top.iterrows():
            watchlist.append({
                "symbol": str(r.get("Symbol", "")),
                "score": float(r.get("Active_Score", 0) or 0),
                "reason": str(r.get("Recommendation", "")),
            })

    return {
        "version": __version__,
        "as_of": os.path.basename(run_dir),
        "regime": metrics.get("market_regime", ""),
        "portfolio": {
            "value_eur": round(total_value, 2),
            "pnl_eur": round(pnl, 2),
            "cash_eur": account.cash_eur,
            "risk_profile": account.risk_profile,
        },
        "actions": actions,
        "holdings": holdings,
        "watchlist": watchlist,
        "blockers": [a["remedy"] for a in actions if a["blocked"]],
    }


def render_html(data: dict) -> str:
    """Render the single-page HTML (spec 5.1 section order)."""
    p = data["portfolio"]
    cash = (f"{p['cash_eur']:.2f} EUR" if p["cash_eur"] is not None
            else "Cash not set. Set it in Portfolio to enable cash-aware recommendations.")

    parts: list[str] = []
    parts.append("<!doctype html><html lang='en'><head><meta charset='utf-8'>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")
    parts.append(f"<title>Quant-AI Briefing {_esc(data['version'])}</title>")
    parts.append(f"<style>{_CSS}</style></head><body>")

    # 1. Header.
    parts.append("<h1>Quant-AI Briefing</h1>")
    parts.append(
        f"<p class='muted'>As-of {_esc(data['as_of'])}. Version {_esc(data['version'])}. "
        f"<a href='https://github.com/akmal523/quant'>Repository</a></p>"
    )

    # 2. Portfolio.
    parts.append("<h2>Portfolio</h2>")
    parts.append(f"<p>Value {p['value_eur']:.2f} EUR (source: portfolio.csv). "
                 f"Broker PnL {p['pnl_eur']:+.2f} EUR (source: broker). "
                 f"Cash {_esc(cash)}. Risk profile {_esc(p['risk_profile'])} "
                 f"(source: account.yaml). Regime {_esc(data['regime'] or 'not estimated')}.</p>")

    # 3. Actions.
    parts.append("<h2>Actions</h2>")
    if data["actions"]:
        parts.append("<table><tr><th>Symbol</th><th>Action</th><th>Amount EUR</th>"
                     "<th>Reason</th></tr>")
        for a in data["actions"]:
            if a["blocked"]:
                parts.append(f"<tr class='blocked'><td>{_esc(a['symbol'])}</td>"
                             f"<td>BLOCKED</td><td></td><td>{_esc(a['reason'])}</td></tr>")
            else:
                sign = "+" if a["action"] == "BUY MORE" else "-"
                parts.append(f"<tr><td>{_esc(a['symbol'])}</td><td>{_esc(a['action'])}</td>"
                             f"<td>{sign}{a['amount_eur']:.0f}</td>"
                             f"<td>{_esc(a['reason'])}</td></tr>")
        parts.append("</table>")
    else:
        parts.append("<p>No actions required today.</p>")

    # 4. Holdings.
    parts.append("<h2>Holdings</h2>")
    if data["holdings"]:
        parts.append("<table><tr><th>Symbol</th><th>Tier</th><th>Weight</th>"
                     "<th>Target</th><th>Drift</th><th>Status</th></tr>")
        for h in data["holdings"]:
            parts.append(
                f"<tr><td>{_esc(h['symbol'])}</td><td>{_esc(h['tier'])}</td>"
                f"<td>{_esc(h['weight'])}</td><td>{_esc(h['target'])}</td>"
                f"<td>{_esc(h['drift'])}</td><td>{_esc(h['status'])}</td></tr>"
            )
        parts.append("</table>")
    else:
        parts.append("<p>No holdings.</p>")

    # 5. Watchlist highlights.
    parts.append("<h2>Watchlist highlights</h2>")
    if data["watchlist"]:
        parts.append("<table><tr><th>Symbol</th><th>Active score</th><th>Reason</th></tr>")
        for w in data["watchlist"]:
            parts.append(f"<tr><td>{_esc(w['symbol'])}</td><td>{w['score']:.1f}</td>"
                         f"<td>{_esc(w['reason'])}</td></tr>")
        parts.append("</table>")
    else:
        parts.append("<p>No watchlist highlights.</p>")

    # 6. Data health (blockers only).
    parts.append("<h2>Data health</h2>")
    if data["blockers"]:
        parts.append("<ul>" + "".join(f"<li>{_esc(b)}</li>" for b in data["blockers"]) + "</ul>")
    else:
        parts.append("<p>No blockers.</p>")

    # 7. Footer.
    parts.append("<h2>Methodology</h2>")
    parts.append(
        "<p class='muted'>Scores combine structural quality, tactical timing, and news "
        "sentiment. Portfolio PnL is copied from the broker, never price-guessed. Actions "
        "cite the drift threshold from quant/config.py that triggered them.</p>"
    )
    parts.append(
        "<p class='muted'>All output is for informational purposes. Probabilistic models "
        "and NLP sentiment analysis involve inherent risk. Past performance does not "
        "guarantee future results.</p>"
    )
    parts.append(f"<p class='muted'>generated by quant publish {_esc(data['version'])} "
                 f"at {_esc(data['as_of'])}</p>")
    parts.append("</body></html>")
    return "".join(parts)


def publish(run_dir: str | None = None, out_dir: str | None = None) -> int:
    """Render the Published Briefing. Returns an exit code (0 success)."""
    run_dir = run_dir or latest_run()
    if not run_dir or not os.path.isdir(run_dir):
        print("error: no run artifacts found.")
        print("remedy: run quant run first, then quant publish.")
        return 1

    data = build_data(run_dir)
    web_dir = out_dir or os.path.join(run_dir, "web")
    os.makedirs(web_dir, exist_ok=True)

    with open(os.path.join(web_dir, "data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    with open(os.path.join(web_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(render_html(data))

    _link_latest(run_dir)
    print(f"published {os.path.join(web_dir, 'index.html')}")
    return 0


def _link_latest(run_dir: str) -> None:
    """Point outputs/run_latest at the published run (symlink, copy fallback)."""
    link = os.path.join(str(paths.OUTPUTS_DIR), "run_latest")
    try:
        if os.path.islink(link) or os.path.exists(link):
            if os.path.islink(link):
                os.remove(link)
            else:
                shutil.rmtree(link, ignore_errors=True)
        os.symlink(run_dir, link)
    except (OSError, NotImplementedError):
        # Symlinks unavailable: copy the web dir instead.
        try:
            shutil.rmtree(link, ignore_errors=True)
            shutil.copytree(os.path.join(run_dir, "web"), link)
        except Exception:
            pass
