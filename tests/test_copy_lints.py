"""test_copy_lints.py — the ONE consolidated copy-correctness lint (v10.8.9, R-F3).

Intent: copy defects kept resurfacing one string at a time ("Telegram: Telegram
on, ...", the duplicated chart caption, doubled words, class synonyms, a bare "-"
coverage line, a parenthesized ticker next to an Instrument column, the "-0.0%"
badge) because there was no single lint module that bans the known bad patterns.
This module is that ONE lint. It scans three sources:

  1. the copy module ([`quant/ui/copy.py`](quant/ui/copy.py)) string constants;
  2. the rendered AppTest page text and dataframes;
  3. the Telegram message builder ([`quant/engine/notify.py`](quant/engine/notify.py)).

A parametrized self-test injects each bad pattern into a fixture string and
asserts the scanner flags it, so a silent regression in the lint itself fails.

Synthetic fixtures only; the user's real DB and data dir are never touched.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_QUANT = _ROOT / "quant"
COPY_PY = _QUANT / "ui" / "copy.py"
DASHBOARD = str(_QUANT / "dashboard.py")

# ── The ONE pattern registry (every known bad copy pattern) ───────────────────
# Each entry: (name, compiled regex). The name is what the self-test injects.
BAD_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Doubled words, incl. "EUR EUR" and a doubled currency/word.
    ("doubled_word", re.compile(r"\b([A-Za-z]{3,})\s+\1\b")),
    # "Telegram: Telegram on, ..." (F4).
    ("telegram_doubled", re.compile(r"Telegram:\s*Telegram")),
    # Adjacent paren groups, incl. a doubled dotted ticker.
    ("adjacent_parens", re.compile(r"\)\s*\(")),
    ("doubled_paren_ticker", re.compile(r"\([\w]+\)\s*\([\w.]+\)")),
    # Class synonyms outside the terminology map (A1).
    ("class_synonym_small_bets", re.compile(r"Small bets")),
    ("class_synonym_fortress", re.compile(r"\bFORTRESS\b")),
    # A bare "-" coverage line (R-P2).
    ("bare_coverage_dash", re.compile(r"coverage:\s*-")),
    # The unrepresentable "-0.0%" badge (R-P3).
    ("minus_zero_pct", re.compile(r"-0\.0%")),
]


def scan(text: str) -> list[tuple[str, str]]:
    """Return ``[(pattern_name, matched_text), ...]`` for every hit in ``text``."""
    hits: list[tuple[str, str]] = []
    for name, pat in BAD_PATTERNS:
        m = pat.search(text or "")
        if m:
            hits.append((name, m.group(0)))
    return hits


def _string_constants(path: Path) -> list[str]:
    """Every user-facing string literal, excluding dict keys and docstrings.

    Docstrings and comments may legitimately QUOTE a banned pattern (to document
    it); only rendered string values are scanned.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    excluded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k in node.keys:
                if isinstance(k, ast.Constant):
                    excluded.add(id(k))
        if isinstance(node, ast.Module | ast.FunctionDef | ast.AsyncFunctionDef
                      | ast.ClassDef):
            body = getattr(node, "body", [])
            first = body[0] if body else None
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                excluded.add(id(first.value))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in excluded:
                continue
            out.append(node.value)
    return out


def _scan_constants(path: Path) -> list[str]:
    offenders: list[str] = []
    for value in _string_constants(path):
        if scan(value):
            offenders.append(value)
    return offenders


# ── Source 1: the copy module ─────────────────────────────────────────────────

def test_copy_module_strings_have_no_bad_patterns():
    """R-F3: no banned pattern in any copy string value."""
    offenders = _scan_constants(COPY_PY)
    assert not offenders, f"banned copy pattern(s): {offenders}"


# ── Source 3: the Telegram message builders ───────────────────────────────────

@pytest.mark.parametrize("rel", ["engine/notify.py"])
def test_telegram_builders_have_no_bad_patterns(rel):
    """R-F3: no banned pattern in a Telegram/Discord message builder string."""
    offenders = _scan_constants(_QUANT / rel)
    assert not offenders, f"banned pattern(s) in {rel}: {offenders}"


# ── Source 2: the rendered pages (AppTest) ────────────────────────────────────

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

_PAGES = {
    "Portfolio": "pages/portfolio.py",
    "Update holdings": "pages/update.py",
    "History": "pages/history.py",
    "Full analysis": "pages/analysis.py",
    "Settings": "pages/settings.py",
}

_TEXTLIKE = ("markdown", "info", "warning", "error", "caption", "text",
             "title", "header", "subheader")


def _run(page: str) -> AppTest:
    at = AppTest.from_file(DASHBOARD, default_timeout=90)
    at.run()
    at.switch_page(page).run()
    return at


def _page_text(at: AppTest) -> str:
    chunks = []
    for attr in _TEXTLIKE:
        for el in getattr(at, attr, []):
            chunks.append(str(getattr(el, "value", "")))
    for el in getattr(at, "dataframe", []):
        v = getattr(el, "value", None)
        if v is not None:
            try:
                chunks.append(str(v.to_string()))
            except Exception:  # noqa: BLE001
                chunks.append(str(v))
    return " ".join(chunks)


@pytest.mark.parametrize("page", list(_PAGES))
def test_rendered_pages_have_no_bad_patterns(page):
    """R-F3: no banned copy pattern on any rendered page."""
    at = _run(_PAGES[page])
    assert not at.exception, f"{page} raised: {at.exception}"
    text = _page_text(at)
    hits = scan(text)
    assert not hits, f"banned copy pattern(s) on {page}: {hits}"


# ── The structural rule: a Name cell never carries a parenthesized ticker
#    next to an Instrument column (A1) ────────────────────────────────────────

def check_name_cells(df) -> list[str]:
    """Violations from a table whose ``Name`` sits next to an ``Instrument`` key.

    A ``Name`` cell must not embed the row's ticker, nor a parenthesized suffix,
    when a separate Instrument/Symbol column carries it.
    """
    if df is None or "Name" not in getattr(df, "columns", []):
        return []
    sym_key = "Instrument" if "Instrument" in df.columns else "Symbol"
    if sym_key not in df.columns:
        return []
    bad: list[str] = []
    for _, r in df.iterrows():
        name = str(r.get("Name") or "")
        sym = str(r.get(sym_key) or "")
        if not name or not sym:
            continue
        if sym.upper() in name.upper() or f"({sym})" in name \
                or f"({sym.split('.')[0]})" in name:
            bad.append(name)
    return bad


def test_scan_flags_a_name_cell_with_a_parenthesized_ticker():
    """Self-test: a Name cell embedding the ticker is flagged."""
    import pandas as pd

    clean = pd.DataFrame([{"Name": "iShares World", "Symbol": "EUNL.DE"}])
    dirty = pd.DataFrame([{"Name": "iShares World (EUNL.DE)", "Symbol": "EUNL.DE"}])
    assert check_name_cells(clean) == []
    assert check_name_cells(dirty) == ["iShares World (EUNL.DE)"]


# ── The parametrized self-test: injecting each bad pattern fails the lint ─────

_INJECTIONS = {
    "doubled_word": "Amount EUR EUR added",
    "telegram_doubled": "Telegram: Telegram on, last test ok",
    "adjacent_parens": "Acme Fund (ACME) (ACME.DE)",
    "doubled_paren_ticker": "Global Aero & Def (5J50) (5J50.DE)",
    "class_synonym_small_bets": "Small bets are high risk",
    "class_synonym_fortress": "assigned tier FORTRESS today",
    "bare_coverage_dash": "coverage: -",
    "minus_zero_pct": "0 EUR (-0.0%)",
}


@pytest.mark.parametrize("name,text", list(_INJECTIONS.items()))
def test_injecting_a_bad_pattern_makes_the_lint_fail(name, text):
    """Acceptance: each injected bad pattern is flagged by the scanner."""
    hits = scan(text)
    assert any(h[0] == name for h in hits), \
        f"the lint missed the injected pattern {name!r} in {text!r}"


def test_a_clean_string_passes_the_lint():
    """Acceptance: a clean string produces no hits."""
    assert scan("Telegram: on, last test ok") == []
    assert scan("coverage: 3/5 symbols, as of 13 Oct 2026") == []


# ── Aliases: the scattered guard names now resolve to this ONE lint ───────────

def test_no_duplicated_ticker_or_word_in_the_ui():
    """Alias for the former duplicated ticker/word guard."""
    for page in _PAGES.values():
        at = _run(page)
        assert not at.exception, f"{page} raised: {at.exception}"
        hits = [h for h in scan(_page_text(at))
                if h[0] in ("doubled_word", "adjacent_parens",
                            "doubled_paren_ticker")]
        assert not hits, f"{hits} on {page}"


def test_no_banned_class_synonyms_in_copy():
    """Alias for the former class-synonym copy guard."""
    offenders = [v for v in _string_constants(COPY_PY)
                 if scan(v) and any(h[0].startswith("class_synonym") for h in scan(v))]
    assert not offenders, f"banned class synonym in copy: {offenders}"


@pytest.mark.parametrize("page", ["pages/portfolio.py", "pages/analysis.py"])
def test_rendered_pages_have_no_banned_class_words(page):
    """Alias for the former rendered class-word guard."""
    at = _run(page)
    assert not at.exception, f"{page} raised: {at.exception}"
    text = _page_text(at)
    assert "Small bets" not in text, f"'Small bets' rendered on {page}"
    assert not re.search(r"\bFORTRESS\b", text), f"'FORTRESS' rendered on {page}"


# ── Source 4: UI f-strings (paths the AppTests may not render) ────────────────

_EUR_FORMATTERS = {"fmt_eur", "fmt_eur_whole"}


def _doubled_currency_fstrings(path: Path) -> list[int]:
    """Lines where an f-string puts literal "EUR" right after a ``fmt_eur*`` value.

    ``fmt_eur`` already appends " EUR", so ``f"{C.fmt_eur(x)} EUR/m"`` renders
    "12.30 EUR EUR/m" (v10.8.10). The rendered-page scan only sees paths the
    AppTests reach; this static check covers the rest.
    """
    lines: list[int] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.JoinedStr):
            continue
        for cur, nxt in zip(node.values, node.values[1:], strict=False):
            call = cur.value if isinstance(cur, ast.FormattedValue) else None
            name = getattr(getattr(call, "func", None), "attr", None) or getattr(
                getattr(call, "func", None), "id", None)
            if (name in _EUR_FORMATTERS and isinstance(nxt, ast.Constant)
                    and str(nxt.value).lstrip().startswith("EUR")):
                lines.append(node.lineno)
    return lines


def test_no_doubled_currency_in_ui_fstrings():
    offenders = {str(p.relative_to(_QUANT)): _doubled_currency_fstrings(p)
                 for p in sorted((_QUANT / "ui").glob("*.py"))}
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, f"fmt_eur(...) followed by a literal EUR: {offenders}"


def test_doubled_currency_check_catches_the_pattern(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text('x = f"+{C.fmt_eur(1)} EUR/m"\n', encoding="utf-8")
    assert _doubled_currency_fstrings(bad) == [1]
