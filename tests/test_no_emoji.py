"""
test_no_emoji.py — No-Emoji Lint (v10.6.3).

Intent: enforce the rule that the whole project (code, docs, strings, tests)
contains zero emoji characters. Scans every text file under the repo root,
excluding third-party and generated directories, with a regex over the emoji
unicode ranges.

Invariants:
  - No emoji in any scanned file.
  - Third-party (.venv, myenv, site, node_modules) and generated (outputs,
    __pycache__, htmlcov) directories are excluded.
  - User data (data/) is excluded: it is not project source.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Emoji unicode ranges: pictographs/emoticons, misc symbols, dingbats,
# misc symbols and arrows, and the variation selector-16 (emoji presentation).
EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF\U0000FE0F]"
)

ROOT = Path(__file__).resolve().parents[1]

# Directories never scanned: third-party, generated, or user-owned data.
EXCLUDE_DIRS = {
    ".git", ".venv", "myenv", "site", "node_modules", "__pycache__",
    "htmlcov", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "outputs", "sec_filings", "data", "dist", "build",
}

# Text file types that must be emoji-free.
SCAN_SUFFIXES = {
    ".py", ".md", ".toml", ".yml", ".yaml", ".cfg", ".txt",
    ".html", ".csv", ".json", ".ini",
}


def _iter_files():
    """Yield every scannable text file under the repo root."""
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDE_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in SCAN_SUFFIXES:
            continue
        yield path


def test_no_emoji_in_project():
    """Fail if any emoji character appears in any scanned project file."""
    failures: list[str] = []
    for path in _iter_files():
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(content.splitlines(), 1):
            if EMOJI_RE.search(line):
                failures.append(f"{path.relative_to(ROOT)}:{i}: {line.strip()}")
    assert failures == [], "emoji characters found:\n" + "\n".join(failures)
