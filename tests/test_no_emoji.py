"""
test_no_emoji.py — No-Emoji Lint (Phase 5 / v10.2).

Intent: enforce the rule that the UI and source contain zero emoji characters.
Scans dashboard.py, notifier.py, reporting.py, main.py source with a regex over
the emoji unicode ranges and fails on any match.
"""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import re
import unittest

# Emoji unicode ranges: pictographs + misc symbols + dingbats.
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF]"
)

FILES = [
    "quant/dashboard.py",
    "quant/reporting/notifier.py",
    "quant/reporting/reporting.py",
    "quant/reporting/reporting_advanced.py",
    "quant/reporting/web.py",
    "quant/reporting/briefing.py",
    "quant/reporting/actions.py",
    "quant/cli/__init__.py",
    "quant/cli/output.py",
    "quant/main.py",
    # v10.5.1: central copy module + UI helpers.
    "quant/ui/copy.py",
    "quant/ui/runner.py",
    "quant/ui/search.py",
]


class TestNoEmoji(unittest.TestCase):
    def test_no_emoji_in_source(self):
        failures = []
        for fname in FILES:
            try:
                with open(fname, encoding="utf-8") as f:
                    content = f.read()
            except FileNotFoundError:
                continue
            for i, line in enumerate(content.splitlines(), 1):
                if EMOJI_RE.search(line):
                    failures.append(f"{fname}:{i}: {line}")
        self.assertEqual(failures, [], "emoji characters found:\n" + "\n".join(failures))


if __name__ == "__main__":
    unittest.main()