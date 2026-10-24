"""test_v10_8_0_palette.py — one palette module, no stray hex literals (v10.8.0, Phase 3).

Every color literal lives in quant/ui/palette.py. A hex color in another UI
module is a theme bug (it would not follow the light/night switch).
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "quant" / "ui"

_HEX = re.compile(r"#[0-9A-Fa-f]{6}\b")


def test_no_hex_literals_outside_the_palette():
    offenders = []
    for path in UI.glob("*.py"):
        if path.name == "palette.py":
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if _HEX.search(line):
                offenders.append(f"{path.name}:{i}: {line.strip()}")
    assert not offenders, "hex color literals outside palette.py:\n" + "\n".join(offenders)
