"""palette.py — the ONE place with color literals (v10.8.0, Phase 3).

Intent: charts and tables take their colors from here, so the light and night
themes stay readable and no UI module hardcodes a hex value. Semantic colors
carry meaning only: green for gain, red for loss and blockers, amber for
warnings. Composition and decoration use the neutral series.

Invariants:
  - Every color is a named constant.
  - No other UI module contains a hex color literal (enforced by a test).
"""
from __future__ import annotations

# Accent (the theme primary).
ACCENT = "#1F3B73"
ACCENT_LIGHT = "#5B83BF"

# Text.
TEXT = "#1F2937"
TEXT_MUTED = "#6B7280"

# Semantic colors: meaning only, never decoration.
GAIN = "#1B7F3B"
LOSS = "#B00020"
WARNING = "#B26A00"

# Neutral series for composition (categorical, never semantic).
SERIES = ["#1F3B73", "#3B5C99", "#5B83BF", "#8FA9CF", "#B8C4D9",
          "#6B7280", "#8A8F99", "#A7ADB8"]

# Chart chrome.
BASELINE = "#888888"
BENCHMARK = "#8A8F99"
ANNOTATION_BG = "rgba(255,255,255,0.85)"
ANNOTATION_TEXT = "#1F2937"
TRANSPARENT = "rgba(0,0,0,0)"
