"""
identifiers.py — Financial identifier validation (v10.5.2, F1).

Intent: F1 — the codebase never invents, guesses, or hardcodes financial
identifiers (ISIN, CUSIP, SEDOL). Identifiers enter only through user files,
validated live metadata from a named source, or a curated file with an audit
trail. This module is the mechanical gate: any ISIN that fails validation is
rejected everywhere (repair refuses to write it; a curated row that fails aborts
repair with a plain error naming the row).

Invariants:
  - is_valid_isin returns True iff the value is a well-formed ISO 6166 ISIN with
    a correct Luhn mod-36 check digit.
  - Pure function; no I/O.

Dependencies: none.
"""
from __future__ import annotations

import re

# ISO 6166 shape: two-letter country prefix, nine alphanumerics, one check digit.
_ISIN_RE = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")


def is_valid_isin(value: str | None) -> bool:
    """Validate an ISIN (ISO 6166 shape + Luhn mod-36 check digit).

    Shape: two-letter country prefix, nine alphanumerics, one check digit.
    Check: expand each character to its base-36 decimal digits (A=10 .. Z=35),
    then apply the Luhn algorithm; the total must be divisible by 10.
    """
    if not value or not isinstance(value, str):
        return False
    # Uppercase is required (ISO 6166); lowercase is rejected, not normalised.
    value = value.strip()
    if not _ISIN_RE.match(value):
        return False
    body, check = value[:-1], int(value[-1])
    digits = "".join(str(int(c, 36)) for c in body)
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (total + check) % 10 == 0
