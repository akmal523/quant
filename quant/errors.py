"""
errors.py — Standardized exception hierarchy (v10.6.5).

Intent: one base exception so callers can catch every quant error with a single
``except QuantError``, plus specific subclasses for the common failure domains.
Existing functions keep their current contracts (many return empty frames rather
than raising); these types are for new code and for callers that want to be
explicit.

Invariants: pure module; no I/O; no imports from the pipeline.
"""
from __future__ import annotations


class QuantError(Exception):
    """Base exception for all quant system errors."""


class DataError(QuantError):
    """Data is missing, corrupted, or invalid."""


class TierError(QuantError):
    """Tier assignments are invalid or inconsistent."""


class AllocationError(QuantError):
    """Tier allocations violate their constraints."""


class CacheError(QuantError):
    """A cache operation failed."""


class DatabaseError(QuantError):
    """A database operation failed."""


class ExternalAPIError(QuantError):
    """An external API (yfinance, SEC EDGAR) failed."""
