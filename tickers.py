"""
tickers.py — watchlist, currency symbols, geo-risk tables, macro tickers.
Add or remove instruments here; the rest of the pipeline adapts automatically.
"""

# ─────────────────────────────────────────────────────────────────────────────
# WATCHLIST  (display name → Yahoo Finance ticker)
# ─────────────────────────────────────────────────────────────────────────────
TICKER_MAP = {
    # ── Uranium / Nuclear ────────────────────────────────────────────────────
    "Cameco":                    "CCJ",
    "NexGen Energy":             "NXE",
    "Centrus Energy":            "LEU",
    "BWX Technologies":          "BWXT",
    "Uranium Nuclear ETF":       "URA",
    "Sprott Uranium ETF":        "URNM",
    # ── Energy & Industrials ────────────────────────────────────────────────
    "Vistra":                    "VST",
    "RWE":                       "RWE.DE",
    "Siemens Energy":            "ENR.DE",
    "Constellation Energy":      "CEG",
    "Equinor":                   "EQNR",
    "Transocean":                "RIG",
    "Technip Energies":          "TE.PA",
    "Eni":                       "ENI.MI",
    # ── Defence ─────────────────────────────────────────────────────────────
    "BAE Systems":               "BA.L",
    "Rheinmetall":               "RHM.DE",
    # ── Cybersecurity ───────────────────────────────────────────────────────
    "Crowdstrike":               "CRWD",
    "Palo Alto Networks":        "PANW",
    # ── Agriculture / Chemicals ─────────────────────────────────────────────
    "Yara International":        "YAR.OL",
    "Bayer":                     "BAYN.DE",
    "Ecolab":                    "ECL",
    # ── Water & Environment ─────────────────────────────────────────────────
    "Veolia":                    "VIE.PA",
    "L&G Clean Water ETF":       "GLUG.L",
    # ── Materials / Mining ──────────────────────────────────────────────────
    "Albemarle":                 "ALB",
    "Zinnwald Lithium":          "ZNWD.L",
    "Neo Performance Materials": "NEO.TO",
    "Northern Star":             "NST.AX",
    "Heidelberg Materials":      "HEI.DE",
    # ── Photonics / Optics ──────────────────────────────────────────────────
    "Coherent":                  "COHR",
    "Lumentum":                  "LITE",
    # ── Logistics & Shipping ────────────────────────────────────────────────
    "Deutsche Post":             "DHL.DE",
    "Maersk A":                  "MAERSK-A.CO",  # A.P. Moller – Mærsk A/S
    "Hapag-Lloyd":               "HLAG.DE",
    # ── Real Estate / Finance ───────────────────────────────────────────────
    "Vonovia":                   "VNA.DE",
    "Raiffeisen Bank":           "RBI.VI",
    # ── Life Sciences ───────────────────────────────────────────────────────
    "Sartorius Vz":              "SRT.DE",
    # ── Technology ──────────────────────────────────────────────────────────
    "Samsung":                   "005930.KS",
    # ── Broad ETFs ──────────────────────────────────────────────────────────
    "Lithium&Battery ETF":       "LIT",
    "Physical Silver ETF":       "SSLN.L",
    "Physical Gold ETF":         "IAUP.L",
    "MSCI EM IMI ETF":           "EIMI.L",
    "MSCI World ETF":            "IWDA.AS",
    "SP500 Inverse ETF":         "SH",
}

# ─────────────────────────────────────────────────────────────────────────────
# CURRENCY DISPLAY SYMBOLS
# ─────────────────────────────────────────────────────────────────────────────
CURRENCY_SYMBOLS = {
    "USD": "$",  "EUR": "€",  "GBP": "£",  "GBX": "p",
    "NOK": "kr", "SEK": "kr", "DKK": "kr",
    "AUD": "A$", "CAD": "C$", "JPY": "¥",
    "KRW": "₩",  "CHF": "Fr",
}

# ─────────────────────────────────────────────────────────────────────────────
# GEO-RISK BASE SCORES  (1 = low risk, 10 = extreme risk)
# ─────────────────────────────────────────────────────────────────────────────
GEO_BASE = {
    "United States":  2, "Canada":        2, "Germany":       2,
    "France":         2, "United Kingdom": 2, "Netherlands":   2,
    "Denmark":        2, "Norway":         2, "Sweden":        2,
    "Australia":      2, "Switzerland":    3, "Japan":         3,
    "Italy":          3, "Austria":        3, "South Korea":   5,
    "China":          9, "Taiwan":         9, "Russia":       10,
    "Iran":          10,
}

GEO_KEYWORDS = [
    "sanctions", "trade war", "tariffs", "conflict", "regulatory", "ban",
    "export control", "war", "embargo", "restrict", "lawsuit", "antitrust",
]

# ─────────────────────────────────────────────────────────────────────────────
# MACRO REFERENCE TICKERS  (for correlation analysis)
# ─────────────────────────────────────────────────────────────────────────────
MACRO_TICKERS = {
    "US_10Y":    "^TNX",
    "OIL":       "CL=F",
    "GOLD":      "GC=F",
    "SP500":     "^GSPC",
    "TIPS_INF":  "TIP",
    "FED_PROXY": "^IRX",
}
