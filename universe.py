"""
universe.py — Sector-based dynamic asset universe.
Replaces the static tickers.py. Each sector maps to {name: yf_symbol}.
"""
from __future__ import annotations

# ─── Strategic Sector Universe ────────────────────────────────────────────────

SECTOR_UNIVERSE: dict[str, dict[str, str]] = {
    "Uranium / Nuclear": {
        "Cameco":          "CCJ",
        "NexGen Energy":   "NXE",
        "Centrus Energy":  "LEU",
        "BWX Technologies":"BWXT",
        "URA ETF":         "URA",
        "URNM ETF":        "URNM",
    },
    "Energy": {
        "Vistra":            "VST",
        "RWE":               "RWE.DE",
        "Siemens Energy":    "ENR.DE",
        "Constellation Energy": "CEG",
        "Equinor":           "EQNR",
        "Transocean":        "RIG",
        "Technip Energies":  "TE.PA",
        "Eni":               "ENI.MI",
    },
    "Defense": {
        "BAE Systems":      "BA.L",
        "Rheinmetall":      "RHM.DE",
        "Northrop Grumman": "NOC",
        "L3Harris":         "LHX",
        "Leonardo":         "LDO.MI",
    },
    "Cybersecurity": {
        "CrowdStrike":        "CRWD",
        "Palo Alto Networks": "PANW",
        "SentinelOne":        "S",
        "Fortinet":           "FTNT",
    },
    "Gold / Precious Metals": {
        "Newmont":            "NEM",
        "Barrick Gold":       "GOLD",
        "Agnico Eagle":       "AEM",
        "Northern Star":      "NST.AX",
        "VanEck Gold Miners": "GDX",
    },
    "Quantum Computing": {
        "IBM":              "IBM",
        "IonQ":             "IONQ",
        "Rigetti Computing":"RGTI",
        "D-Wave Quantum":   "QBTS",
    },
    "Logistics / Shipping": {
        "Deutsche Post":    "DPW.DE",
        "AP Moller-Maersk": "MAERSK-A.CO",
        "Hapag-Lloyd":      "HLAG.DE",
        "FedEx":            "FDX",
    },
    "Materials / Mining": {
        "Albemarle":              "ALB",
        "Heidelberg Materials":   "HEI.DE",
        "Neo Performance":        "NEO.TO",
        "Zinnwald Lithium":       "ZNWD.L",
    },
    "Agriculture / Chemicals": {
        "Yara International": "YAR.OL",
        "Bayer":              "BAYN.DE",
        "Ecolab":             "ECL",
    },
    "Water / Environment": {
        "Veolia":              "VIE.PA",
        "L&G Clean Water ETF": "GLUG.L",
    },
    "Life Sciences": {
        "Sartorius Vz":   "SRT3.DE",
        "Thermo Fisher":  "TMO",
    },
    "Finance": {
        "Vonovia":          "VNA.DE",
        "Raiffeisen Bank":  "RBI.VI",
    },
    "Technology": {
        "Samsung": "005930.KS",
    },
    "Broad ETFs": {
        "MSCI World (IWDA)":    "IWDA.AS",
        "EM IMI (EIMI)":        "EIMI.L",
        "LIT Lithium ETF":      "LIT",
        "Short S&P500 (SH)":    "SH",
        "Global Clean Energy":  "ICLN",
    },
}

# ─── Public API ───────────────────────────────────────────────────────────────

def get_market_universe() -> dict[str, str]:
    """Return flat {name: symbol} dict across all sectors."""
    flat: dict[str, str] = {}
    for assets in SECTOR_UNIVERSE.values():
        flat.update(assets)
    return flat


def symbol_to_sector(symbol: str) -> str:
    """Reverse-lookup sector from yf symbol. Falls back to 'Unknown'."""
    for sector, assets in SECTOR_UNIVERSE.items():
        if symbol in assets.values():
            return sector
    return "Unknown"


# ─── Geo Risk Tables ──────────────────────────────────────────────────────────

GEO_BASE: dict[str, int] = {
    "United States":  2,
    "Germany":        2,
    "France":         2,
    "United Kingdom": 2,
    "Norway":         2,
    "Canada":         2,
    "Australia":      3,
    "Sweden":         2,
    "Denmark":        2,
    "Netherlands":    2,
    "Switzerland":    2,
    "Italy":          3,
    "South Korea":    5,
    "China":          6,
    "Russia":         9,
    "Ukraine":        9,
    "Israel":         7,
    "Iran":           10,
    "Brazil":         5,
    "India":          4,
    "Austria":        3,   # RBI — Russian exposure
}

GEO_KEYWORDS: list[str] = [
    "sanction", "war", "conflict", "invasion", "default",
    "recession", "fraud", "investigation", "ban", "tariff",
    "embargo", "collapse", "crisis", "downgrade", "lawsuit",
]

# ─── Currency Display Symbols ─────────────────────────────────────────────────

CURRENCY_SYMBOLS: dict[str, str] = {
    "USD": "$",  "EUR": "€",  "GBP": "£",  "GBX": "p",
    "NOK": "kr", "DKK": "kr", "SEK": "kr", "AUD": "A$",
    "CAD": "C$", "JPY": "¥",  "KRW": "₩",  "CHF": "Fr",
}
