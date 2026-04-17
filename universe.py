"""
universe.py — Sector-based dynamic asset universe (~300 instruments, 20 sectors).
Each sector maps to {display_name: yf_symbol}.

Sectors:
  Uranium / Nuclear · Energy · Oil & Gas · Defense · Cybersecurity
  Gold / Precious Metals · Silver & Royalties · Copper & Battery Metals
  Lithium · Quantum Computing · Semiconductors · AI & Cloud
  Logistics / Shipping · Finance & Banking · Insurance
  Healthcare / Pharma · Water & Environment · Agriculture / Chemicals
  Real Estate (REIT) · Broad ETFs
"""
from __future__ import annotations

SECTOR_UNIVERSE: dict[str, dict[str, str]] = {

    # ── Uranium / Nuclear ─────────────────────────────────────────────────────
    "Uranium / Nuclear": {
        "Cameco":                "CCJ",
        "NexGen Energy":         "NXE",
        "Centrus Energy":        "LEU",
        "BWX Technologies":      "BWXT",
        "Uranium Energy":        "UEC",
        "Denison Mines":         "DNN",
        "Paladin Energy":        "PDN.AX",
        "Boss Energy":           "BOE.AX",
        "Yellow Cake":           "YCA.L",
        "Kazatomprom":           "KAP.L",
        "Global Atomic":         "GLO.TO",
        "URA ETF":               "URA",
        "URNM ETF":              "URNM",
    },

    # ── Energy (Utilities / Renewables) ───────────────────────────────────────
    "Energy": {
        "Vistra":                "VST",
        "Constellation Energy":  "CEG",
        "RWE":                   "RWE.DE",
        "Siemens Energy":        "ENR.DE",
        "Equinor":               "EQNR",
        "Technip Energies":      "TE.PA",
        "Eni":                   "ENI.MI",
        "Enel":                  "ENEL.MI",
        "Iberdrola":             "IBE.MC",
        "Orsted":                "ORSTED.CO",
        "Vestas Wind":           "VWS.CO",
        "NextEra Energy":        "NEE",
        "Brookfield Renewable":  "BEPC",
        "AES Corporation":       "AES",
        "Global Clean Energy":   "ICLN",
    },

    # ── Oil & Gas ─────────────────────────────────────────────────────────────
    "Oil & Gas": {
        "Shell":                 "SHEL",
        "TotalEnergies":         "TTE.PA",
        "BP":                    "BP.L",
        "ExxonMobil":            "XOM",
        "Chevron":               "CVX",
        "ConocoPhillips":        "COP",
        "Schlumberger (SLB)":    "SLB",
        "Halliburton":           "HAL",
        "Transocean":            "RIG",
        "Vaar Energi":           "VAR.OL",
        "Aker BP":               "AKRBP.OL",
        "Cenovus Energy":        "CVE.TO",
        "Suncor Energy":         "SU.TO",
        "Devon Energy":          "DVN",
        "Energy Select (XLE)":   "XLE",
    },

    # ── Defense ───────────────────────────────────────────────────────────────
    "Defense": {
        "Rheinmetall":           "RHM.DE",
        "BAE Systems":           "BA.L",
        "Leonardo":              "LDO.MI",
        "Thales":                "HO.PA",
        "Airbus":                "AIR.PA",
        "Safran":                "SAF.PA",
        "Dassault Aviation":     "AM.PA",
        "Northrop Grumman":      "NOC",
        "Lockheed Martin":       "LMT",
        "Raytheon Technologies": "RTX",
        "L3Harris":              "LHX",
        "General Dynamics":      "GD",
        "Textron":               "TXT",
        "Leidos":                "LDOS",
        "HEICO":                 "HEI",
        "ITA Defense ETF":       "ITA",
        "EUAD EU Defense ETF":   "EUAD.L",
    },

    # ── Cybersecurity ─────────────────────────────────────────────────────────
    "Cybersecurity": {
        "CrowdStrike":           "CRWD",
        "Palo Alto Networks":    "PANW",
        "SentinelOne":           "S",
        "Fortinet":              "FTNT",
        "Zscaler":               "ZS",
        "Okta":                  "OKTA",
        "Varonis":               "VRNS",
        "Tenable":               "TENB",
        "Qualys":                "QLYS",
        "Check Point Software":  "CHKP",
        "Darktrace":             "DARK.L",
        "NCC Group":             "NCC.L",
        "CIBR ETF":              "CIBR",
        "BUG ETF":               "BUG",
    },

    # ── Gold / Precious Metals ────────────────────────────────────────────────
    "Gold / Precious Metals": {
        "Newmont":               "NEM",
        "Barrick Gold":          "GOLD",
        "Agnico Eagle":          "AEM",
        "Kinross Gold":          "KGC",
        "Gold Fields":           "GFI",
        "AngloGold Ashanti":     "AU",
        "Northern Star":         "NST.AX",
        "Evolution Mining":      "EVN.AX",
        "Endeavour Mining":      "EDV.L",
        "Harmony Gold":          "HMY",
        "Sibanye-Stillwater":    "SBSW",
        "VanEck Gold Miners":    "GDX",
        "VanEck Jr Gold Miners": "GDXJ",
        "SPDR Gold Shares":      "GLD",
        "iShares Gold (SGLN)":   "SGLN.L",
    },

    # ── Silver & Royalties ────────────────────────────────────────────────────
    "Silver & Royalties": {
        "Wheaton Precious Metals":"WPM",
        "Royal Gold":            "RGLD",
        "Franco-Nevada":         "FNV",
        "Pan American Silver":   "PAAS",
        "First Majestic Silver": "AG",
        "Silvercorp Metals":     "SVM",
        "MAG Silver":            "MAG.TO",
        "iShares Silver (SLV)":  "SLV",
    },

    # ── Copper & Battery Metals ───────────────────────────────────────────────
    "Copper & Battery Metals": {
        "Freeport-McMoRan":      "FCX",
        "Southern Copper":       "SCCO",
        "Glencore":              "GLEN.L",
        "Rio Tinto":             "RIO.L",
        "BHP Group":             "BHP.AX",
        "Anglo American":        "AAL.L",
        "Teck Resources":        "TECK",
        "Antofagasta":           "ANTO.L",
        "Lundin Mining":         "LUN.TO",
        "Ivanhoe Mines":         "IVN.TO",
        "Copper ETF (COPX)":     "COPX",
    },

    # ── Lithium ───────────────────────────────────────────────────────────────
    "Lithium": {
        "Albemarle":             "ALB",
        "SQM":                   "SQM",
        "Livent / Arcadium":     "ALTM",
        "Pilbara Minerals":      "PLS.AX",
        "Liontown Resources":    "LTR.AX",
        "Core Lithium":          "CXO.AX",
        "Sigma Lithium":         "SGML",
        "Zinnwald Lithium":      "ZNWD.L",
        "Neo Performance":       "NEO.TO",
        "LIT Lithium ETF":       "LIT",
        "BATT ETF":              "BATT",
    },

    # ── Quantum Computing ─────────────────────────────────────────────────────
    "Quantum Computing": {
        "IBM":                   "IBM",
        "IonQ":                  "IONQ",
        "Rigetti Computing":     "RGTI",
        "D-Wave Quantum":        "QBTS",
        "Quantum Computing Inc": "QUBT",
        "Microsoft":             "MSFT",
        "Alphabet":              "GOOGL",
        "Honeywell":             "HON",
    },

    # ── Semiconductors ────────────────────────────────────────────────────────
    "Semiconductors": {
        "NVIDIA":                "NVDA",
        "AMD":                   "AMD",
        "Intel":                 "INTC",
        "TSMC":                  "TSM",
        "ASML":                  "ASML.AS",
        "Applied Materials":     "AMAT",
        "Lam Research":          "LRCX",
        "KLA Corporation":       "KLAC",
        "Marvell Technology":    "MRVL",
        "Broadcom":              "AVGO",
        "Qualcomm":              "QCOM",
        "STMicroelectronics":    "STM.MI",
        "Infineon":              "IFX.DE",
        "NXP Semiconductors":    "NXPI",
        "ON Semiconductor":      "ON",
        "Wolfspeed":             "WOLF",
        "SOXX Semiconductor ETF":"SOXX",
        "SMH ETF":               "SMH",
    },

    # ── AI & Cloud ────────────────────────────────────────────────────────────
    "AI & Cloud": {
        "Salesforce":            "CRM",
        "ServiceNow":            "NOW",
        "Palantir":              "PLTR",
        "C3.ai":                 "AI",
        "UiPath":                "PATH",
        "Snowflake":             "SNOW",
        "Datadog":               "DDOG",
        "MongoDB":               "MDB",
        "Cloudflare":            "NET",
        "Workday":               "WDAY",
        "Veeva Systems":         "VEEV",
        "BOTZ AI & Robotics ETF":"BOTZ",
        "ROBO ETF":              "ROBO",
        "AIQ ETF":               "AIQ",
    },

    # ── Logistics / Shipping ──────────────────────────────────────────────────
    "Logistics / Shipping": {
        "Deutsche Post / DHL":   "DPW.DE",
        "AP Moller-Maersk":      "MAERSK-A.CO",
        "Hapag-Lloyd":           "HLAG.DE",
        "FedEx":                 "FDX",
        "UPS":                   "UPS",
        "XPO Logistics":         "XPO",
        "Kuehne+Nagel":          "KNIN.SW",
        "DSV":                   "DSV.CO",
        "Expeditors Intl":       "EXPD",
        "C.H. Robinson":         "CHRW",
        "ZIM Integrated":        "ZIM",
        "Star Bulk Carriers":    "SBLK",
        "Navios Maritime":       "NMM",
    },

    # ── Finance & Banking ─────────────────────────────────────────────────────
    "Finance & Banking": {
        "JPMorgan Chase":        "JPM",
        "BNP Paribas":           "BNP.PA",
        "Deutsche Bank":         "DBK.DE",
        "Commerzbank":           "CBK.DE",
        "ING Group":             "INGA.AS",
        "UniCredit":             "UCG.MI",
        "Santander":             "SAN.MC",
        "BBVA":                  "BBVA.MC",
        "ABN AMRO":              "ABN.AS",
        "Raiffeisen Bank":       "RBI.VI",
        "Erste Group":           "EBS.VI",
        "Goldman Sachs":         "GS",
        "Morgan Stanley":        "MS",
        "BlackRock":             "BLK",
        "KBW Bank ETF":          "KBE",
    },

    # ── Insurance ─────────────────────────────────────────────────────────────
    "Insurance": {
        "Allianz":               "ALV.DE",
        "Munich Re":             "MUV2.DE",
        "Hannover Re":           "HNR1.DE",
        "Swiss Re":              "SREN.SW",
        "Zurich Insurance":      "ZURN.SW",
        "AXA":                   "CS.PA",
        "Generali":              "G.MI",
        "Berkshire Hathaway B":  "BRK-B",
        "Chubb":                 "CB",
        "Markel":                "MKL",
    },

    # ── Healthcare / Pharma ───────────────────────────────────────────────────
    "Healthcare / Pharma": {
        "Novo Nordisk":          "NOVO-B.CO",
        "Roche":                 "ROG.SW",
        "Novartis":              "NOVN.SW",
        "AstraZeneca":           "AZN.L",
        "Sanofi":                "SAN.PA",
        "Bayer":                 "BAYN.DE",
        "Merck KGaA":            "MRK.DE",
        "Thermo Fisher":         "TMO",
        "Danaher":               "DHR",
        "Sartorius Vz":          "SRT3.DE",
        "Lonza":                 "LONN.SW",
        "Straumann":             "STMN.SW",
        "Genmab":                "GMAB",
        "UCB":                   "UCB.BR",
        "Ipsen":                 "IPN.PA",
        "iShares Healthcare ETF":"IHF",
    },

    # ── Water & Environment ───────────────────────────────────────────────────
    "Water & Environment": {
        "Veolia":                "VIE.PA",
        "Xylem":                 "XYL",
        "Pentair":               "PNR",
        "Watts Water":           "WTS",
        "Ecolab":                "ECL",
        "L&G Clean Water ETF":   "GLUG.L",
        "Invesco Water ETF":     "PHO",
    },

    # ── Agriculture / Chemicals ───────────────────────────────────────────────
    "Agriculture / Chemicals": {
        "Yara International":    "YAR.OL",
        "Nutrien":               "NTR.TO",
        "Mosaic":                "MOS",
        "CF Industries":         "CF",
        "ICL Group":             "ICL",
        "BASF":                  "BAS.DE",
        "Lanxess":               "LXS.DE",
        "Corteva":               "CTVA",
        "FMC Corporation":       "FMC",
        "Archer-Daniels-Midland":"ADM",
        "Bunge Global":          "BG",
        "Deere & Company":       "DE",
        "AGCO Corporation":      "AGCO",
        "CNH Industrial":        "CNHI.MI",
        "MOO Agriculture ETF":   "MOO",
    },

    # ── Real Estate (REIT) ────────────────────────────────────────────────────
    "Real Estate (REIT)": {
        "Vonovia":               "VNA.DE",
        "LEG Immobilien":        "LEG.DE",
        "TAG Immobilien":        "TEG.DE",
        "Unibail-Rodamco":       "URW.PA",
        "Klepierre":             "LI.PA",
        "Segro":                 "SGRO.L",
        "Land Securities":       "LAND.L",
        "British Land":          "BLND.L",
        "Realty Income":         "O",
        "Prologis":              "PLD",
        "Digital Realty":        "DLR",
        "Equinix":               "EQIX",
        "American Tower":        "AMT",
        "SBA Communications":    "SBAC",
        "VNQI Global ex-US REIT":"VNQI",
        "IPRP EU REIT ETF":      "IPRP.L",
    },

    # ── Broad / Macro ETFs ────────────────────────────────────────────────────
    "Broad ETFs": {
        "MSCI World (IWDA)":         "IWDA.AS",
        "MSCI World (SWRD)":         "SWRD.L",
        "MSCI EM IMI (EIMI)":        "EIMI.L",
        "S&P 500 (CSPX)":            "CSPX.L",
        "S&P 500 (VOO)":             "VOO",
        "EURO STOXX 50 (EXW1)":      "EXW1.DE",
        "DAX (EXIA)":                "EXIA.DE",
        "FTSE 100 (ISF)":            "ISF.L",
        "Short S&P 500 (SH)":        "SH",
        "2x Short S&P (SDS)":        "SDS",
        "Commodities (PDBC)":        "PDBC",
        "Treasury Bonds 20yr (TLT)": "TLT",
        "EUR Govt Bonds (IEGA)":     "IEGA.L",
        "TIPS Inflation (ITPS)":     "ITPS.L",
        "MSCI Momentum (IWMO)":      "IWMO.AS",
        "MSCI Min Vol (IWVL)":       "IWVL.AS",
        "MSCI Quality (IWQU)":       "IWQU.AS",
        "Samsung":                   "005930.KS",
    },
}


# ─── Public API ───────────────────────────────────────────────────────────────

def get_market_universe() -> dict[str, str]:
    """
    Return flat {name: symbol} dict across all sectors.
    Deduplicates on symbol — first sector occurrence wins.
    """
    flat: dict[str, str] = {}
    seen: set[str] = set()
    for assets in SECTOR_UNIVERSE.values():
        for name, symbol in assets.items():
            if symbol not in seen:
                flat[name] = symbol
                seen.add(symbol)
    return flat


def symbol_to_sector(symbol: str) -> str:
    """Reverse-lookup sector from yf symbol. Returns first sector match."""
    for sector, assets in SECTOR_UNIVERSE.items():
        if symbol in assets.values():
            return sector
    return "Unknown"


def get_sector_symbols(sector: str) -> dict[str, str]:
    """Return {name: symbol} for a single named sector."""
    return SECTOR_UNIVERSE.get(sector, {})


def universe_stats() -> None:
    """Print a sector-by-sector count summary."""
    total = 0
    for sector, assets in SECTOR_UNIVERSE.items():
        n = len(assets)
        total += n
        print(f"  {sector:<32}  {n:>3} instruments")
    flat = get_market_universe()
    print(f"\n  {'TOTAL (deduplicated)':<32}  {len(flat):>3} unique symbols")
    print(f"  {'TOTAL (with dupes)':<32}  {total:>3} entries")


# ─── Geo Risk Tables ──────────────────────────────────────────────────────────

GEO_BASE: dict[str, int] = {
    "United States":              2,
    "Germany":                    2,
    "France":                     2,
    "United Kingdom":             2,
    "Netherlands":                2,
    "Switzerland":                2,
    "Norway":                     2,
    "Denmark":                    2,
    "Sweden":                     2,
    "Finland":                    2,
    "Canada":                     2,
    "Australia":                  3,
    "New Zealand":                2,
    "Japan":                      3,
    "South Korea":                5,
    "Taiwan":                     6,
    "Hong Kong":                  5,
    "Singapore":                  3,
    "China":                      7,
    "India":                      4,
    "Brazil":                     5,
    "Mexico":                     5,
    "South Africa":               5,
    "Russia":                     10,
    "Ukraine":                    10,
    "Israel":                     7,
    "Iran":                       10,
    "Turkey":                     6,
    "Austria":                    3,
    "Belgium":                    2,
    "Spain":                      3,
    "Italy":                      3,
    "Portugal":                   3,
    "Greece":                     4,
    "Poland":                     3,
    "Hungary":                    4,
    "Czech Republic":             3,
    "Romania":                    4,
    "Chile":                      4,
    "Colombia":                   5,
    "Peru":                       5,
    "Kazakhstan":                 6,
    "Zambia":                     6,
    "Democratic Republic Congo":  7,
}

GEO_KEYWORDS: list[str] = [
    "sanction",    "war",          "conflict",    "invasion",    "default",
    "recession",   "fraud",        "investigation","ban",         "tariff",
    "embargo",     "collapse",     "crisis",       "downgrade",   "lawsuit",
    "bankruptcy",  "seizure",      "nationalise",  "nationalize", "fine",
    "probe",       "scandal",      "corruption",   "coup",        "protest",
    "strike",      "cyberattack",  "hack",         "data breach", "penalty",
]

# ─── Currency Display Symbols ─────────────────────────────────────────────────

CURRENCY_SYMBOLS: dict[str, str] = {
    "USD": "$",    "EUR": "€",    "GBP": "£",    "GBX": "p",
    "NOK": "kr",   "DKK": "kr",   "SEK": "kr",   "AUD": "A$",
    "CAD": "C$",   "JPY": "¥",    "KRW": "₩",    "CHF": "Fr",
    "HKD": "HK$",  "SGD": "S$",   "NZD": "NZ$",  "BRL": "R$",
    "ZAR": "R",    "MXN": "$",    "PLN": "zl",   "CZK": "Kc",
}


if __name__ == "__main__":
    universe_stats()
