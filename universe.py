import pandas as pd

def get_market_universe() -> dict:
    return {
        # --- URANIUM (Уран) ---
        "Cameco": "CCJ",
        "NexGen Energy": "NXE",
        "Uranium Energy": "UEC",
        "Sprott Uranium": "SRUUF", # Замена на OTC-тикер (более стабильный для API)
        
        # --- GOLD ---
        "Barrick Gold": "GOLD",
        "Newmont": "NEM",
        "Franco-Nevada": "FNV",

        # --- TECHNOLOGY & QUANTUM ---
        "NVIDIA": "NVDA",
        "IonQ": "IONQ",
        "Alphabet": "GOOGL",
        "Palo Alto": "PANW",
        "CrowdStrike": "CRWD",

        # --- DEFENSE ---
        "Rheinmetall": "RHM.DE",
        "Lockheed Martin": "LMT",
        "BAE Systems": "BA.L",

        # --- INDUSTRY & LOGISTICS ---
        "Maersk": "MAERSK-A.CO",
        "DHL Group": "DHL.DE",
        "Ecolab": "ECL",
        "Heidelberg": "HEI.DE",
        "Equinor": "EQNR"
    }
