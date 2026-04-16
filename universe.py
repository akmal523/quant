import pandas as pd

def get_market_universe() -> dict:
    """
    Расширенная секторная матрица. 
    Тикеры адаптированы под требования Yahoo Finance API.
    """
    return {
        # --- URANIUM (Уран: Энергия будущего) ---
        "Cameco": "CCJ",
        "NexGen Energy": "NXE",
        "Uranium Energy Corp": "UEC",
        "Global X Uranium ETF": "URA",
        "Sprott Physical Uranium": "U-U.TO", # Исправлено: Канадский формат

        # --- GOLD (Золото: Защитный актив) ---
        "Barrick Gold": "GOLD",
        "Newmont": "NEM",
        "Franco-Nevada": "FNV",
        "VanEck Gold Miners": "GDX",

        # --- QUANTUM & AI (Квантовый скачок) ---
        "IonQ": "IONQ",
        "Rigetti Computing": "RGTI",
        "D-Wave Quantum": "QBTS",
        "NVIDIA": "NVDA",
        "Alphabet": "GOOGL",

        # --- CYBERSECURITY (Защита данных) ---
        "Palo Alto": "PANW",
        "CrowdStrike": "CRWD",
        "Fortinet": "FTNT",
        "Zscaler": "ZS",

        # --- DEFENSE (Оборонный сектор) ---
        "Rheinmetall": "RHM.DE",
        "Lockheed Martin": "LMT",
        "BAE Systems": "BA.L",
        "General Dynamics": "GD",

        # --- LOGISTICS & STABILITY (Инфраструктура) ---
        "Maersk": "MAERSK-A.CO",
        "DHL Group": "DHL.DE",
        "Ecolab": "ECL",
        "Heidelberg Materials": "HEI.DE",
        "Equinor": "EQNR"
    }
