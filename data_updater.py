"""
Entry point: `python3 data_updater.py`.

Thin wrapper so the documented command keeps working after the package move.
Runs the market-data update + funnel in `quant.data.data_updater`.
"""
from quant.data.data_updater import main

if __name__ == "__main__":
    main()
