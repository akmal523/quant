"""
Entry point: `python3 main.py`.

Thin wrapper so the documented command keeps working after the package move.
Runs the full analysis pipeline defined in `quant.main`.
"""
from quant.main import main

if __name__ == "__main__":
    main()
