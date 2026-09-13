"""
Entry point: `python3 data_updater.py` (compatibility shim).

Prefer the console command `quant update`. Kept so the documented command keeps
working after the package/CLI move.
"""
from quant.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["update"]))
