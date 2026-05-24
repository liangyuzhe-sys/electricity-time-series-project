"""
Run the three modelling scripts in this project.

Default order:
    1. sarima.py
    2. var.py
    3. garch.py

Examples:
    python code/main.py
    python code/main.py var garch
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
SCRIPTS = {
    "sarima": CODE_DIR / "sarima.py",
    "var": CODE_DIR / "var.py",
    "garch": CODE_DIR / "garch.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run project model scripts.")
    parser.add_argument(
        "models",
        nargs="*",
        choices=tuple(SCRIPTS),
        help="Models to run. Default: sarima var garch.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    models = args.models or ["sarima", "var", "garch"]
    for model in models:
        script = SCRIPTS[model]
        print(f"[main] running {script.name}", flush=True)
        subprocess.run([sys.executable, str(script)], check=True)
    print("[main] all requested models completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
