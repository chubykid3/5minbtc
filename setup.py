"""
Cross-platform setup script.
Installs all dependencies and verifies the environment.

Usage:
  python setup.py
"""

import subprocess
import sys
import os
from pathlib import Path


def run(cmd, check=True):
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=check, capture_output=False)
    return result.returncode == 0


def main():
    print("=" * 60)
    print("  BTC Decision Bot — Environment Setup")
    print("=" * 60)
    print()

    py = sys.executable

    # Upgrade pip
    print("[1/4] Upgrading pip...")
    run([py, "-m", "pip", "install", "--upgrade", "pip"], check=False)

    # Try TensorFlow — optional
    print("\n[2/4] Installing core dependencies (no TensorFlow)...")
    core_pkgs = [
        "websockets>=12.0",
        "aiohttp>=3.9.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.4.0",
        "xgboost>=2.0.0",
        "pandas>=2.1.0",
        "web3>=6.15.0",
        "rich>=13.7.0",
        "python-dateutil>=2.8.0",
    ]
    run([py, "-m", "pip", "install"] + core_pkgs)

    print("\n[3/4] Attempting TensorFlow install (optional — LSTM model)...")
    tf_ok = run([py, "-m", "pip", "install", "tensorflow>=2.15.0"], check=False)
    if not tf_ok:
        print("  ⚠ TensorFlow install failed. LSTM model will be disabled.")
        print("  The bot works without it (XGBoost takes LSTM weight).")

    print("\n[4/4] Creating directories...")
    for d in ["data", "logs", "models/saved"]:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  created: {d}/")

    print()
    print("=" * 60)
    print("  Setup complete!")
    print()
    print("  Quick start:")
    print("    python main.py --train-only    # fetch data + train (30 min)")
    print("    python main.py                 # run the live bot")
    print("    python main.py --no-train      # skip training, use saved models")
    print("    python main.py --backtest      # backtest on saved data")
    print()
    print("  All decisions logged to:")
    print("    logs/decisions.log   — human-readable")
    print("    logs/metrics.jsonl   — structured JSON lines")
    print("    data/bot.db          — SQLite database")
    print("=" * 60)


if __name__ == "__main__":
    main()
