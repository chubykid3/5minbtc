#!/bin/bash
# BTC 5-Min Polymarket Bot — launcher
# Activates venv if available, then runs the bot.

cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

python3 main.py "$@"
