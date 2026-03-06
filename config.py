"""
BTC 5-Minute Polymarket Decision Bot — Configuration
All tunable parameters and API endpoints in one place.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR  = BASE_DIR / "logs"
MODEL_DIR = BASE_DIR / "models" / "saved"

DATA_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ─── Decision Logic ───────────────────────────────────────────────────────────
WINDOW_SECONDS   = 300          # 5-minute windows
DECISION_OFFSET  = 150          # Decide at T=150s (2:30 remaining)
EDGE_THRESHOLD   = 0.04         # Need P(UP) > 0.54 or < 0.46 for confident bet
HIGH_CONF_THRESH = 0.08         # |P - 0.5| > 0.08 = high-confidence call

# Default weights for ensemble (tuned via stacking later)
ENSEMBLE_WEIGHTS = {
    "logistic":  0.15,
    "xgboost":   0.40,
    "lstm":      0.25,
    "bayesian":  0.20,
}

# Mean-reversion thresholds for Bayesian model
MEAN_REVERSION_THRESHOLD = 0.0030   # 0.30% — beyond this, lean against direction
MOMENTUM_THRESHOLD       = 0.0008   # 0.08% — below this, follow momentum

# ─── Training Schedule ────────────────────────────────────────────────────────
RETRAIN_EVERY_HOURS = 24            # Retrain XGBoost/LogReg every 24h
RETRAIN_LOOKBACK_DAYS = 30          # Use last 30 days of data
LSTM_RETRAIN_DAYS = 7               # Retrain LSTM weekly
MIN_TRAINING_SAMPLES = 500          # Don't train until we have this many rows
LSTM_SEQUENCE_LENGTH = 20           # 20 prior windows as LSTM input

# ─── Binance WebSocket ────────────────────────────────────────────────────────
BINANCE_WS_BASE     = "wss://stream.binance.com:9443/stream"
BINANCE_REST_BASE   = "https://api.binance.com"
BINANCE_SYMBOL      = "BTCUSDT"
BINANCE_SYMBOL_LOWER = "btcusdt"
ORDERBOOK_DEPTH     = 10            # Top 10 levels

# ─── Chainlink Oracle ─────────────────────────────────────────────────────────
# Chainlink BTC/USD Aggregator on Ethereum Mainnet
CHAINLINK_BTC_USD_ADDRESS = "0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88"
CHAINLINK_ABI = [
    {
        "inputs": [],
        "name": "latestRoundData",
        "outputs": [
            {"internalType": "uint80",  "name": "roundId",         "type": "uint80"},
            {"internalType": "int256",  "name": "answer",          "type": "int256"},
            {"internalType": "uint256", "name": "startedAt",       "type": "uint256"},
            {"internalType": "uint256", "name": "updatedAt",       "type": "uint256"},
            {"internalType": "uint80",  "name": "answeredInRound", "type": "uint80"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"internalType": "uint8", "name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
]
# Public Ethereum RPC (no key required; fallback chain used if Eth unavailable)
ETH_RPC_URLS = [
    os.getenv("ETH_RPC_URL", "https://eth.llamarpc.com"),
    "https://rpc.ankr.com/eth",
    "https://ethereum.publicnode.com",
]
CHAINLINK_POLL_INTERVAL = 5         # Seconds between oracle polls

# ─── Polymarket ───────────────────────────────────────────────────────────────
POLYMARKET_REST_BASE  = "https://clob.polymarket.com"
POLYMARKET_WS_BASE    = "wss://ws-subscriptions-clob.polymarket.com/ws/"
# BTC 5-min market slug pattern — used for REST lookups
POLYMARKET_BTC_MARKET_TAG = "btc-usd-5-minutes"
POLYMARKET_POLL_INTERVAL  = 3       # Seconds between REST probability polls

# ─── Feature Engineering ──────────────────────────────────────────────────────
RSI_PERIOD           = 14
EMA_SHORT            = 9
EMA_LONG             = 21
MACD_FAST            = 12
MACD_SLOW            = 26
MACD_SIGNAL          = 9
BB_PERIOD            = 20
BB_STD               = 2
ATR_PERIOD           = 14
OBV_LOOKBACK         = 20           # candles
BID_ASK_LEVELS       = 5           # levels for imbalance calc
WALL_THRESHOLD_RATIO = 3.0         # 3x avg size = wall

# ─── Session Definitions (UTC hours) ─────────────────────────────────────────
ASIA_SESSION_START   = 0
ASIA_SESSION_END     = 8
EU_SESSION_START     = 7
EU_SESSION_END       = 16
US_SESSION_START     = 13
US_SESSION_END       = 22

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_FILE         = LOG_DIR / "decisions.log"
METRICS_FILE     = LOG_DIR / "metrics.jsonl"
DB_FILE          = DATA_DIR / "bot.db"
DISPLAY_REFRESH  = 1.0              # Console refresh rate (seconds)

# ─── Misc ─────────────────────────────────────────────────────────────────────
RECENT_WINDOWS_LOOKBACK = 20        # For recent win-rate feature
MAX_TICK_BUFFER  = 5000             # Max ticks to keep in memory per window
FUNDING_RATE_SYMBOL = "BTCUSDT"
LIQUIDATION_FEED_URL = "wss://fstream.binance.com/stream?streams=btcusdt@forceOrder"
