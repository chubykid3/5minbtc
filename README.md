# BTC 5-Minute Polymarket Decision Bot

A fully autonomous decision bot targeting **Polymarket BTC/USD 5-minute markets**.
Every window it decides UP or DOWN at the 2:30 mark (T=150s), using a 4-model ensemble
trained on real Binance OHLCV data and live market signals.

---

## Architecture

```
INPUT LAYER
├── Binance BTC/USDT WebSocket — tick data + order book (real-time)
├── Chainlink BTC/USD Oracle   — on-chain price (polled every 5s via web3)
├── Polymarket CLOB REST API   — implied UP/DOWN probabilities (polled every 3s)
└── Derived: CVD, funding rate, session flags, candle indicators

FEATURE ENGINEERING (35 features at T=150s)
├── Price displacement from T=0 reference
├── Price momentum/velocity (30s, 60s slopes from tick data)
├── Technical indicators on 1-min candles (RSI-14, EMA-9/21, MACD, BB, ATR, OBV)
├── Order book (bid/ask imbalance, wall distances, spread)
├── Oracle lag vs spot (Chainlink staleness arbitrage signal)
├── Polymarket crowd probability + 30s delta
└── Session (hour/day cyclical encoding, US/Asia/EU flags, streak, recent win rate)

MODEL ENSEMBLE
├── A: Logistic Regression   (weight 0.15) — linear baseline
├── B: XGBoost               (weight 0.40) — primary non-linear model
├── C: LSTM                  (weight 0.25) — sequential patterns across windows
└── D: Bayesian/Heuristic    (weight 0.20) — mean-reversion + oracle lag rules

DECISION RULE
├── |P(UP) - 0.5| > 0.04 → follow model
├── Near 50/50 → follow Polymarket crowd
└── Still tied → default UP (BTC drift bias + Chainlink tie goes to UP)

OUTPUT
├── Terminal dashboard (rich)
├── logs/decisions.log   — human-readable decision log
├── logs/metrics.jsonl   — structured JSONL for analysis
└── data/bot.db          — SQLite (candles, decisions, outcomes, accuracy)
```

---

## Requirements

- Python 3.10+
- Internet access (Binance WebSocket, Ethereum RPC for Chainlink)
- Works on **Windows** (CMD/PowerShell) and **Ubuntu/Linux** terminal

---

## Quick Start

### 1. Install dependencies

```bash
# Recommended: create a virtual environment first
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Run the setup script
python setup.py

# Or install manually:
pip install -r requirements.txt
```

> **TensorFlow is optional.** If it fails to install, the bot runs without the LSTM
> model — XGBoost takes its weight automatically.

### 2. Fetch historical data + train

```bash
python main.py --train-only
```

Downloads 30 days of Binance 1-minute OHLCV data and trains all models.
Takes ~5–15 minutes depending on your connection.

### 3. Run the live bot

```bash
python main.py
```

The bot will:
- Connect to Binance WebSocket, Chainlink oracle, and Polymarket API
- Align to the next 5-minute UTC window boundary
- Every window: log the decision at T=150s, log the outcome at T=300s
- Display a live terminal dashboard
- Retrain models every 24 hours automatically

### 4. Other modes

```bash
python main.py --no-train      # Use saved models, skip initial training
python main.py --backtest      # Run backtest on last 500 windows
python main.py --log-level DEBUG  # Verbose logging
```

---

## File Structure

```
.
├── main.py                  — Entry point + window manager
├── config.py                — All tunable parameters
├── setup.py                 — Dependency installer
├── candle_store.py          — Live 1-min OHLCV candle builder from ticks
├── decision_logger.py       — Logging + rich terminal dashboard
│
├── data_feeds/
│   ├── binance_feed.py      — Binance trades + order book WebSocket
│   ├── chainlink_feed.py    — Chainlink BTC/USD oracle via web3
│   └── polymarket_feed.py   — Polymarket implied probability REST poller
│
├── features/
│   ├── feature_engineer.py  — 35-feature vector computation at T=150s
│   └── technical_indicators.py — RSI, EMA, MACD, Bollinger, ATR, OBV
│
├── models/
│   ├── logistic_model.py    — Logistic Regression (sklearn)
│   ├── xgboost_model.py     — XGBoost classifier
│   ├── lstm_model.py        — LSTM (TensorFlow/Keras, optional)
│   ├── bayesian_model.py    — Heuristic mean-reversion + oracle lag model
│   ├── ensemble.py          — Weighted ensemble + meta-learner + decision rule
│   └── saved/               — Serialized trained model files
│
├── training/
│   ├── data_fetcher.py      — Binance historical OHLCV + SQLite storage
│   └── trainer.py           — Training pipeline + walk-forward backtest
│
├── data/
│   └── bot.db               — SQLite database (candles, decisions, outcomes)
└── logs/
    ├── decisions.log        — Human-readable log
    └── metrics.jsonl        — Structured JSON lines for analysis
```

---

## Key Config Parameters (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EDGE_THRESHOLD` | 0.04 | Min \|P-0.5\| to make a confident call |
| `HIGH_CONF_THRESH` | 0.08 | \|P-0.5\| > this = high-confidence call |
| `ENSEMBLE_WEIGHTS` | LR=0.15, XGB=0.40, LSTM=0.25, BAY=0.20 | Initial model weights |
| `MEAN_REVERSION_THRESHOLD` | 0.003 (0.3%) | Beyond this, lean against the move |
| `RETRAIN_EVERY_HOURS` | 24 | Auto-retrain frequency |
| `RETRAIN_LOOKBACK_DAYS` | 30 | Training data window size |
| `ETH_RPC_URL` (env var) | Public endpoints | Ethereum RPC for Chainlink |

---

## Decision Logic

```
At T=150s (2:30 remaining in each 5-min window):

1. Compute 35-feature vector from all live data feeds
2. Run all 4 models → get P(UP) per model
3. Weighted ensemble → final P(UP)

Decision rule:
  P(UP) > 0.54  → UP   (confident bet)
  P(UP) < 0.46  → DOWN (confident bet)
  0.46 ≤ P ≤ 0.54 → near 50/50:
    → Follow Polymarket crowd if they have conviction
    → In high-volatility: follow current window momentum
    → Default: UP (Chainlink tie goes UP, BTC has positive drift)
```

---

## Accuracy Target

| Scenario | Expected Accuracy |
|----------|------------------|
| Random walk | ~50% |
| BTC positive drift only | ~51–52% |
| Academic ML on 5-min BTC | 55–67% (varies widely) |
| **This bot, realistic live** | **54–57%** |
| High-confidence calls only | Target 58–62% |

At 55% accuracy over 288 windows/day: ~158 correct vs ~130 wrong. Small but real edge.

---

## Logs

**`logs/decisions.log`**:
```
14:30:00  DECISION | Window 14:30:00 UTC | Side=UP | P(UP)=0.581 | Conf=0.081 | ...
14:35:01  OUTCOME  | Window 14:30:00 UTC | Result=UP | Δ=+0.0234% | ✓ CORRECT
```

**Query the SQLite DB directly**:
```sql
-- Overall accuracy
SELECT COUNT(*), AVG(correct) FROM decisions WHERE resolved_up IS NOT NULL;

-- High-confidence accuracy
SELECT COUNT(*), AVG(correct) FROM decisions
WHERE confidence > 0.08 AND resolved_up IS NOT NULL;

-- Accuracy by session hour
SELECT CAST(strftime('%H', datetime(window_start, 'unixepoch')) AS INT) AS hour,
       COUNT(*), AVG(correct)
FROM decisions GROUP BY hour ORDER BY hour;
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `ETH_RPC_URL` | Ethereum RPC for Chainlink oracle (optional — public nodes used by default) |

```bash
export ETH_RPC_URL="https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY"
python main.py
```

---

## Notes

- **No trading keys required** — bot reads public market data only. It predicts, it does not trade.
- **Chainlink oracle**: defaults to free public Ethereum RPC endpoints. Set `ETH_RPC_URL` for reliability.
- **Models retrain automatically** every 24 hours. First training ~5–15 min.
- **Windows align to UTC** 5-minute boundaries (00:00, 00:05, 00:10, ...).
- **Polymarket market discovery**: the bot auto-finds the active BTC 5-min market via the CLOB API. If the market is between windows, implied prob defaults to 0.50.