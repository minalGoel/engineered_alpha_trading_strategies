"""Pipeline configuration — paths, constants, defaults.

Updated for 5-second index option trading on NIFTY/BANKNIFTY.
"""
from pathlib import Path
import os
import math

# ── Root paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = ROOT / "trading_strategies" / "unique_strategies"
DATA_DIR = ROOT / "data"
FIVE_SEC_DIR = ROOT / "5second_data"
RESULTS_DIR = ROOT / "strategy_results"
OUTPUTS_DIR = ROOT / "outputs"

# ── 5-second data files ─────────────────────────────────────────────────────
SPOT_FILE = FIVE_SEC_DIR / "spot_candles.parquet"
OPTION_FILE = FIVE_SEC_DIR / "option_candles.parquet"

# Symbols in spot data
NIFTY_SYMBOL = "NSE:NIFTY50-INDEX"
BANKNIFTY_SYMBOL = "NSE:NIFTYBANK-INDEX"
VIX_SYMBOL = "NSE:INDIAVIX-INDEX"

# ── Time constants (IST) ───────────────────────────────────────────────────
MARKET_OPEN_H, MARKET_OPEN_M = 9, 15     # 09:15 IST
MARKET_CLOSE_H, MARKET_CLOSE_M = 15, 30  # 15:30 IST
EOD_FLATTEN_H, EOD_FLATTEN_M = 15, 25    # 15:25 IST — flatten all positions
EXPIRY_FLATTEN_H, EXPIRY_FLATTEN_M = 15, 20  # 15:20 IST on expiry days
DEFAULT_SESSION_START = (9, 20)           # 09:20 IST (skip first 5 min)
DEFAULT_SESSION_END = (15, 25)            # 15:25 IST

# ── Leave-one-day-out CV ──────────────────────────────────────────────────
# 12 days of data — use leave-one-day-out cross-validation
TOTAL_TRADING_DAYS = 12
MIN_PROFITABLE_DAYS = 9  # must be profitable on >= 9 of 12 test days

# ── Backtesting defaults ───────────────────────────────────────────────────
DEFAULT_MAX_HOLD_BARS = 24        # 24 × 5s = 120 seconds max hold
DEFAULT_STOP_POINTS = 5.0         # default stop in option premium points
MIN_TRADES_FULL = 50              # minimum trades on full training period
MIN_TRADES_PER_UNDERLYING = 30    # Chan's minimum for statistical validity

# ── Lot sizes ────────────────────────────────────────────────────────────
NIFTY_LOT = 65
BANKNIFTY_LOT = 30

# ── Cost model ───────────────────────────────────────────────────────────
STT_RATE = 0.0015             # 0.15% on sell-side premium (from 1 Apr 2025)
BROKERAGE_PER_ORDER = 20      # INR flat per order
EXCHANGE_TXN_RATE = 0.0003553 # 0.03553% on premium turnover (both sides, NSE Mar 2026)
CAPITAL_PER_ENTRY = 500_000   # ₹5 lakh per strategy entry order
MIN_POINTS_VIABLE = 1.5       # minimum premium move per trade to be viable

# ── Optimization ────────────────────────────────────────────────────────
OPTUNA_TRIALS = 100
OPTUNA_TRIALS_PER_FOLD = 50     # Reduced per-fold in nested CV (12 folds × 50 = 600 total)
OPTUNA_TIMEOUT_SECS = 30 * 60   # 30-minute safety timeout per strategy
OPTUNA_FOLD_TIMEOUT_SECS = 120  # 2-min timeout per CV fold optimization
OVERFIT_SHARPE_RATIO = 3.0      # flag if optimized/default > 3x
SUSPICIOUS_SHARPE = 3.0         # Sharpe > 3.0 is suspicious

# ── Parallelism defaults ───────────────────────────────────────────────
DEFAULT_PARALLEL_STRATEGIES = 8
DEFAULT_CORES_PER_STRATEGY = 4

# ── Signal actions (option trading) ──────────────────────────────────────
ACTION_BUY_CE = "BUY_CE"
ACTION_SELL_CE = "SELL_CE"
ACTION_BUY_PE = "BUY_PE"
ACTION_SELL_PE = "SELL_PE"
ACTION_FLAT = "FLAT"

# ── Position states (Numba int constants) ───────────────────────────────
STATE_FLAT = 0
STATE_LONG_CE = 1
STATE_LONG_PE = 3

# ── Exit reasons ────────────────────────────────────────────────────────
EXIT_STOP = "STOP"
EXIT_TARGET = "TARGET"
EXIT_SIGNAL = "SIGNAL"
EXIT_TIME = "TIME"
EXIT_EOD = "EOD"

# ── Data-snooping correction ───────────────────────────────────────────
BASE_SHARPE_THRESHOLD = 0.5

def required_sharpe(n_strategies: int) -> float:
    """Minimum Sharpe for significance given N independent tests (Chan's rule)."""
    if n_strategies <= 1:
        return BASE_SHARPE_THRESHOLD
    return BASE_SHARPE_THRESHOLD * math.sqrt(1 + math.log(n_strategies))

# ── Logging format ──────────────────────────────────────────────────────
LOG_FORMAT = "[{idx}/{total}] {name} — Phase {phase} — {msg}"
