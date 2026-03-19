"""Pipeline configuration — paths, constants, defaults."""
from pathlib import Path
import os

# ── Root paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = ROOT / "trading_strategies" / "unique_strategies"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "strategy_results"
OUTPUTS_DIR = ROOT / "outputs"

# Strategy source subdirectories
STRATEGY_SOURCES = [
    "claude_project", "cursor_gemini31pro", "cursor_gpt54",
    "cursor_opus46max", "gemini", "gpt", "grok", "opus",
]

# ── Data files ──────────────────────────────────────────────────────────────
TIMEFRAME_FILES = {
    "1min": DATA_DIR / "stocks_1min.parquet",
    "5min": DATA_DIR / "stocks_5min.parquet",
    "15min": DATA_DIR / "stocks_15min.parquet",
    "30min": DATA_DIR / "stocks_30min.parquet",
    "1h": DATA_DIR / "stocks_1h.parquet",
    "daily": DATA_DIR / "stocks_daily.parquet",
}

VIX_FILES = {
    "1min": DATA_DIR / "india_vix_1min.parquet",
    "5min": DATA_DIR / "india_vix_5min.parquet",
    "15min": DATA_DIR / "india_vix_15min.parquet",
    "30min": DATA_DIR / "india_vix_30min.parquet",
    "1h": DATA_DIR / "india_vix_1h.parquet",
    "daily": DATA_DIR / "india_vix_daily.parquet",
}

INDEX_FILES = {
    "1min": DATA_DIR / "index_1min.parquet",
    "5min": DATA_DIR / "index_5min.parquet",
    "15min": DATA_DIR / "index_15min.parquet",
    "30min": DATA_DIR / "index_30min.parquet",
    "1h": DATA_DIR / "index_1h.parquet",
    "daily": DATA_DIR / "index_daily.parquet",
}

STOCKS_LIST_FILE = DATA_DIR / "stocks_list.parquet"

# ── Time constants (IST) ───────────────────────────────────────────────────
MARKET_OPEN_H, MARKET_OPEN_M = 9, 15     # 09:15 IST
MARKET_CLOSE_H, MARKET_CLOSE_M = 15, 30  # 15:30 IST
EOD_FLATTEN_H, EOD_FLATTEN_M = 15, 20    # 15:20 IST — flatten all positions
DEFAULT_SESSION_START = (9, 15)
DEFAULT_SESSION_END = (15, 20)

# ── Train / Test split ─────────────────────────────────────────────────────
TRAIN_END = "2024-12-31"      # inclusive
TEST_START = "2025-01-01"     # inclusive
# CV fold boundaries (H1 = Jan-Jun, H2 = Jul-Dec)
CV_FOLDS = [
    ("2022-07-01", "2022-12-31"),  # Fold 1
    ("2023-01-01", "2023-06-30"),  # Fold 2
    ("2023-07-01", "2023-12-31"),  # Fold 3
    ("2024-01-01", "2024-06-30"),  # Fold 4
    ("2024-07-01", "2024-12-31"),  # Fold 5
]

# ── Backtesting defaults ───────────────────────────────────────────────────
DEFAULT_CAPITAL_PER_TRADE = 100_000
DEFAULT_MAX_HOLD_BARS = 120
DEFAULT_STOP_LOSS_PCT = 0.5     # 0.5% if not specified
MIN_TRADES_FULL = 50            # minimum trades on full training period
MIN_TRADES_PER_STOCK = 30       # Chan's minimum for statistical validity
MIN_CV_FOLDS_PROFITABLE = 4     # out of 5 folds

# ── Optimization ────────────────────────────────────────────────────────────
OPTUNA_TRIALS = 100
OPTUNA_TIMEOUT_SECS = 30 * 60   # 30-minute safety timeout per strategy
OVERFIT_SHARPE_RATIO = 3.0      # flag if optimized/default > 3x
SUSPICIOUS_SHARPE = 3.0         # Sharpe > 3.0 is suspicious

# ── Stock filter thresholds (Phase 5) ──────────────────────────────────────
FILTER_MIN_TRADES = 30
FILTER_MIN_PROFIT_FACTOR = 1.2
FILTER_MIN_SHARPE = 0.5
FILTER_MAX_DD_TO_PNL = 0.5     # max drawdown < 50% of total PnL
MIN_PASSING_STOCKS = 10

# ── OOS validation thresholds (Phase 6) ────────────────────────────────────
OOS_MIN_PROFIT_FACTOR = 1.0
OOS_MIN_SHARPE = 0.0
OOS_SHARPE_DECAY_LIMIT = 0.30  # test Sharpe >= 30% of train Sharpe
OOS_MIN_STOCKS_PROFITABLE_PCT = 0.50

# ── Parallelism defaults ───────────────────────────────────────────────────
# 8 workers × ~8GB peak per worker = 64GB, safely within 128GB on c7a.16xlarge.
# 16 workers would exceed 128GB during indicator computation (16 × 8GB = 128GB
# with no headroom for OS/JIT/Optuna overhead).
DEFAULT_PARALLEL_STRATEGIES = 8
DEFAULT_CORES_PER_STRATEGY = 4

# ── Signal actions ──────────────────────────────────────────────────────────
ACTION_BUY = "BUY"
ACTION_SELL = "SELL"
ACTION_SHORT = "SHORT"
ACTION_COVER = "COVER"
ACTION_FLAT = "FLAT"

# ── Position states (Numba int constants) ───────────────────────────────────
STATE_FLAT = 0
STATE_LONG = 1
STATE_SHORT = 2

# ── Exit reasons ────────────────────────────────────────────────────────────
EXIT_STOP = "STOP"
EXIT_TARGET = "TARGET"
EXIT_TRAILING = "TRAILING"
EXIT_SIGNAL = "SIGNAL"
EXIT_TIME = "TIME"
EXIT_EOD = "EOD"

# ── Data-snooping correction ───────────────────────────────────────────────
BASE_SHARPE_THRESHOLD = 0.5

def required_sharpe(n_strategies: int) -> float:
    """Minimum Sharpe for significance given N independent tests (Chan's rule)."""
    import math
    if n_strategies <= 1:
        return BASE_SHARPE_THRESHOLD
    return BASE_SHARPE_THRESHOLD * math.sqrt(1 + math.log(n_strategies))

# ── Logging format ──────────────────────────────────────────────────────────
LOG_FORMAT = "[{idx}/{total}] {name} — Phase {phase} — {msg}"
