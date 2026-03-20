"""narrow_range_breakout_v1 — Intraday micro-compression breakout on NIFTY.

Thesis:
    When NIFTY's 60-second high-low range compresses below 60% of the prior 10-minute
    average range, it signals a momentary equilibrium (algorithmic market-makers balanced).
    A close that breaks above/below this compressed zone triggers cascading algo entries
    (stat-arb, CTA, delta-hedgers) that extend the move 30-90 seconds.

Adapted from:
    Original NR4/NR7 equity strategy (Strategy_176.json) — daily range compression on
    NIFTY200 stocks. Compression window scaled from multi-day daily ranges to intraday
    60-second micro-ranges, proportional to 30-90s hold vs original 30-90min hold.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.max(arr[i - window + 1 : i + 1])
    return result


def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.min(arr[i - window + 1 : i + 1])
    return result


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.mean(arr[i - window + 1 : i + 1])
    return result


def _compute_vwap(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Session VWAP, reset per day_id."""
    n = len(close)
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_v = 0.0
    prev_day = -999999
    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_v = 0.0
            prev_day = day_id[i]
        v = volume[i] if volume[i] > 0 else 1.0
        cum_pv += close[i] * v
        cum_v += v
        vwap[i] = cum_pv / cum_v
    return vwap


class Strategy(BaseStrategy):
    name = "narrow_range_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip opening noise, need warmup
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 240            # 20-min warmup: 120 bars range_12 + 120 bars for mean
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("nr_ratio", 0.6, 0.4, 0.8),    # compression threshold
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Extract arrays — forward-fill NaN before converting
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        nr_ratio = params.get("nr_ratio", 0.6)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # ── Compression indicators ──────────────────────────────────────────────

        # 12-bar (60-second) rolling high/low — the "narrow range" window
        high_12 = _rolling_max(high, 12)
        low_12 = _rolling_min(low, 12)
        range_12 = high_12 - low_12
        # Replace NaN with 0 (no range yet — will be excluded by is_narrow check)
        range_12 = np.where(np.isnan(range_12), 0.0, range_12)

        # 120-bar (10-min) mean of range_12 — defines "typical" session range
        range_mean_120 = _rolling_mean(range_12, 120)
        # Where mean is NaN or zero, set to large value so is_narrow stays False
        range_mean_120 = np.where(
            np.isnan(range_mean_120) | (range_mean_120 <= 0.0),
            999.0,
            range_mean_120,
        )

        # ── VWAP ───────────────────────────────────────────────────────────────
        vwap = _compute_vwap(close, volume, day_id)

        # ── Signals ────────────────────────────────────────────────────────────

        # Narrow range condition: current 60s range < nr_ratio × 10-min mean range
        is_narrow = range_12 < (nr_ratio * range_mean_120)

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        buy_ce = np.zeros(n, dtype=bool)
        buy_pe = np.zeros(n, dtype=bool)

        for i in range(1, n):
            if not in_session[i]:
                continue
            # Previous bar must have been in compression
            if not is_narrow[i - 1]:
                continue
            prev_high = high_12[i - 1]
            prev_low = low_12[i - 1]
            if np.isnan(prev_high) or np.isnan(prev_low):
                continue
            # Bullish breakout: close above compressed zone high AND above VWAP
            if close[i] > prev_high and close[i] > vwap[i]:
                buy_ce[i] = True
            # Bearish breakout: close below compressed zone low AND below VWAP
            elif close[i] < prev_low and close[i] < vwap[i]:
                buy_pe[i] = True

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,            # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
