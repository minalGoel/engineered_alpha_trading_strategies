# AUDIT FIX: Added session time filter — entries were firing outside session_start/session_end window
"""MA Crossover Momentum — Grok_7_of_10

Thesis: 9/21 EMA crossover with volume and index direction confirmation
in trending sessions. ATR-based stops and targets.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    if n < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "ma_crossover_momentum_v1"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_max", default=20.0, low=12.0, high=30.0),
            TunableParam("rel_vol_thresh", default=1.4, low=1.0, high=3.0),
            TunableParam("stop_atr_mult", default=1.0, low=0.5, high=2.0),
            TunableParam("target_atr_mult", default=2.0, low=1.0, high=4.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_max = params.get("vix_max", 20.0)
        rv_thresh = params.get("rel_vol_thresh", 1.4)
        stop_atr = params.get("stop_atr_mult", 1.0)
        tgt_atr = params.get("target_atr_mult", 2.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        idx = np.nan_to_num(df["index_close"].to_numpy().astype(np.float64), nan=0.0)

        # ── Session time filter ──
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        atr14 = _compute_atr(high, low, close, 14)

        # EMA 9 and 21
        ema9 = df["close"].ewm_mean(span=9).to_numpy().astype(np.float64)
        ema21 = df["close"].ewm_mean(span=21).to_numpy().astype(np.float64)
        ema9 = np.nan_to_num(ema9, nan=0.0)
        ema21 = np.nan_to_num(ema21, nan=0.0)

        # Relative volume
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)
        rel_vol = volume / avg_vol_20

        # Index 5-bar return
        idx_ret5 = np.zeros(n, dtype=np.float64)
        for i in range(5, n):
            if idx[i-5] > 0:
                idx_ret5[i] = (idx[i] - idx[i-5]) / idx[i-5]

        # Filters
        vix_ok = vix < vix_max
        vol_ok = rel_vol > rv_thresh

        # Entry: EMA crossover + direction confirmation
        long_entry = (ema9 > ema21) & vol_ok & vix_ok & (idx_ret5 > 0) & time_ok
        short_entry = (ema9 < ema21) & vol_ok & vix_ok & (idx_ret5 < 0) & time_ok

        # Signal exit: opposite crossover
        signal_exit_long = ema9 < ema21
        signal_exit_short = ema9 > ema21

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=stop_atr,
            target_atr_mult=tgt_atr,
            time_stop_bars=30,
        )
