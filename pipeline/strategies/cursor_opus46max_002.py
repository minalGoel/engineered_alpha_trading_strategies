"""VWAP Z-Score Bounce v1 — cursor_opus46max_002

Thesis: Z-score of (price - VWAP) normalised by rolling std identifies
statistically extreme deviations (>2 sigma). Enter when z-score turns back
from extreme; target is z-score crossing zero.
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
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "cursor_opus46max_002"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 900     # 15:00
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_thresh", default=2.0, low=1.5, high=3.0),
            TunableParam("vol_ratio_thresh", default=1.2, low=0.8, high=2.0),
            TunableParam("vix_max", default=24.0, low=16.0, high=30.0),
            TunableParam("stop_atr_mult", default=1.2, low=0.8, high=2.0),
            TunableParam("trailing_pct", default=0.003, low=0.001, high=0.006),
            TunableParam("trailing_activate", default=0.003, low=0.001, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_thresh = params.get("zscore_thresh", 2.0)
        vol_thresh = params.get("vol_ratio_thresh", 1.2)
        vix_max = params.get("vix_max", 24.0)
        stop_atr = params.get("stop_atr_mult", 1.2)
        trail_pct = params.get("trailing_pct", 0.003)
        trail_act = params.get("trailing_activate", 0.003)

        n = len(df)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP deviation and z-score ──
        vwap_dev = close - vwap
        std_30 = df["close"].rolling_std(30).to_numpy().astype(np.float64)
        std_30 = np.nan_to_num(std_30, nan=1e10)
        std_30 = np.clip(std_30, 1e-10, None)
        zscore = vwap_dev / std_30
        zscore = np.nan_to_num(zscore, nan=0.0)

        # ── Volume ratio ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_thresh * avg_vol

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Z-score turning (confirmation) ──
        zs_turning_up = np.zeros(n, dtype=np.bool_)
        zs_turning_down = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            zs_turning_up[i] = zscore[i] > zscore[i - 1]
            zs_turning_down[i] = zscore[i] < zscore[i - 1]

        # ── Filters ──
        time_ok = (time_mins >= 585) & (time_mins <= 870)
        vix_ok = vix < vix_max

        # ── Entries ──
        long_entry = (zscore < -zs_thresh) & zs_turning_up & vol_ok & vix_ok & time_ok
        short_entry = (zscore > zs_thresh) & zs_turning_down & vol_ok & vix_ok & time_ok

        # ── Signal exits: z-score further extreme OR z-score crosses zero ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            # exit long if z-score crosses zero or goes to -2.5
            if zscore[i] >= 0.0 and zscore[i - 1] < 0.0:
                sig_exit_long[i] = True
            if zscore[i] < -(zs_thresh + 0.5):
                sig_exit_long[i] = True
            # exit short if z-score crosses zero or goes to +2.5
            if zscore[i] <= 0.0 and zscore[i - 1] > 0.0:
                sig_exit_short[i] = True
            if zscore[i] > (zs_thresh + 0.5):
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=vwap.copy(),
            stop_loss_atr_mult=stop_atr,
            use_target_indicator=True,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=45,
        )
