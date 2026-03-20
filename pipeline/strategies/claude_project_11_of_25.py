"""Previous Day High/Low Breakout v1 — claude_project_11_of_25

Thesis: Breaking the previous day's high or low with volume confirms
directional conviction beyond the prior session's range.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "prev_day_hl_breakout_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_mult", default=1.5, low=1.0, high=3.0),
            TunableParam("target_pct", default=0.005, low=0.002, high=0.010),
            TunableParam("stop_loss_pct", default=0.003, low=0.001, high=0.006),
            TunableParam("breakeven_pct", default=0.003, low=0.001, high=0.006),
            TunableParam("vix_max", default=24.0, low=16.0, high=32.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_mult = params.get("vol_mult", 1.5)
        target_pct = params.get("target_pct", 0.005)
        stop_pct = params.get("stop_loss_pct", 0.003)
        be_pct = params.get("breakeven_pct", 0.003)
        vix_max = params.get("vix_max", 24.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── Volume average ──
        avg_vol = df["volume"].rolling_mean(30).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)

        # ── Previous day high/low ──
        prev_high = np.zeros(n, dtype=np.float64)
        prev_low = np.zeros(n, dtype=np.float64)
        prev_valid = np.zeros(n, dtype=np.bool_)

        cur_day = -1
        day_h = -1e18
        day_l = 1e18
        last_day_h = 0.0
        last_day_l = 0.0
        has_prev = False
        for i in range(n):
            if day_id[i] != cur_day:
                if cur_day >= 0:
                    last_day_h = day_h
                    last_day_l = day_l
                    has_prev = True
                cur_day = day_id[i]
                day_h = high[i]
                day_l = low[i]
            else:
                day_h = max(day_h, high[i])
                day_l = min(day_l, low[i])
            if has_prev:
                prev_high[i] = last_day_h
                prev_low[i] = last_day_l
                prev_valid[i] = True

        # ── Filters ──
        vol_ok = volume > vol_mult * avg_vol
        vix_ok = vix < vix_max
        # Skip first 15 min of session
        time_ok = time_mins >= 585

        # ── Entry ──
        long_entry = prev_valid & (close > prev_high) & vol_ok & (close > vwap) & vix_ok & time_ok
        short_entry = prev_valid & (close < prev_low) & vol_ok & (close < vwap) & vix_ok & time_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            breakeven_pct=be_pct,
            time_stop_bars=240,
        )
