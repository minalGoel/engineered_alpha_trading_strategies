"""VWAP Band Squeeze v1 — cursor_opus46max_004

Thesis: When VWAP bands (VWAP +/- k*rolling_std) compress below the 20th
percentile of session bandwidth, a breakout is imminent. Enter on breakout
outside the bands with volume confirmation. Target = 2 ATR from entry.
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
    name = "cursor_opus46max_004"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 870     # 14:30
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("bw_pctile_thresh", default=20.0, low=10.0, high=35.0),
            TunableParam("vol_ratio_thresh", default=1.5, low=1.0, high=2.5),
            TunableParam("vix_max", default=22.0, low=14.0, high=28.0),
            TunableParam("target_atr_mult", default=2.0, low=1.0, high=3.0),
            TunableParam("bw_exit_pctile", default=60.0, low=40.0, high=80.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        bw_pctile = params.get("bw_pctile_thresh", 20.0)
        vol_thresh = params.get("vol_ratio_thresh", 1.5)
        vix_max = params.get("vix_max", 22.0)
        tgt_atr = params.get("target_atr_mult", 2.0)
        bw_exit = params.get("bw_exit_pctile", 60.0)

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
        day_id = df["day_id"].to_numpy()

        # ── VWAP bands: VWAP +/- 2 * rolling_std(close-vwap, 20) ──
        dev = close - vwap
        std_20 = df.with_columns(
            (pl.col("close") - pl.lit(pl.Series(vwap))).alias("_dev")
        )["_dev"].rolling_std(20).to_numpy().astype(np.float64)
        std_20 = np.nan_to_num(std_20, nan=1e10)
        std_20 = np.clip(std_20, 1e-10, None)

        vwap_upper = vwap + 2.0 * std_20
        vwap_lower = vwap - 2.0 * std_20

        safe_vwap = np.where(vwap > 0, vwap, 1.0)
        bw = (vwap_upper - vwap_lower) / safe_vwap * 100.0

        # ── Bandwidth percentile within session (per day_id) ──
        bw_pct_arr = np.zeros(n, dtype=np.float64)
        unique_days = np.unique(day_id)
        for d in unique_days:
            mask = day_id == d
            idx = np.where(mask)[0]
            bw_day = bw[idx]
            for j, gi in enumerate(idx):
                if j < 20:
                    bw_pct_arr[gi] = 50.0
                else:
                    window = bw_day[:j + 1]
                    bw_pct_arr[gi] = np.sum(window < bw_day[j]) / len(window) * 100.0

        # ── Volume ratio ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_thresh * avg_vol

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Bar range vs ATR threshold ──
        bar_range = high - low
        atr_frac = atr / np.sqrt(14.0)
        atr_frac = np.clip(atr_frac, 1e-10, None)
        big_bar = bar_range > 1.5 * atr_frac

        # ── Filters ──
        time_ok = (time_mins >= 600) & (time_mins <= 870)
        vix_ok = (vix >= 12) & (vix <= vix_max)

        # ── Entries ──
        long_entry = (
            (bw_pct_arr < bw_pctile)
            & (close > vwap_upper)
            & vol_ok
            & (close > vwap)
            & big_bar
            & time_ok & vix_ok
        )
        short_entry = (
            (bw_pct_arr < bw_pctile)
            & (close < vwap_lower)
            & vol_ok
            & (close < vwap)
            & big_bar
            & time_ok & vix_ok
        )

        # ── Signal exit: bands expanding past threshold ──
        sig_exit_long = bw_pct_arr > bw_exit
        sig_exit_short = bw_pct_arr > bw_exit

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=vwap.copy(),
            target_atr_mult=tgt_atr,
            stop_loss_pct=0.0,
            use_target_indicator=False,
            time_stop_bars=45,
        )
