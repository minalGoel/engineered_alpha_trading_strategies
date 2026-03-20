"""ORB 5-min v1 — cursor_opus46max_086

Thesis: 5-minute opening range (09:15-09:20) breakout with volume confirmation.
Fast variant for high-conviction gap days. Sub-50% win rate but 1.8:1 payoff.
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
    name = "cursor_opus46max_086"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 840     # 14:00
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_mult", default=1.5, low=1.0, high=2.5),
            TunableParam("orb_range_vs_atr_max", default=1.5, low=1.0, high=2.0),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("target_atr_mult", default=2.0, low=1.0, high=3.0),
            TunableParam("vix_low", default=12.0, low=8.0, high=16.0),
            TunableParam("vix_high", default=22.0, low=18.0, high=28.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_mult = params.get("vol_mult", 1.5)
        orb_atr_max = params.get("orb_range_vs_atr_max", 1.5)
        stop_pct = params.get("stop_loss_pct", 0.003)
        target_atr_mult = params.get("target_atr_mult", 2.0)
        vix_lo = params.get("vix_low", 12.0)
        vix_hi = params.get("vix_high", 22.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_min = df["time_minutes"].to_numpy()
        day_id = df["day_id"].to_numpy()

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── ORB 5-min (09:15-09:20 = time_min 555-560) ──
        orb_high = np.zeros(n, dtype=np.float64)
        orb_low = np.zeros(n, dtype=np.float64)
        orb_range = np.zeros(n, dtype=np.float64)
        orb_computed = np.zeros(n, dtype=np.bool_)

        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                cur_orb_h = high[i]
                cur_orb_l = low[i]
            if time_min[i] < 560:
                cur_orb_h = max(cur_orb_h, high[i])
                cur_orb_l = min(cur_orb_l, low[i])
            orb_high[i] = cur_orb_h
            orb_low[i] = cur_orb_l
            orb_range[i] = cur_orb_h - cur_orb_l
            if time_min[i] >= 560:
                orb_computed[i] = True

        # ── ATR(14) ──
        atr = _compute_atr(high, low, close, 14)

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol

        # ── ORB range not over-extended vs ATR ──
        atr_safe = np.clip(atr, 1e-10, None)
        orb_vs_atr = np.where(atr_safe > 0, orb_range / atr_safe, 10.0)
        range_ok = orb_vs_atr < orb_atr_max

        vix_ok = (vix >= vix_lo) & (vix <= vix_hi)
        time_ok = (time_min >= 560) & (time_min <= 585)  # 09:20-09:45

        long_entry = orb_computed & (close > orb_high) & (close > vwap) & vol_ok & range_ok & vix_ok & time_ok
        short_entry = orb_computed & (close < orb_low) & (close < vwap) & vol_ok & range_ok & vix_ok & time_ok

        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_atr_mult=target_atr_mult,
            trailing_stop_pct=0.003,
            trailing_activate_pct=0.003,
            time_stop_bars=120,
        )
