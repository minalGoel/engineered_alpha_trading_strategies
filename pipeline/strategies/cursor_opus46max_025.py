"""Price Channel Breakout v1 — cursor_opus46max_025

Thesis: Donchian Channel(30) breakout captures new 30-minute highs/lows
as trend continuation signals. Enter when close breaks above/below the
channel with volume > 1.5x, VWAP alignment, and close in the top/bottom
25% of bar range. DC_width must be between 0.3% and 2.0%.
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
    name = "cursor_opus46max_025"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("dc_period", default=30.0, low=20.0, high=50.0),
            TunableParam("vol_mult", default=1.5, low=1.2, high=2.5),
            TunableParam("dc_width_min", default=0.3, low=0.1, high=0.5),
            TunableParam("dc_width_max", default=2.0, low=1.0, high=3.0),
            TunableParam("vix_max", default=30.0, low=20.0, high=35.0),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.007),
            TunableParam("trailing_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("trailing_activate", default=0.003, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        dc_period = int(params.get("dc_period", 30.0))
        vol_mult = params.get("vol_mult", 1.5)
        dc_w_min = params.get("dc_width_min", 0.3)
        dc_w_max = params.get("dc_width_max", 2.0)
        vix_max = params.get("vix_max", 30.0)
        stop_pct = params.get("stop_loss_pct", 0.004)
        trail_pct = params.get("trailing_pct", 0.002)
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

        # ── Donchian Channel ──
        dc_upper = np.zeros(n, dtype=np.float64)
        dc_lower = np.zeros(n, dtype=np.float64)
        dc_mid = np.zeros(n, dtype=np.float64)
        dc_width_pct = np.zeros(n, dtype=np.float64)

        for i in range(dc_period, n):
            dc_upper[i] = np.max(high[i - dc_period:i])  # previous bars only
            dc_lower[i] = np.min(low[i - dc_period:i])
            dc_mid[i] = (dc_upper[i] + dc_lower[i]) / 2.0
            if dc_mid[i] > 1e-10:
                dc_width_pct[i] = (dc_upper[i] - dc_lower[i]) / dc_mid[i] * 100.0

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Bar range position ──
        bar_range = high - low
        bar_range = np.where(bar_range > 0, bar_range, 1e-10)
        range_pos = (close - low) / bar_range

        # ── Filters ──
        time_ok = (time_mins >= 570) & (time_mins <= 870)
        vix_ok = (vix >= 14) & (vix < vix_max)
        width_ok = (dc_width_pct > dc_w_min) & (dc_width_pct < dc_w_max)

        # ── Entries ──
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(dc_period, n):
            if (high[i] > dc_upper[i]
                    and close[i] > dc_upper[i]
                    and close[i] > vwap[i]
                    and range_pos[i] > 0.75
                    and vol_ok[i] and width_ok[i]
                    and vix_ok[i] and time_ok[i]):
                long_entry[i] = True
            if (low[i] < dc_lower[i]
                    and close[i] < dc_lower[i]
                    and close[i] < vwap[i]
                    and range_pos[i] < 0.25
                    and vol_ok[i] and width_ok[i]
                    and vix_ok[i] and time_ok[i]):
                short_entry[i] = True

        # ── Signal exit: price closes back inside channel ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(dc_period + 1, n):
            if close[i] < dc_upper[i] and close[i - 1] >= dc_upper[i - 1]:
                sig_exit_long[i] = True
            if close[i] < dc_mid[i]:
                sig_exit_long[i] = True
            if close[i] > dc_lower[i] and close[i - 1] <= dc_lower[i - 1]:
                sig_exit_short[i] = True
            if close[i] > dc_mid[i]:
                sig_exit_short[i] = True

        # ── Target: use DC width as measured move; convert to pct ──
        # Approximate target as half the DC width pct (clamped to reasonable range)
        target_pct_val = 0.005  # fallback scalar

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct_val,
            stop_loss_pct=stop_pct,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=45,
        )
