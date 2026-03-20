"""Drawdown Adaptive v1 — cursor_opus46max_176

Thesis: Anti-martingale position sizing using Supertrend + VWAP trend-following.
During drawdown periods, reduce position size to preserve capital; during growth,
increase to compound gains. Supertrend(10,3) for direction, VWAP for confirmation,
close must remain on Supertrend side for 2 consecutive bars.
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


def _compute_supertrend(high, low, close, atr, multiplier):
    """Compute Supertrend indicator. Returns supertrend line and direction (+1/-1)."""
    n = len(close)
    supertrend = np.zeros(n, dtype=np.float64)
    direction = np.ones(n, dtype=np.float64)  # +1 = bullish, -1 = bearish

    upper_band = np.zeros(n, dtype=np.float64)
    lower_band = np.zeros(n, dtype=np.float64)

    for i in range(n):
        hl2 = (high[i] + low[i]) / 2.0
        upper_band[i] = hl2 + multiplier * atr[i]
        lower_band[i] = hl2 - multiplier * atr[i]

    for i in range(1, n):
        if lower_band[i] < lower_band[i-1] and close[i-1] > lower_band[i-1]:
            lower_band[i] = lower_band[i-1]
        if upper_band[i] > upper_band[i-1] and close[i-1] < upper_band[i-1]:
            upper_band[i] = upper_band[i-1]

        if direction[i-1] == 1.0:
            if close[i] < lower_band[i]:
                direction[i] = -1.0
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1.0
                supertrend[i] = lower_band[i]
        else:
            if close[i] > upper_band[i]:
                direction[i] = 1.0
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1.0
                supertrend[i] = upper_band[i]

    return supertrend, direction


class Strategy(BaseStrategy):
    name = "cursor_opus46max_176"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("st_period", default=10.0, low=7.0, high=14.0),
            TunableParam("st_mult", default=3.0, low=2.0, high=4.0),
            TunableParam("confirm_bars", default=2.0, low=1.0, high=4.0),
            TunableParam("stop_loss_pct", default=0.0025, low=0.0015, high=0.004),
            TunableParam("target_pct", default=0.0045, low=0.003, high=0.007),
            TunableParam("trailing_activate_pct", default=0.0025, low=0.0015, high=0.004),
            TunableParam("trailing_stop_pct", default=0.0015, low=0.001, high=0.003),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        st_period = int(params.get("st_period", 10.0))
        st_mult = params.get("st_mult", 3.0)
        confirm = int(params.get("confirm_bars", 2.0))
        stop_pct = params.get("stop_loss_pct", 0.0025)
        target_pct = params.get("target_pct", 0.0045)
        trail_act = params.get("trailing_activate_pct", 0.0025)
        trail_pct = params.get("trailing_stop_pct", 0.0015)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ATR and Supertrend
        atr = _compute_atr(high, low, close, st_period)
        supertrend, st_dir = _compute_supertrend(high, low, close, atr, st_mult)

        # Volume filter: volume > SMA(volume, 15)
        avg_vol_15 = np.zeros(n, dtype=np.float64)
        for i in range(14, n):
            avg_vol_15[i] = np.mean(volume[i-14:i+1])
        avg_vol_15 = np.clip(avg_vol_15, 1.0, None)
        vol_ok = volume > avg_vol_15

        # Confirmation: close above/below supertrend for `confirm` consecutive bars
        above_st = close > supertrend
        below_st = close < supertrend
        consec_above = np.zeros(n, dtype=np.int32)
        consec_below = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            consec_above[i] = (consec_above[i-1] + 1) if above_st[i] else 0
            consec_below[i] = (consec_below[i-1] + 1) if below_st[i] else 0

        time_ok = (time_mins >= 560) & (time_mins <= 915)

        # Entry conditions
        long_entry = (
            (consec_above >= confirm)
            & (close > vwap)
            & vol_ok
            & time_ok
        )
        short_entry = (
            (consec_below >= confirm)
            & (close < vwap)
            & vol_ok
            & time_ok
        )

        # Signal exit: Supertrend flips direction
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if st_dir[i-1] == 1.0 and st_dir[i] == -1.0:
                sig_exit_long[i] = True
            if st_dir[i-1] == -1.0 and st_dir[i] == 1.0:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_activate_pct=trail_act,
            trailing_stop_pct=trail_pct,
            time_stop_bars=75,
        )
