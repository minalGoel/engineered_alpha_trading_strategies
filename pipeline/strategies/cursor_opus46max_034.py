"""SuperTrend Momentum v1 — cursor_opus46max_034

Thesis: SuperTrend(10, 3.0) on 1-min bars exploits the self-fulfilling
prophecy effect — many Indian retail traders on Zerodha use this indicator.
Entry on SuperTrend flip with volume confirmation and VWAP alignment.
Quick scalp target before the crowd.
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


def _compute_supertrend(high, low, close, period, multiplier):
    n = len(close)
    atr = _compute_atr(high, low, close, period)
    hl2 = (high + low) / 2.0

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend = np.zeros(n, dtype=np.float64)
    direction = np.ones(n, dtype=np.int32)  # 1=bull, -1=bear

    supertrend[0] = upper_band[0]
    direction[0] = -1

    for i in range(1, n):
        if close[i-1] > supertrend[i-1]:
            # Previous was bullish
            lower_band[i] = max(lower_band[i], lower_band[i-1]) if lower_band[i-1] > 0 else lower_band[i]

        if close[i-1] < supertrend[i-1]:
            # Previous was bearish
            upper_band[i] = min(upper_band[i], upper_band[i-1]) if upper_band[i-1] > 0 else upper_band[i]

        if supertrend[i-1] == upper_band[i-1]:
            # Was bearish
            if close[i] > upper_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
        else:
            # Was bullish
            if close[i] < lower_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1

    return supertrend, direction


class Strategy(BaseStrategy):
    name = "cursor_opus46max_034"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 885     # 14:45
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("st_period", default=10.0, low=7.0, high=15.0),
            TunableParam("st_mult", default=3.0, low=2.0, high=4.0),
            TunableParam("vol_ratio_min", default=1.5, low=1.0, high=2.5),
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("target_pct", default=0.0035, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        st_period = int(params.get("st_period", 10.0))
        st_mult = params.get("st_mult", 3.0)
        vol_min = params.get("vol_ratio_min", 1.5)
        stop_pct = params.get("stop_loss_pct", 0.002)
        tgt_pct = params.get("target_pct", 0.0035)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── ATR(14) ──
        atr = _compute_atr(high, low, close, 14)

        # ── SuperTrend ──
        st_val, st_dir = _compute_supertrend(high, low, close, st_period, st_mult)

        # ── Volume ratio ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ratio = volume / avg_vol

        # ── Flip detection ──
        flip_bull = np.zeros(n, dtype=np.bool_)
        flip_bear = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if st_dir[i] == 1 and st_dir[i-1] == -1:
                flip_bull[i] = True
            if st_dir[i] == -1 and st_dir[i-1] == 1:
                flip_bear[i] = True

        # ── Flip bar closes above prev bar high (long) or below prev low (short) ──
        above_prev_high = np.zeros(n, dtype=np.bool_)
        below_prev_low = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            above_prev_high[i] = close[i] > high[i-1]
            below_prev_low[i] = close[i] < low[i-1]

        # ── Whipsaw filter: count flips in last 20 bars ──
        flips = (flip_bull | flip_bear).astype(np.int32)
        flip_count_20 = np.zeros(n, dtype=np.int32)
        for i in range(n):
            start = max(0, i - 19)
            flip_count_20[i] = np.sum(flips[start:i+1])

        # ── Time filter ──
        time_ok = (time_mins >= 570) & (time_mins <= 885)

        # ── Entries ──
        long_entry = (
            flip_bull & (close > vwap) & (vol_ratio > vol_min) &
            above_prev_high & (flip_count_20 <= 3) & time_ok
        )
        short_entry = (
            flip_bear & (close < vwap) & (vol_ratio > vol_min) &
            below_prev_low & (flip_count_20 <= 3) & time_ok
        )

        # ── Signal exits: ST re-flips ──
        signal_exit_long = flip_bear
        signal_exit_short = flip_bull

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=tgt_pct,
            trailing_stop_pct=0.001,
            trailing_activate_pct=0.002,
            time_stop_bars=30,
        )
