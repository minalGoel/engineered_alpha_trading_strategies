"""Expiry Day Pattern v1 — cursor_opus46max_123

Thesis: F&O expiry days exhibit distinct patterns: morning pin toward
max pain, afternoon unwind with breakouts.  Proxy max pain using VWAP
and detect gamma-squeeze-like breakouts in the afternoon.
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
    name = "cursor_opus46max_123"
    is_long_only = False
    session_start = 575   # 09:35
    session_end = 929     # 15:29
    max_trades_per_day = 6
    assumptions = ["Max pain proxied via VWAP; gamma squeeze proxied via breakout with volume"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("morning_vwap_dev", default=30.0, low=15.0, high=50.0),
            TunableParam("afternoon_lookback", default=30, low=15, high=60),
            TunableParam("vol_surge", default=1.5, low=1.2, high=2.5),
            TunableParam("vix_min", default=12.0, low=8.0, high=16.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.0015, low=0.0008, high=0.003),
            TunableParam("target_pct", default=0.003, low=0.0015, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        morning_dev = params.get("morning_vwap_dev", 30.0)
        afternoon_lb = int(params.get("afternoon_lookback", 30))
        vol_surge = params.get("vol_surge", 1.5)
        vix_min = params.get("vix_min", 12.0)
        vix_max = params.get("vix_max", 25.0)
        stop_pct = params.get("stop_loss_pct", 0.0015)
        target_pct = params.get("target_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        volume = np.clip(volume, 1.0, None)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP (proxy for max pain)
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = np.nan_to_num(df_v["_vwap"].to_numpy().astype(np.float64), nan=close[0] if n > 0 else 0.0)

        # Distance to VWAP in bps (proxy for distance to max pain)
        safe_vwap = np.where(vwap > 1e-10, vwap, 1e-10)
        dist_vwap = (close - vwap) / safe_vwap * 10000.0

        # Volume ratio
        avg_vol = np.ones(n, dtype=np.float64)
        for i in range(19, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ratio = volume / avg_vol

        # Phase detection
        morning_phase = (time_mins >= 575) & (time_mins < 780)  # until 13:00
        afternoon_phase = (time_mins >= 810) & (time_mins <= 925)  # 13:30-15:25

        # Rolling high/low for afternoon breakout
        roll_high = np.zeros(n, dtype=np.float64)
        roll_low = np.full(n, 1e10, dtype=np.float64)
        for i in range(afternoon_lb, n):
            roll_high[i] = np.max(high[i - afternoon_lb: i])
            roll_low[i] = np.min(low[i - afternoon_lb: i])

        vix_ok = (vix > vix_min) & (vix < vix_max)

        # Morning: mean reversion toward VWAP (proxy max pain)
        morning_long = np.zeros(n, dtype=np.bool_)
        morning_short = np.zeros(n, dtype=np.bool_)
        for i in range(2, n):
            if (morning_phase[i] and dist_vwap[i] < -morning_dev and
                    close[i] > close[i-1] and close[i-1] > close[i-2] and vix_ok[i]):
                morning_long[i] = True
            if (morning_phase[i] and dist_vwap[i] > morning_dev and
                    close[i] < close[i-1] and close[i-1] < close[i-2] and vix_ok[i]):
                morning_short[i] = True

        # Afternoon: breakout (gamma squeeze proxy)
        afternoon_long = (afternoon_phase & (close > roll_high) &
                          (vol_ratio > vol_surge) & vix_ok)
        afternoon_short = (afternoon_phase & (close < roll_low) &
                           (vol_ratio > vol_surge) & vix_ok)

        long_entry = morning_long | afternoon_long
        short_entry = morning_short | afternoon_short

        # Signal exit: morning trades by 13:00 transition, afternoon by price reversal
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            # Close morning trades at phase transition
            if morning_phase[i-1] and not morning_phase[i]:
                sig_exit_long[i] = True
                sig_exit_short[i] = True

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=vwap.copy(),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_stop_pct=0.001,
            trailing_activate_pct=0.002,
            time_stop_bars=45,
        )
