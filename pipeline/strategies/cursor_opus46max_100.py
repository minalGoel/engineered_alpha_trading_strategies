"""ORB Closing Range v1 — cursor_opus46max_100

Thesis: The "Closing Range Breakout" applies the ORB concept to the last 30 minutes
of the trading day (14:30-15:00), trading breakouts from this range in the final
29 minutes (15:00-15:29). The closing range captures institutional end-of-day
positioning (mutual fund cash deployment, FII overnight adjustments, short-covering).
Mandatory exit by 15:28 IST.
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
    name = "cursor_opus46max_100"
    is_long_only = False
    session_start = 870   # 14:30
    session_end = 929     # 15:29
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("target_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("vix_high", default=22.0, low=18.0, high=28.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        stop_pct = params.get("stop_loss_pct", 0.002)
        target_pct = params.get("target_pct", 0.003)
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

        # -- VWAP (session VWAP) --
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # -- Session high/low (running through the day) --
        session_high = np.zeros(n, dtype=np.float64)
        session_low = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                session_high[i] = high[i]
                session_low[i] = low[i]
            else:
                session_high[i] = max(session_high[i - 1], high[i])
                session_low[i] = min(session_low[i - 1], low[i])

        # -- Closing Range Breakout (CRB): 14:30-15:00 range --
        crb_high = np.zeros(n, dtype=np.float64)
        crb_low = np.zeros(n, dtype=np.float64)
        crb_computed = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                cr_h = 0.0
                cr_l = 1e18
            # Build closing range during 14:30-15:00
            if time_min[i] >= 870 and time_min[i] < 900:
                if cr_h == 0.0:
                    cr_h = high[i]
                    cr_l = low[i]
                else:
                    cr_h = max(cr_h, high[i])
                    cr_l = min(cr_l, low[i])
            crb_high[i] = cr_h
            crb_low[i] = cr_l if cr_l < 1e18 else 0.0
            if time_min[i] >= 900:
                crb_computed[i] = True

        # -- Close vs session range position --
        sess_range = session_high - session_low
        sess_range_safe = np.where(sess_range > 0, sess_range, 1.0)
        close_vs_session = (close - session_low) / sess_range_safe

        # -- Momentum 5 --
        momentum_5 = np.zeros(n, dtype=np.float64)
        for i in range(5, n):
            momentum_5[i] = close[i] - close[i - 5]

        # -- Bar range position --
        bar_range = high - low
        bar_range_safe = np.where(bar_range > 0, bar_range, 1.0)
        bar_position = (close - low) / bar_range_safe

        # -- CRB range vs ATR filter --
        atr10 = _compute_atr(high, low, close, 10)

        vix_ok = vix < vix_hi
        # Enter 15:00-15:20 only (need time for trade to develop)
        time_ok = (time_min >= 900) & (time_min <= 920)

        # Skip if CRB range > 2 * ATR_10
        crb_range = crb_high - crb_low
        atr_safe = np.where(atr10 > 0, atr10, 1e18)
        crb_not_too_wide = crb_range < 2.0 * atr_safe

        long_entry = (
            crb_computed & (close > crb_high) & (crb_high > 0) &
            (close > vwap) & (close_vs_session > 0.5) &
            (bar_position > 0.7) & (momentum_5 > 0) &
            crb_not_too_wide & vix_ok & time_ok
        )
        short_entry = (
            crb_computed & (close < crb_low) & (crb_low > 0) &
            (close < vwap) & (close_vs_session < 0.5) &
            (bar_position < 0.3) & (momentum_5 < 0) &
            crb_not_too_wide & vix_ok & time_ok
        )

        # -- Signal exit: price crosses CRB mid (re-enters range fully) --
        crb_mid = (crb_high + crb_low) / 2.0
        signal_exit_long = (close < crb_mid) & crb_computed
        signal_exit_short = (close > crb_mid) & crb_computed

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            time_stop_bars=29,
        )
