"""Dynamic Stop Regime v1 — cursor_opus46max_180

Thesis: Fixed stop-losses are suboptimal. Dynamically set stop distance based on
ATR percentile regime. Low-vol: stop=1.0*ATR. Med-vol: 1.5*ATR. High-vol: 2.5*ATR.
Base signal is EMA(10)/EMA(25) crossover with VWAP confirmation.
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


def _compute_ema(data, period):
    n = len(data)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    ema[0] = data[0]
    k = 2.0 / (period + 1.0)
    for i in range(1, n):
        ema[i] = data[i] * k + ema[i-1] * (1.0 - k)
    return ema


class Strategy(BaseStrategy):
    name = "cursor_opus46max_180"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 915     # 15:15
    max_trades_per_day = 7

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("ema_fast", default=10.0, low=5.0, high=15.0),
            TunableParam("ema_slow", default=25.0, low=18.0, high=35.0),
            TunableParam("atr_period", default=14.0, low=10.0, high=20.0),
            TunableParam("atr_pctile_window", default=120.0, low=60.0, high=180.0),
            TunableParam("rr_ratio", default=1.8, low=1.3, high=2.5),
            TunableParam("max_stop_bps", default=60.0, low=40.0, high=80.0),
            TunableParam("vix_max", default=28.0, low=20.0, high=35.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        ema_fast_p = int(params.get("ema_fast", 10.0))
        ema_slow_p = int(params.get("ema_slow", 25.0))
        atr_period = int(params.get("atr_period", 14.0))
        pctile_win = int(params.get("atr_pctile_window", 120.0))
        rr_ratio = params.get("rr_ratio", 1.8)
        max_stop_bps = params.get("max_stop_bps", 60.0)
        vix_max = params.get("vix_max", 28.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # EMA crossover
        ema_f = _compute_ema(close, ema_fast_p)
        ema_s = _compute_ema(close, ema_slow_p)

        # ATR
        atr = _compute_atr(high, low, close, atr_period)

        # ATR percentile
        atr_pctile = np.zeros(n, dtype=np.float64)
        for i in range(pctile_win, n):
            window = atr[i-pctile_win+1:i+1]
            cnt = np.sum(window <= atr[i])
            atr_pctile[i] = cnt / pctile_win * 100.0

        # Dynamic stop multiplier based on vol regime
        stop_mult = np.where(atr_pctile < 30, 1.0,
                   np.where(atr_pctile < 70, 1.5, 2.5))

        # Stop distance in bps
        safe_close = np.where(close > 0, close, 1.0)
        stop_dist_bps = stop_mult * atr / safe_close * 10000

        # Volume filter
        avg_vol_15 = np.zeros(n, dtype=np.float64)
        for i in range(14, n):
            avg_vol_15[i] = np.mean(volume[i-14:i+1])
        avg_vol_15 = np.clip(avg_vol_15, 1.0, None)
        vol_ok = volume > avg_vol_15

        # EMA crossover detection
        cross_up = np.zeros(n, dtype=np.bool_)
        cross_down = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if ema_f[i-1] <= ema_s[i-1] and ema_f[i] > ema_s[i]:
                cross_up[i] = True
            if ema_f[i-1] >= ema_s[i-1] and ema_f[i] < ema_s[i]:
                cross_down[i] = True

        time_ok = (time_mins >= 565) & (time_mins <= 915)
        vix_ok = vix < vix_max

        long_entry = (
            cross_up & (close > vwap)
            & (stop_dist_bps < max_stop_bps)
            & vol_ok & vix_ok & time_ok
        )
        short_entry = (
            cross_down & (close < vwap)
            & (stop_dist_bps < max_stop_bps)
            & vol_ok & vix_ok & time_ok
        )

        # Signal exit: EMA crosses against position
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if ema_f[i-1] >= ema_s[i-1] and ema_f[i] < ema_s[i]:
                sig_exit_long[i] = True
            if ema_f[i-1] <= ema_s[i-1] and ema_f[i] > ema_s[i]:
                sig_exit_short[i] = True

        # Use median stop as scalar parameter (regime-average)
        median_stop_pct = np.median(stop_mult * atr / safe_close)
        median_stop_pct = max(0.001, min(median_stop_pct, 0.006))
        target_pct = median_stop_pct * rr_ratio

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=median_stop_pct,
            stop_loss_atr_mult=1.5,
            target_pct=target_pct,
            trailing_activate_pct=median_stop_pct,
            trailing_stop_pct=median_stop_pct * 0.6,
            time_stop_bars=75,
        )
