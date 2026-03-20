# AUDIT FIX: Added session window enforcement — entries were firing outside session_start/session_end
"""VWAP Z-Score Mean Reversion — gemini_1_of_20

Thesis: When price deviates significantly from VWAP in a low-trend environment
(low ADX, low VIX), fade the deviation expecting reversion to VWAP.
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
    name = "vwap_zscore_mean_reversion"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 870     # 14:30
    max_trades_per_day = 10

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_long_thresh", default=-3.0, low=-5.0, high=-1.5),
            TunableParam("zscore_short_thresh", default=3.0, low=1.5, high=5.0),
            TunableParam("adx_max", default=20.0, low=10.0, high=30.0),
            TunableParam("vix_max", default=20.0, low=12.0, high=30.0),
            TunableParam("stop_atr_mult", default=1.5, low=0.5, high=3.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_long = params.get("zscore_long_thresh", -3.0)
        zs_short = params.get("zscore_short_thresh", 3.0)
        adx_max = params.get("adx_max", 20.0)
        vix_max = params.get("vix_max", 20.0)
        stop_atr = params.get("stop_atr_mult", 1.5)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=close[0] if n > 0 else 0.0)

        # VWAP z-score over 20 bars
        dev = close - vwap
        period = 20
        std_arr = np.full(n, 1e10, dtype=np.float64)
        for i in range(period - 1, n):
            s = np.std(dev[i - period + 1: i + 1])
            std_arr[i] = max(s, 1e-10)
        zscore = dev / std_arr
        zscore = np.nan_to_num(zscore, nan=0.0)

        # ADX(14)
        atr14 = _compute_atr(high, low, close, 14)
        plus_dm = np.zeros(n, dtype=np.float64)
        minus_dm = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            up = high[i] - high[i - 1]
            dn = low[i - 1] - low[i]
            plus_dm[i] = up if (up > dn and up > 0) else 0.0
            minus_dm[i] = dn if (dn > up and dn > 0) else 0.0

        smooth_plus = np.zeros(n, dtype=np.float64)
        smooth_minus = np.zeros(n, dtype=np.float64)
        adx_period = 14
        if n >= adx_period + 1:
            smooth_plus[adx_period] = np.mean(plus_dm[1:adx_period + 1])
            smooth_minus[adx_period] = np.mean(minus_dm[1:adx_period + 1])
            for i in range(adx_period + 1, n):
                smooth_plus[i] = (smooth_plus[i - 1] * (adx_period - 1) + plus_dm[i]) / adx_period
                smooth_minus[i] = (smooth_minus[i - 1] * (adx_period - 1) + minus_dm[i]) / adx_period

        safe_atr = np.where(atr14 > 1e-10, atr14, 1e-10)
        plus_di = 100.0 * smooth_plus / safe_atr
        minus_di = 100.0 * smooth_minus / safe_atr
        dx = np.where((plus_di + minus_di) > 1e-10,
                       100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0.0)

        adx = np.zeros(n, dtype=np.float64)
        adx_start = 2 * adx_period
        if n > adx_start:
            adx[adx_start] = np.mean(dx[adx_period:adx_start + 1])
            for i in range(adx_start + 1, n):
                adx[i] = (adx[i - 1] * (adx_period - 1) + dx[i]) / adx_period

        # ATR(20)
        atr20 = _compute_atr(high, low, close, 20)

        # Session window filter
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        in_session = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # Entries
        long_entry = (zscore < zs_long) & (adx < adx_max) & (vix < vix_max) & in_session
        short_entry = (zscore > zs_short) & (adx < adx_max) & (vix < vix_max) & in_session

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr20,
            target_indicator=vwap.copy(),
            use_target_indicator=True,
            stop_loss_atr_mult=stop_atr,
            time_stop_bars=60,
        )
