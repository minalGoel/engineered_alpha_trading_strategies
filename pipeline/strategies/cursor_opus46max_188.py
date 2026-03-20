"""Maximum Entropy v1 — cursor_opus46max_188

Thesis: Use rolling moments (mean, variance, skewness) of 1-min returns to
construct a calibrated distribution. When observed return falls in extreme
tails (beyond 5th/95th percentile), fade the extreme with confirmation.
Simplified: use empirical percentiles instead of full MaxEnt fitting.
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
    name = "cursor_opus46max_188"
    is_long_only = False
    session_start = 615   # 10:15
    session_end = 910     # 15:10
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("moment_window", default=60.0, low=40.0, high=90.0),
            TunableParam("tail_pctile", default=5.0, low=2.0, high=10.0),
            TunableParam("skew_max", default=1.0, low=0.5, high=2.0),
            TunableParam("vol_mult", default=1.5, low=1.0, high=2.5),
            TunableParam("stop_loss_pct", default=0.0020, low=0.0012, high=0.003),
            TunableParam("target_pct", default=0.0030, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        mom_win = int(params.get("moment_window", 60.0))
        tail_p = params.get("tail_pctile", 5.0)
        skew_max = params.get("skew_max", 1.0)
        vol_mult = params.get("vol_mult", 1.5)
        stop_pct = params.get("stop_loss_pct", 0.0020)
        target_pct = params.get("target_pct", 0.0030)

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

        # 1-min returns
        ret = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i-1] > 0:
                ret[i] = (close[i] - close[i-1]) / close[i-1] * 10000  # bps

        # Rolling empirical percentiles (MaxEnt proxy)
        p5 = np.zeros(n, dtype=np.float64)
        p95 = np.zeros(n, dtype=np.float64)
        rolling_skew = np.zeros(n, dtype=np.float64)
        for i in range(mom_win, n):
            seg = ret[i-mom_win+1:i+1]
            p5[i] = np.percentile(seg, tail_p)
            p95[i] = np.percentile(seg, 100.0 - tail_p)
            mu = np.mean(seg)
            std = np.std(seg)
            if std > 1e-10:
                rolling_skew[i] = np.mean(((seg - mu) / std) ** 3)

        # Tail event detection
        extreme_low = ret < p5  # extreme negative
        extreme_high = ret > p95  # extreme positive

        # Volume confirmation
        avg_vol = np.zeros(n, dtype=np.float64)
        for i in range(19, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_surge = volume > (vol_mult * avg_vol)

        # Next bar confirmation: uptick after extreme low, downtick after extreme high
        confirm_bounce = np.zeros(n, dtype=np.bool_)
        confirm_fade = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if i >= 2 and extreme_low[i-1] and ret[i] > 0:
                confirm_bounce[i] = True
            if i >= 2 and extreme_high[i-1] and ret[i] < 0:
                confirm_fade[i] = True

        time_ok = (time_mins >= 615) & (time_mins <= 910)

        long_entry = (
            confirm_bounce
            & (close < vwap)
            & (rolling_skew > -skew_max)
            & vol_surge
            & time_ok
        )
        short_entry = (
            confirm_fade
            & (close > vwap)
            & (rolling_skew < skew_max)
            & vol_surge
            & time_ok
        )

        # Signal exit: return back within normal range
        sig_exit_long = (ret > 0) & (ret < p95)  # normalized
        sig_exit_short = (ret < 0) & (ret > p5)

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_activate_pct=0.0020,
            trailing_stop_pct=0.0010,
            time_stop_bars=45,
        )
