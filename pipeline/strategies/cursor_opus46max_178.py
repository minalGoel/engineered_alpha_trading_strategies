"""Momentum Decay Adaptive v1 — cursor_opus46max_178

Thesis: Momentum decays at different rates depending on market conditions.
Compute the half-life via AR(1) coefficient on 1-min returns. When half-life
is short (<10 bars), use tight targets/quick exits. When long (>30 bars),
use wide targets/trailing stops. Skip MEDIUM regime.
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
    name = "cursor_opus46max_178"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 910     # 15:10
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("ar_window", default=60.0, low=40.0, high=90.0),
            TunableParam("mom_period", default=10.0, low=5.0, high=20.0),
            TunableParam("mom_thresh_bps", default=15.0, low=8.0, high=25.0),
            TunableParam("confirm_bars", default=3.0, low=2.0, high=5.0),
            TunableParam("fast_target_pct", default=0.0020, low=0.001, high=0.003),
            TunableParam("slow_target_pct", default=0.0050, low=0.003, high=0.007),
            TunableParam("fast_stop_pct", default=0.0012, low=0.0008, high=0.002),
            TunableParam("slow_stop_pct", default=0.0025, low=0.0015, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        ar_win = int(params.get("ar_window", 60.0))
        mom_period = int(params.get("mom_period", 10.0))
        mom_thresh = params.get("mom_thresh_bps", 15.0)
        confirm = int(params.get("confirm_bars", 3.0))
        fast_target = params.get("fast_target_pct", 0.0020)
        slow_target = params.get("slow_target_pct", 0.0050)
        fast_stop = params.get("fast_stop_pct", 0.0012)
        slow_stop = params.get("slow_stop_pct", 0.0025)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
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
                ret[i] = (close[i] - close[i-1]) / close[i-1]

        # Rolling AR(1) coefficient (phi)
        ar1_phi = np.zeros(n, dtype=np.float64)
        for i in range(ar_win, n):
            y = ret[i-ar_win+1:i+1]    # returns t
            x = ret[i-ar_win:i]          # returns t-1
            x_m = x - np.mean(x)
            y_m = y - np.mean(y)
            denom = np.sum(x_m ** 2)
            if denom > 1e-15:
                ar1_phi[i] = np.sum(x_m * y_m) / denom

        # Half-life from AR(1)
        half_life = np.full(n, 999.0, dtype=np.float64)
        for i in range(ar_win, n):
            phi = abs(ar1_phi[i])
            if 0.001 < phi < 1.0:
                half_life[i] = -np.log(2.0) / np.log(phi)

        # Regime: FAST_DECAY (<10), SLOW_DECAY (>30), MEDIUM (skip)
        is_fast = half_life < 10.0
        is_slow = half_life > 30.0
        tradeable = is_fast | is_slow

        # Momentum signal: ROC(close, mom_period) in bps
        mom_signal = np.zeros(n, dtype=np.float64)
        for i in range(mom_period, n):
            if close[i-mom_period] > 0:
                mom_signal[i] = (close[i] - close[i-mom_period]) / close[i-mom_period] * 10000

        # Confirmation: momentum remains positive/negative for `confirm` bars
        mom_pos_consec = np.zeros(n, dtype=np.int32)
        mom_neg_consec = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            mom_pos_consec[i] = (mom_pos_consec[i-1] + 1) if mom_signal[i] > 0 else 0
            mom_neg_consec[i] = (mom_neg_consec[i-1] + 1) if mom_signal[i] < 0 else 0

        time_ok = (time_mins >= 585) & (time_mins <= 910)

        long_entry = (
            (mom_signal > mom_thresh)
            & (close > vwap)
            & tradeable
            & (ar1_phi > 0)
            & (mom_pos_consec >= confirm)
            & time_ok
        )
        short_entry = (
            (mom_signal < -mom_thresh)
            & (close < vwap)
            & tradeable
            & (ar1_phi > 0)
            & (mom_neg_consec >= confirm)
            & time_ok
        )

        # Signal exit: AR(1) coeff turns negative
        sig_exit_long = ar1_phi < 0
        sig_exit_short = ar1_phi < 0

        atr = _compute_atr(high, low, close, 14)

        # Use slow regime parameters as defaults (more conservative)
        # The backtester uses fixed scalar exits; use average of fast/slow
        avg_target = (fast_target + slow_target) / 2.0
        avg_stop = (fast_stop + slow_stop) / 2.0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=avg_stop,
            target_pct=avg_target,
            trailing_activate_pct=0.0030,
            trailing_stop_pct=0.0015,
            time_stop_bars=90,
        )
