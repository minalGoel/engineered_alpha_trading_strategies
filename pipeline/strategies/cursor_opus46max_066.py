"""Gap to VWAP v1 — cursor_opus46max_066

Thesis: After gap, price reverts to developing VWAP as institutional algos
pull price back. Enter when VWAP deviation >20bps with RSI confirmation.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        diff = close[i] - close[i-1]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff
    avg_gain = np.mean(gains[1:period+1])
    avg_loss = np.mean(losses[1:period+1])
    if avg_loss > 1e-10:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    else:
        rsi[period] = 100.0
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss > 1e-10:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        else:
            rsi[i] = 100.0
    return rsi


def _ema(data, period):
    n = len(data)
    out = np.zeros(n, dtype=np.float64)
    out[0] = data[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, n):
        out[i] = alpha * data[i] + (1 - alpha) * out[i-1]
    return out


class Strategy(BaseStrategy):
    name = "cursor_opus46max_066"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 870     # 14:30
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_dev_thresh_bps", default=20.0, low=10.0, high=40.0),
            TunableParam("rsi_low", default=35.0, low=20.0, high=45.0),
            TunableParam("rsi_high", default=65.0, low=55.0, high=80.0),
            TunableParam("vix_max", default=20.0, low=14.0, high=26.0),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.007),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vwap_thresh = params.get("vwap_dev_thresh_bps", 20.0)
        rsi_low = params.get("rsi_low", 35.0)
        rsi_high = params.get("rsi_high", 65.0)
        vix_max = params.get("vix_max", 20.0)
        stop_pct = params.get("stop_loss_pct", 0.004)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df2 = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume")).cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns(
            (pl.col("_ctv") / pl.col("_cv")).alias("_vwap")
        )
        vwap = df2["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # VWAP deviation in bps
        vwap_dev = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if vwap[i] > 1e-8:
                vwap_dev[i] = (close[i] - vwap[i]) / vwap[i] * 10000.0

        rsi10 = _compute_rsi(close, 10)
        ema5 = _ema(close, 5)

        vix_ok = vix < vix_max
        time_ok = (time_mins >= 565) & (time_mins <= 660)  # 09:25-11:00

        # EMA5 cross from below (long) or above (short) while deviated from VWAP
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if (vwap_dev[i] < -vwap_thresh and rsi10[i] < rsi_low
                    and vix_ok[i] and time_ok[i]
                    and close[i-1] < ema5[i-1] and close[i] >= ema5[i]
                    and close[i] < vwap[i]):
                long_entry[i] = True
            if (vwap_dev[i] > vwap_thresh and rsi10[i] > rsi_high
                    and vix_ok[i] and time_ok[i]
                    and close[i-1] > ema5[i-1] and close[i] <= ema5[i]
                    and close[i] > vwap[i]):
                short_entry[i] = True

        # Use VWAP as target indicator
        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=vwap,
            use_target_indicator=True,
            stop_loss_pct=stop_pct,
            time_stop_bars=60,
        )
