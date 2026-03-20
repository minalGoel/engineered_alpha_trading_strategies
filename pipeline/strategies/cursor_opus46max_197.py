"""Rights Issue Arb v1 — cursor_opus46max_197

Thesis: On rights issue ex-date, actual price adjustment is often incomplete.
Overshoot (drop more than TERP) = buy. Undershoot = sell. Adapted: detect
large overnight gaps (proxy for corporate action ex-dates) and trade the
reversion toward the prior session's adjusted value.
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
    name = "cursor_opus46max_197"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 915     # 15:15
    max_trades_per_day = 1

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_thresh_bps", default=30.0, low=15.0, high=60.0),
            TunableParam("wait_bars", default=15.0, low=5.0, high=30.0),
            TunableParam("vol_surge_mult", default=1.5, low=1.0, high=3.0),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.006),
            TunableParam("target_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("trailing_activate_pct", default=0.002, low=0.001, high=0.003),
            TunableParam("trailing_stop_pct", default=0.0015, low=0.001, high=0.003),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_th = params.get("gap_thresh_bps", 30.0)
        wait = int(params.get("wait_bars", 15.0))
        vol_mult = params.get("vol_surge_mult", 1.5)
        stop_pct = params.get("stop_loss_pct", 0.004)
        target_pct = params.get("target_pct", 0.003)
        trail_act = params.get("trailing_activate_pct", 0.002)
        trail_pct = params.get("trailing_stop_pct", 0.0015)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # Detect gap at open
        gap_bps = np.zeros(n, dtype=np.float64)
        day_open_idx = np.zeros(n, dtype=np.int64)
        bars_since_open = np.zeros(n, dtype=np.int32)

        for i in range(n):
            if i == 0 or day_id[i] != day_id[i-1]:
                day_open_idx[i] = i
                bars_since_open[i] = 0
                if i > 0 and close[i-1] > 0:
                    gap_bps[i] = (opn[i] - close[i-1]) / close[i-1] * 10000
            else:
                day_open_idx[i] = day_open_idx[i-1]
                bars_since_open[i] = bars_since_open[i-1] + 1
                gap_bps[i] = gap_bps[int(day_open_idx[i])]

        # Volume surge
        avg_vol = np.zeros(n, dtype=np.float64)
        for i in range(19, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > (vol_mult * avg_vol)

        # After wait bars, check recovery direction
        ema10 = _compute_ema(close, 10)

        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)

        for i in range(n):
            bso = bars_since_open[i]
            if bso < wait or bso > 60:
                continue
            g = gap_bps[i]
            # Overshoot down: gap < -thresh, price recovering
            if g < -gap_th and close[i] > vwap[i] and close[i] > ema10[i] and vol_ok[i]:
                long_entry[i] = True
            # Undershoot: gap > thresh, price fading
            elif g > gap_th and close[i] < vwap[i] and close[i] < ema10[i] and vol_ok[i]:
                short_entry[i] = True

        # Signal exit: VWAP slope turns against position
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(10, n):
            vwap_slope = vwap[i] - vwap[i-10]
            if vwap_slope < 0:
                sig_exit_long[i] = True
            if vwap_slope > 0:
                sig_exit_short[i] = True

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
            trailing_activate_pct=trail_act,
            trailing_stop_pct=trail_pct,
            time_stop_bars=180,
        )
