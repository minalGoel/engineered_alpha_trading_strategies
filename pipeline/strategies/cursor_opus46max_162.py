"""Adaptive Threshold v1 — cursor_opus46max_162

Thesis: Dynamic RSI thresholds using rolling 120-bar percentiles instead
of static 30/70. Combined with BB squeeze detection and VWAP distance.
Late session start (11:15 = 675 min) for 120-bar percentile calibration.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain[i] / avg_loss[i])
    return rsi


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
    name = "cursor_opus46max_162"
    is_long_only = False
    session_start = 675   # 11:15
    session_end = 910     # 15:10
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_upper_pctile", default=95.0, low=85.0, high=99.0),
            TunableParam("rsi_lower_pctile", default=5.0, low=1.0, high=15.0),
            TunableParam("vwap_dist_bps", default=20.0, low=10.0, high=40.0),
            TunableParam("stop_loss_pct", default=0.0035, low=0.002, high=0.005),
            TunableParam("trailing_stop_pct", default=0.0015, low=0.001, high=0.003),
            TunableParam("trailing_activate_pct", default=0.0025, low=0.0015, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        upper_pctile = params.get("rsi_upper_pctile", 95.0)
        lower_pctile = params.get("rsi_lower_pctile", 5.0)
        vwap_dist_bps = params.get("vwap_dist_bps", 20.0)
        stop_pct = params.get("stop_loss_pct", 0.0035)
        trail_pct = params.get("trailing_stop_pct", 0.0015)
        trail_act = params.get("trailing_activate_pct", 0.0025)

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

        # RSI
        rsi = _compute_rsi(close, 14)

        # Rolling percentile thresholds for RSI (120-bar window)
        rsi_upper_thresh = np.full(n, 70.0, dtype=np.float64)
        rsi_lower_thresh = np.full(n, 30.0, dtype=np.float64)
        for i in range(119, n):
            window = rsi[i - 119:i + 1]
            rsi_upper_thresh[i] = np.percentile(window, upper_pctile)
            rsi_lower_thresh[i] = np.percentile(window, lower_pctile)

        # BB Width for squeeze detection
        sma20 = df["close"].rolling_mean(20).to_numpy().astype(np.float64)
        sma20 = np.nan_to_num(sma20, nan=0.0)
        std20 = df["close"].rolling_std(20).to_numpy().astype(np.float64)
        std20 = np.nan_to_num(std20, nan=1e10)
        safe_sma = np.clip(np.abs(sma20), 1e-10, None)
        bb_width = 4.0 * std20 / safe_sma

        # BB width threshold (20th percentile over 120 bars)
        bb_width_thresh = np.full(n, 0.01, dtype=np.float64)
        for i in range(119, n):
            window = bb_width[i - 119:i + 1]
            bb_width_thresh[i] = np.percentile(window, 20.0)

        # Recent squeeze: BB width was below threshold within last 10 bars
        recent_squeeze = np.zeros(n, dtype=np.bool_)
        for i in range(10, n):
            for j in range(i - 10, i):
                if bb_width[j] < bb_width_thresh[j]:
                    recent_squeeze[i] = True
                    break

        # VWAP distance
        safe_vwap = np.where(vwap > 0, vwap, 1.0)
        vwap_dist = (close - vwap) / safe_vwap * 10000.0

        # RSI turning
        rsi_turning_up = np.zeros(n, dtype=np.bool_)
        rsi_turning_dn = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if rsi[i] > rsi[i - 1]:
                rsi_turning_up[i] = True
            if rsi[i] < rsi[i - 1]:
                rsi_turning_dn[i] = True

        # Volume filter
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        time_ok = (time_mins >= 675) & (time_mins <= 910)

        # Long: RSI below dynamic lower threshold + recent squeeze + below VWAP + turning up
        long_entry = (rsi < rsi_lower_thresh) & recent_squeeze & \
                     (vwap_dist < -vwap_dist_bps) & rsi_turning_up & vol_ok & time_ok

        # Short: RSI above dynamic upper threshold + recent squeeze + above VWAP + turning down
        short_entry = (rsi > rsi_upper_thresh) & recent_squeeze & \
                      (vwap_dist > vwap_dist_bps) & rsi_turning_dn & vol_ok & time_ok

        # Signal exit: target is VWAP touch
        # Also signal exit when RSI crosses median of distribution
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(119, n):
            rsi_median = np.percentile(rsi[i - 119:i + 1], 50.0)
            if rsi[i] > rsi_median:
                sig_exit_long[i] = True
            if rsi[i] < rsi_median:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=vwap.copy(),
            stop_loss_pct=stop_pct,
            use_target_indicator=True,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=60,
        )
