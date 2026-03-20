"""VWAP Volume Divergence v1 — cursor_opus46max_007

Thesis: When price moves away from VWAP but volume is declining
(price-volume divergence), the move lacks conviction and will reverse.
Enter when volume returns with a confirming bar.
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


def _linreg_slope(arr, period):
    n = len(arr)
    slope = np.zeros(n, dtype=np.float64)
    x = np.arange(period, dtype=np.float64)
    x_mean = x.mean()
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx < 1e-10:
        return slope
    for i in range(period - 1, n):
        window = arr[i - period + 1:i + 1]
        y_mean = np.mean(window)
        ss_xy = np.sum((x - x_mean) * (window - y_mean))
        slope[i] = ss_xy / ss_xx
    return slope


class Strategy(BaseStrategy):
    name = "cursor_opus46max_007"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 900     # 15:00
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_dev_thresh", default=0.5, low=0.3, high=1.0),
            TunableParam("vix_max", default=25.0, low=16.0, high=30.0),
            TunableParam("stop_atr_mult", default=1.5, low=0.8, high=2.5),
            TunableParam("breakeven_pct", default=0.003, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        dev_thresh = params.get("vwap_dev_thresh", 0.5)
        vix_max = params.get("vix_max", 25.0)
        stop_atr = params.get("stop_atr_mult", 1.5)
        be_pct = params.get("breakeven_pct", 0.003)

        n = len(df)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP deviation pct ──
        safe_vwap = np.where(vwap > 0, vwap, 1.0)
        vwap_dev_pct = (close - vwap) / safe_vwap * 100.0

        # ── Volume slope (linreg over 10 bars) ──
        vol_slope = _linreg_slope(volume, 10)

        # ── OBV approximation ──
        obv = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]
        obv_slope = _linreg_slope(obv, 10)

        # ── Close > low of last 10 bars / < high of last 10 bars ──
        lo10 = np.zeros(n, dtype=np.float64)
        hi10 = np.zeros(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - 9)
            lo10[i] = np.min(low[start:i + 1])
            hi10[i] = np.max(high[start:i + 1])

        # ── Confirmation: volume returning with price change ──
        vol_return_buy = np.zeros(n, dtype=np.bool_)
        vol_return_sell = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            vol_return_buy[i] = (volume[i] > volume[i - 1]) and (close[i] > close[i - 1])
            vol_return_sell[i] = (volume[i] > volume[i - 1]) and (close[i] < close[i - 1])

        # ── Volume floor ──
        avg_vol = df["volume"].rolling_mean(10).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_floor = volume > 0.5 * avg_vol

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Filters ──
        time_ok = (time_mins >= 570) & (time_mins <= 870)
        vix_ok = vix < vix_max

        # ── Entries ──
        long_entry = (
            (vwap_dev_pct < -dev_thresh)
            & (vol_slope < 0)
            & (close > lo10)
            & vol_return_buy
            & vol_floor & vix_ok & time_ok
        )
        short_entry = (
            (vwap_dev_pct > dev_thresh)
            & (vol_slope < 0)
            & (close < hi10)
            & vol_return_sell
            & vol_floor & vix_ok & time_ok
        )

        # ── Signal exit: volume surges 3x away from VWAP ──
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)
        sig_exit_long = (volume > 3.0 * avg_vol_20) & (close < close - 0.001) & (vwap_dev_pct < -dev_thresh * 1.5)
        sig_exit_short = (volume > 3.0 * avg_vol_20) & (vwap_dev_pct > dev_thresh * 1.5)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=vwap.copy(),
            stop_loss_atr_mult=stop_atr,
            use_target_indicator=True,
            breakeven_pct=be_pct,
            time_stop_bars=50,
        )
