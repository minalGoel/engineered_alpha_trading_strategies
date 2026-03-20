"""Fractal Breakout with Alligator — cursor_opus46max_146

Thesis: Williams Fractals (5-bar swing highs/lows) provide noise-filtered
breakout levels. Combined with Alligator indicator (3 smoothed moving averages)
to confirm trending conditions.
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


def _sma(arr, period):
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    if n < period:
        return out
    cumsum = np.cumsum(arr)
    out[period-1] = cumsum[period-1] / period
    for i in range(period, n):
        out[i] = (cumsum[i] - cumsum[i - period]) / period
    return out


class Strategy(BaseStrategy):
    name = "cursor_opus46max_146"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 915     # 15:15
    max_trades_per_day = 12

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("fractal_min_dist_bps", default=5.0, low=2.0, high=12.0),
            TunableParam("fractal_max_age", default=20.0, low=10.0, high=30.0),
            TunableParam("vix_max", default=25.0, low=15.0, high=30.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        frac_dist = params.get("fractal_min_dist_bps", 5.0)
        frac_age = int(params.get("fractal_max_age", 20.0))
        vix_max = params.get("vix_max", 25.0)

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

        atr = _compute_atr(high, low, close, 14)

        # Median price
        median_price = (high + low) / 2

        # Alligator: shifted SMAs of median price
        jaw_raw = _sma(median_price, 13)
        teeth_raw = _sma(median_price, 8)
        lips_raw = _sma(median_price, 5)

        # Apply shifts (jaw +8, teeth +5, lips +3)
        jaw = np.zeros(n, dtype=np.float64)
        teeth = np.zeros(n, dtype=np.float64)
        lips = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i >= 8:
                jaw[i] = jaw_raw[i - 8]
            if i >= 5:
                teeth[i] = teeth_raw[i - 5]
            if i >= 3:
                lips[i] = lips_raw[i - 3]

        # Alligator trending
        alligator_up = (lips > teeth) & (teeth > jaw) & (jaw > 0)
        alligator_dn = (lips < teeth) & (teeth < jaw) & (jaw > 0)

        # Williams Fractals (confirmed 2 bars later)
        fractal_up = np.zeros(n, dtype=np.float64)  # swing high level
        fractal_dn = np.zeros(n, dtype=np.float64)  # swing low level
        last_frac_up = 0.0
        last_frac_up_bar = -100
        last_frac_dn = 0.0
        last_frac_dn_bar = -100

        for i in range(4, n):
            # Up fractal at i-2
            mid = i - 2
            if (high[mid] > high[mid-1] and high[mid] > high[mid-2] and
                    high[mid] > high[mid+1] and high[mid] > high[mid+2]):
                last_frac_up = high[mid]
                last_frac_up_bar = i
            # Down fractal at i-2
            if (low[mid] < low[mid-1] and low[mid] < low[mid-2] and
                    low[mid] < low[mid+1] and low[mid] < low[mid+2]):
                last_frac_dn = low[mid]
                last_frac_dn_bar = i

            fractal_up[i] = last_frac_up
            fractal_dn[i] = last_frac_dn

        # Volume filter
        avg_vol = df["volume"].rolling_mean(10).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        vix_ok = (vix >= 12) & (vix <= vix_max)
        time_ok = (time_mins >= 565) & (time_mins <= 900)

        # Fractal distance from current price
        safe_close = np.clip(close, 1e-10, None)
        up_dist_bps = (fractal_up - close) / safe_close * 10000.0
        dn_dist_bps = (close - fractal_dn) / safe_close * 10000.0

        # Long: breakout above up fractal, alligator trending up, above VWAP
        # Sustained breakout: close > fractal_up for 2 bars
        long_break = np.zeros(n, dtype=np.bool_)
        short_break = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if close[i] > fractal_up[i] and close[i-1] > fractal_up[i-1] and fractal_up[i] > 0:
                long_break[i] = True
            if close[i] < fractal_dn[i] and close[i-1] < fractal_dn[i-1] and fractal_dn[i] > 0:
                short_break[i] = True

        long_entry = long_break & alligator_up & (close > vwap) & vol_ok & vix_ok & time_ok
        short_entry = short_break & alligator_dn & (close < vwap) & vol_ok & vix_ok & time_ok

        # Signal exit: close crosses teeth against position or opposite fractal forms
        sig_exit_long = close < teeth
        sig_exit_short = close > teeth

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=0.002,
            stop_loss_pct=0.0015,
            time_stop_bars=15,
        )
