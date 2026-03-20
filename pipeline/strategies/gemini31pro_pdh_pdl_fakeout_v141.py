"""PDH/PDL Fakeout — cursor_gemini31pro_strategy_141

Thesis: Previous Day High/Low are obvious technical levels where retail traders
place stop losses. Institutional players drive price just beyond these levels
to trigger stops (liquidity hunt) before reversing.
Long: low crosses under PDL but close recovers above PDL.
Short: high crosses over PDH but close falls back below PDH.
Target: VWAP. Volume surge confirmation. VIX < 25.
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
    name = "gemini31pro_pdh_pdl_fakeout_v141"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 915     # 15:15
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_max", default=25.0, low=15.0, high=35.0),
            TunableParam("stop_atr_mult", default=1.5, low=0.8, high=3.0),
            TunableParam("breakeven_pct", default=0.005, low=0.002, high=0.01),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_max = params.get("vix_max", 25.0)
        stop_atr = params.get("stop_atr_mult", 1.5)
        be_pct = params.get("breakeven_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high_arr = df["high"].to_numpy().astype(np.float64)
        low_arr = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        day_id = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        atr14 = _compute_atr(high_arr, low_arr, close, 14)

        # Compute previous day high and low
        day_high_map = {}
        day_low_map = {}
        for i in range(n):
            d = day_id[i]
            if d not in day_high_map:
                day_high_map[d] = high_arr[i]
                day_low_map[d] = low_arr[i]
            else:
                day_high_map[d] = max(day_high_map[d], high_arr[i])
                day_low_map[d] = min(day_low_map[d], low_arr[i])

        sorted_days = sorted(set(day_id))
        day_to_pdh = {}
        day_to_pdl = {}
        for idx, d in enumerate(sorted_days):
            if idx == 0:
                day_to_pdh[d] = np.nan
                day_to_pdl[d] = np.nan
            else:
                prev_d = sorted_days[idx - 1]
                day_to_pdh[d] = day_high_map[prev_d]
                day_to_pdl[d] = day_low_map[prev_d]

        pdh = np.full(n, np.nan, dtype=np.float64)
        pdl = np.full(n, np.nan, dtype=np.float64)
        for i in range(n):
            pdh[i] = day_to_pdh.get(day_id[i], np.nan)
            pdl[i] = day_to_pdl.get(day_id[i], np.nan)
        pdh = np.nan_to_num(pdh, nan=high_arr[0])
        pdl = np.nan_to_num(pdl, nan=low_arr[0])

        # Fakeout detection
        # Long: low pierced PDL (low < pdl) but close recovered above PDL
        # Need previous bar's low to be >= pdl for the "cross under" aspect
        prev_low = np.roll(low_arr, 1)
        prev_low[0] = low_arr[0]
        long_fakeout = (prev_low >= pdl) & (low_arr < pdl) & (close > pdl)

        # Short: high pierced PDH (high > pdh) but close fell back below PDH
        prev_high = np.roll(high_arr, 1)
        prev_high[0] = high_arr[0]
        short_fakeout = (prev_high <= pdh) & (high_arr > pdh) & (close < pdh)

        # Volume surge
        vol_sma20 = np.full(n, 1.0, dtype=np.float64)
        for i in range(19, n):
            vol_sma20[i] = np.mean(volume[i - 19:i + 1])
        vol_sma20 = np.clip(vol_sma20, 1.0, None)
        vol_ok = volume > vol_sma20

        # VWAP for target
        df2 = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume")).cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns(
            (pl.col("_ctv") / pl.col("_cv")).alias("_vwap")
        )
        vwap = np.nan_to_num(df2["_vwap"].to_numpy().astype(np.float64), nan=close[0])

        vix_ok = vix < vix_max
        time_ok = (time_mins >= 570) & (time_mins <= 870)

        long_entry = long_fakeout & vol_ok & vix_ok & time_ok
        short_entry = short_fakeout & vol_ok & vix_ok & time_ok

        # Signal exit: price reaches VWAP — long exits above VWAP, short exits below
        signal_exit_long = close >= vwap
        signal_exit_short = close <= vwap

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr14,
            target_indicator=vwap,
            use_target_indicator=True,
            stop_loss_atr_mult=stop_atr,
            breakeven_pct=be_pct,
            time_stop_bars=60,
        )
