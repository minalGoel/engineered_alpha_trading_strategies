"""Index Rebalance Flow — cursor_opus46max_127

Thesis: Stocks added/removed from NIFTY indices experience predictable
passive fund flow. Proxied by: stocks trading above VWAP with elevated volume
(additions) or below VWAP with elevated volume (removals). Since we lack
announcement data, we detect flow via volume surge + VWAP direction.
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
    name = "cursor_opus46max_127"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 915     # 15:15
    max_trades_per_day = 3
    assumptions = [
        "No index rebalance announcement data; proxied by volume surge + VWAP trend",
    ]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_mult", default=1.3, low=1.1, high=2.0),
            TunableParam("vwap_offset_bps", default=5.0, low=2.0, high=15.0),
            TunableParam("target_bps", default=30.0, low=15.0, high=50.0),
            TunableParam("stop_bps", default=20.0, low=10.0, high=35.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_mult = params.get("vol_mult", 1.3)
        vwap_off = params.get("vwap_offset_bps", 5.0)
        target_bps = params.get("target_bps", 30.0)
        stop_bps = params.get("stop_bps", 20.0)
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

        atr = _compute_atr(high, low, close, 20)

        # Rolling average volume
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)

        vol_ok = volume > vol_mult * avg_vol
        vix_ok = vix < vix_max
        time_ok = (time_mins >= 570) & (time_mins <= 840)  # no entry after 14:00

        # Close vs VWAP in bps
        safe_vwap = np.clip(np.abs(vwap), 1e-10, None)
        vwap_dev = (close - vwap) / safe_vwap * 10000.0

        # Long: close above VWAP + offset, volume surge (passive buying)
        long_entry = (vwap_dev > vwap_off) & vol_ok & vix_ok & time_ok
        # Short: close below VWAP - offset, volume surge (passive selling)
        short_entry = (vwap_dev < -vwap_off) & vol_ok & vix_ok & time_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_bps / 10000.0,
            target_pct=target_bps / 10000.0,
            trailing_stop_pct=0.0015,
            trailing_activate_pct=0.002,
            time_stop_bars=180,
        )
