"""Order Block / Fair Value Gap Entry — cursor_opus46max_144

Thesis: Order Blocks (last opposite-direction candle before impulse) and
Fair Value Gaps (3-bar pattern with gap) mark institutional supply/demand zones.
First revisit of these zones produces a bounce.
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
    name = "cursor_opus46max_144"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 915     # 15:15
    max_trades_per_day = 8
    assumptions = [
        "Order blocks identified retrospectively after impulse moves",
        "FVG zones tracked for first-touch revisit only",
    ]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("impulse_bps", default=10.0, low=5.0, high=20.0),
            TunableParam("zone_max_age", default=60.0, low=30.0, high=90.0),
            TunableParam("vol_thresh", default=1.3, low=1.0, high=2.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=28.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        impulse_bps = params.get("impulse_bps", 10.0)
        zone_max_age = int(params.get("zone_max_age", 60.0))
        vol_thresh = params.get("vol_thresh", 1.3)
        vix_max = params.get("vix_max", 22.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP for trend filter
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        atr = _compute_atr(high, low, close, 14)

        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)

        # Detect Order Blocks and FVGs, store zones
        # Bullish OB: bearish bar followed by bullish impulse
        # Bearish OB: bullish bar followed by bearish impulse
        # FVG: 3-bar gap pattern

        # Use lists to store active zones: (zone_lo, zone_hi, bar_idx, type, visited)
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)

        # Bullish zones (demand): zone_lo, zone_hi, created_bar
        bull_zones = []
        # Bearish zones (supply): zone_lo, zone_hi, created_bar
        bear_zones = []

        for i in range(3, n):
            # Detect bullish OB: bar[i-2] is bearish, then bars[i-1..i] impulse up
            if close[i-2] < open_[i-2]:  # bearish candle
                impulse = (close[i] - close[i-2]) / max(close[i-2], 1e-10) * 10000.0
                if impulse > impulse_bps and volume[i] > vol_thresh * avg_vol[i]:
                    bull_zones.append((low[i-2], high[i-2], i))

            # Detect bearish OB: bar[i-2] is bullish, then bars[i-1..i] impulse down
            if close[i-2] > open_[i-2]:  # bullish candle
                impulse = (close[i-2] - close[i]) / max(close[i-2], 1e-10) * 10000.0
                if impulse > impulse_bps and volume[i] > vol_thresh * avg_vol[i]:
                    bear_zones.append((low[i-2], high[i-2], i))

            # Detect FVGs
            if i >= 2:
                # Bullish FVG: gap up (bar[i-2].high < bar[i].low)
                if high[i-2] < low[i]:
                    bull_zones.append((high[i-2], low[i], i))
                # Bearish FVG: gap down (bar[i-2].low > bar[i].high)
                if low[i-2] > high[i]:
                    bear_zones.append((high[i], low[i-2], i))

            # Check zone revisits
            if vix[i] > vix_max or time_mins[i] < 565 or time_mins[i] > 900:
                continue

            # Check bullish zones (long entries)
            new_bull = []
            for (zlo, zhi, created) in bull_zones:
                age = i - created
                if age > zone_max_age:
                    continue  # expired
                if close[i] >= zlo and close[i] <= zhi and close[i] > vwap[i]:
                    long_entry[i] = True
                    continue  # consumed (first touch only)
                new_bull.append((zlo, zhi, created))
            bull_zones = new_bull

            # Check bearish zones (short entries)
            new_bear = []
            for (zlo, zhi, created) in bear_zones:
                age = i - created
                if age > zone_max_age:
                    continue
                if close[i] >= zlo and close[i] <= zhi and close[i] < vwap[i]:
                    short_entry[i] = True
                    continue
                new_bear.append((zlo, zhi, created))
            bear_zones = new_bear

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=1.0,
            target_pct=0.002,
            trailing_stop_pct=0.0005,
            trailing_activate_pct=0.001,
            time_stop_bars=20,
        )
