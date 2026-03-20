"""Volume Profile Breakout v1 — claude_project_7_of_25

Thesis: Price breaking above the value area high (where 70% of volume
traded) with strong volume signals institutional conviction.
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
    name = "volume_profile_breakout_v1"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 870     # 14:30
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_ratio_thresh", default=1.5, low=1.0, high=3.0),
            TunableParam("target_pct", default=0.008, low=0.003, high=0.015),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.008),
            TunableParam("trail_atr_mult", default=1.0, low=0.5, high=2.5),
            TunableParam("num_bins", default=20.0, low=10.0, high=40.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_ratio = params.get("vol_ratio_thresh", 1.5)
        target_pct = params.get("target_pct", 0.008)
        stop_pct = params.get("stop_loss_pct", 0.004)
        trail_atr = params.get("trail_atr_mult", 1.0)
        num_bins = int(params.get("num_bins", 20.0))

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        day_id = df["day_id"].to_numpy()

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        atr = _compute_atr(high, low, close, 14)

        # ── Volume average ──
        avg_vol = df["volume"].rolling_mean(30).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)

        # ── Compute volume profile (POC, VAH, VAL) per day ──
        # Simplified: accumulate volume in price bins for each day so far
        poc = np.zeros(n, dtype=np.float64)
        vah = np.zeros(n, dtype=np.float64)
        val_arr = np.zeros(n, dtype=np.float64)
        profile_valid = np.zeros(n, dtype=np.bool_)

        cur_day = -1
        day_start = 0
        for i in range(n):
            if day_id[i] != cur_day:
                cur_day = day_id[i]
                day_start = i
            # Need at least 10 bars to build profile
            if i - day_start < 10:
                continue
            day_close = close[day_start:i + 1]
            day_vol = volume[day_start:i + 1]
            day_high = high[day_start:i + 1]
            day_low = low[day_start:i + 1]
            price_min = np.min(day_low)
            price_max = np.max(day_high)
            if price_max <= price_min:
                continue
            bin_edges = np.linspace(price_min, price_max, num_bins + 1)
            bin_vol = np.zeros(num_bins, dtype=np.float64)
            for j in range(len(day_close)):
                typical = (day_high[j] + day_low[j] + day_close[j]) / 3.0
                b = int((typical - price_min) / (price_max - price_min) * (num_bins - 1))
                b = min(max(b, 0), num_bins - 1)
                bin_vol[b] += day_vol[j]
            # POC: bin with max volume
            poc_bin = np.argmax(bin_vol)
            poc[i] = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2.0
            # Value area: 70% of total volume, expanding from POC
            total_vol = np.sum(bin_vol)
            if total_vol < 1:
                continue
            va_vol = bin_vol[poc_bin]
            lo_bin = poc_bin
            hi_bin = poc_bin
            while va_vol < 0.70 * total_vol:
                add_lo = bin_vol[lo_bin - 1] if lo_bin > 0 else 0
                add_hi = bin_vol[hi_bin + 1] if hi_bin < num_bins - 1 else 0
                if add_lo >= add_hi and lo_bin > 0:
                    lo_bin -= 1
                    va_vol += add_lo
                elif hi_bin < num_bins - 1:
                    hi_bin += 1
                    va_vol += add_hi
                else:
                    break
            vah[i] = bin_edges[min(hi_bin + 1, num_bins)]
            val_arr[i] = bin_edges[lo_bin]
            profile_valid[i] = True

        # ── Entry conditions ──
        vol_ok = volume > vol_ratio * avg_vol
        # AUDIT FIX: missing session time filter caused entries outside session window
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)
        long_entry = profile_valid & (close > vah) & vol_ok & (close > vwap) & time_ok
        short_entry = profile_valid & (close < val_arr) & vol_ok & (close < vwap) & time_ok

        # Trailing stop in pct from ATR
        median_atr = np.median(atr[atr > 0]) if np.any(atr > 0) else 0.0
        median_close = np.median(close[close > 0]) if np.any(close > 0) else 1.0
        trailing_pct = trail_atr * median_atr / median_close if median_close > 0 else 0.005

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_stop_pct=trailing_pct,
            time_stop_bars=120,
        )
