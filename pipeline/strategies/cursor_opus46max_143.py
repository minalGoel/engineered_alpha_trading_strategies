"""Market Profile POC Reversion — cursor_opus46max_143

Thesis: Price deviates beyond the Value Area then fails to sustain breakout;
it reverts to the Point of Control (highest volume price level). POC and VA
computed from volume-at-price distribution.
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
    name = "cursor_opus46max_143"
    is_long_only = False
    session_start = 600   # 10:00 (need 45 min for profile)
    session_end = 915     # 15:15
    max_trades_per_day = 8
    assumptions = [
        "Volume-at-price approximated by distributing bar volume across bar range",
    ]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("va_pct", default=70.0, low=60.0, high=80.0),
            TunableParam("poc_dist_min_bps", default=10.0, low=5.0, high=20.0),
            TunableParam("stop_beyond_va_bps", default=10.0, low=5.0, high=20.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=28.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        va_pct = params.get("va_pct", 70.0)
        poc_dist_min = params.get("poc_dist_min_bps", 10.0)
        stop_bps = params.get("stop_beyond_va_bps", 10.0)
        vix_max = params.get("vix_max", 22.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()

        atr = _compute_atr(high, low, close, 14)

        # Compute POC and VA per day using volume-at-price bins
        poc = np.zeros(n, dtype=np.float64)
        va_high = np.zeros(n, dtype=np.float64)
        va_low = np.zeros(n, dtype=np.float64)

        # Compute POC/VA once per day (end of day), then broadcast
        unique_days = np.unique(day_id)
        for d in unique_days:
            day_mask = np.where(day_id == d)[0]
            if len(day_mask) < 10:
                poc[day_mask] = close[day_mask[-1]]
                va_high[day_mask] = np.max(high[day_mask])
                va_low[day_mask] = np.min(low[day_mask])
                continue

            day_h = np.max(high[day_mask])
            day_l = np.min(low[day_mask])
            rng = day_h - day_l
            if rng < 1e-10:
                poc[day_mask] = close[day_mask[-1]]
                va_high[day_mask] = day_h
                va_low[day_mask] = day_l
                continue

            num_bins = 50
            bin_size = rng / num_bins
            vol_profile = np.zeros(num_bins, dtype=np.float64)

            for j in day_mask:
                bar_rng = high[j] - low[j]
                if bar_rng < 1e-10:
                    b_idx = min(int((close[j] - day_l) / bin_size), num_bins - 1)
                    vol_profile[b_idx] += volume[j]
                else:
                    lo_b = max(0, min(int((low[j] - day_l) / bin_size), num_bins - 1))
                    hi_b = max(0, min(int((high[j] - day_l) / bin_size), num_bins - 1))
                    cnt = hi_b - lo_b + 1
                    per = volume[j] / max(cnt, 1)
                    for b in range(lo_b, hi_b + 1):
                        vol_profile[b] += per

            poc_bin = int(np.argmax(vol_profile))
            poc_val = day_l + (poc_bin + 0.5) * bin_size

            total_vol = np.sum(vol_profile)
            if total_vol < 1e-10:
                poc[day_mask] = poc_val
                va_high[day_mask] = day_h
                va_low[day_mask] = day_l
                continue

            va_vol = vol_profile[poc_bin]
            lo_cursor = poc_bin - 1
            hi_cursor = poc_bin + 1
            target_vol = total_vol * va_pct / 100.0

            while va_vol < target_vol and (lo_cursor >= 0 or hi_cursor < num_bins):
                add_lo = vol_profile[lo_cursor] if lo_cursor >= 0 else 0
                add_hi = vol_profile[hi_cursor] if hi_cursor < num_bins else 0
                if add_hi >= add_lo:
                    va_vol += add_hi
                    hi_cursor += 1
                else:
                    va_vol += add_lo
                    lo_cursor -= 1

            va_l = day_l + max(lo_cursor + 1, 0) * bin_size
            va_h = day_l + min(hi_cursor, num_bins) * bin_size

            poc[day_mask] = poc_val
            va_high[day_mask] = va_h
            va_low[day_mask] = va_l

        # POC distance in bps
        safe_poc = np.clip(np.abs(poc), 1e-10, None)
        poc_dist = (close - poc) / safe_poc * 10000.0

        # VA breakout failed detection (was outside VA, now back inside)
        was_above_va = np.zeros(n, dtype=np.bool_)
        was_below_va = np.zeros(n, dtype=np.bool_)
        for i in range(5, n):
            for j in range(i-5, i):
                if close[j] > va_high[j]:
                    was_above_va[i] = True
                if close[j] < va_low[j]:
                    was_below_va[i] = True

        back_in_va = (close >= va_low) & (close <= va_high)

        vix_ok = vix < vix_max
        time_ok = (time_mins >= 600) & (time_mins <= 870)

        # Volume confirmation
        avg_vol = df["volume"].rolling_mean(10).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        # Long: was below VA, came back, still below POC
        long_entry = was_below_va & back_in_va & (poc_dist < -poc_dist_min) & vix_ok & time_ok & vol_ok
        # Short: was above VA, came back, still above POC
        short_entry = was_above_va & back_in_va & (poc_dist > poc_dist_min) & vix_ok & time_ok & vol_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr,
            target_indicator=poc.copy(),
            use_target_indicator=True,
            stop_loss_pct=stop_bps / 10000.0,
            trailing_stop_pct=0.0005,
            trailing_activate_pct=0.001,
            time_stop_bars=30,
        )
