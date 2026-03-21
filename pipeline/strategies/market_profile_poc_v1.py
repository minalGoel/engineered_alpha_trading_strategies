"""Market Profile POC Reversion (5s NIFTY Options)

When NIFTY probes outside the developing session Value Area (the price range containing
70% of cumulative session volume) but retreats back inside within 30 seconds, buy
ATM options to trade the reversion toward the Point of Control (highest volume node).

Market Profile originated as a CME futures/index theory — NIFTY, driven by institutional
VWAP/TWAP algorithms benchmarked to volume-weighted price, is the ideal environment
for POC gravity at 5-second resolution.
"""

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


def _compute_market_profile(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    time_min: np.ndarray,
    day_id: np.ndarray,
    session_start: int = 555,
    bin_size: float = 5.0,
    va_pct: float = 0.70,
    min_bars_for_profile: int = 60,
) -> tuple:
    """Incrementally compute developing POC and Value Area for each 5-second bar.

    Profile accumulates from session_start (usually 09:15 = 555 min). Resets per day.
    POC and VA are only returned after min_bars_for_profile bars have accumulated.

    Returns:
        poc:         ndarray — price level with highest cumulative volume
        va_high_arr: ndarray — upper Value Area boundary (covers va_pct of session vol)
        va_low_arr:  ndarray — lower Value Area boundary
    """
    n = len(close)
    poc = close.copy()
    va_high_arr = close.copy()
    va_low_arr = close.copy()

    for day in np.unique(day_id):
        day_mask = day_id == day
        day_indices = np.where(day_mask)[0]
        if len(day_indices) == 0:
            continue

        # Use full day's range to size bins (backtest context — all bars available)
        day_lo = np.nanmin(low[day_indices])
        day_hi = np.nanmax(high[day_indices])
        if np.isnan(day_lo) or np.isnan(day_hi) or day_hi <= day_lo:
            continue

        margin = bin_size * 3
        bin_min = day_lo - margin
        bin_max = day_hi + margin
        nbins = max(int((bin_max - bin_min) / bin_size) + 1, 10)
        bins = np.linspace(bin_min, bin_max, nbins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        vol_at_price = np.zeros(nbins)
        bars_in_session = 0

        for idx in day_indices:
            t = int(time_min[idx])
            if t < session_start:
                # Before session open — profile not yet started
                poc[idx] = close[idx]
                va_high_arr[idx] = close[idx]
                va_low_arr[idx] = close[idx]
                continue

            # Distribute this bar's volume uniformly across its high-low range
            lo_p = low[idx] if not np.isnan(low[idx]) else close[idx]
            hi_p = high[idx] if not np.isnan(high[idx]) else close[idx]
            vol_p = volume[idx] if not np.isnan(volume[idx]) else 0.0

            if hi_p > lo_p and vol_p > 0.0:
                lo_bin = max(0, int((lo_p - bin_min) / bin_size))
                hi_bin = min(nbins - 1, int((hi_p - bin_min) / bin_size) + 1)
                if hi_bin > lo_bin:
                    vol_at_price[lo_bin:hi_bin] += vol_p / (hi_bin - lo_bin)

            bars_in_session += 1

            # Need minimum profile depth before generating signals
            if bars_in_session < min_bars_for_profile:
                poc[idx] = close[idx]
                va_high_arr[idx] = close[idx]
                va_low_arr[idx] = close[idx]
                continue

            # POC = bin with highest cumulative volume
            poc_bin = int(np.argmax(vol_at_price))
            poc[idx] = bin_centers[poc_bin]

            # Value Area: expand outward from POC until va_pct of total volume covered
            total_vol = float(np.sum(vol_at_price))
            if total_vol <= 0.0:
                va_high_arr[idx] = close[idx]
                va_low_arr[idx] = close[idx]
                continue

            target_vol = va_pct * total_vol
            current_vol = float(vol_at_price[poc_bin])
            lo_ptr = poc_bin - 1
            hi_ptr = poc_bin + 1

            while current_vol < target_vol:
                lo_val = float(vol_at_price[lo_ptr]) if lo_ptr >= 0 else -1.0
                hi_val = float(vol_at_price[hi_ptr]) if hi_ptr < nbins else -1.0

                if lo_val < 0.0 and hi_val < 0.0:
                    break

                if lo_val >= hi_val:
                    current_vol += lo_val
                    lo_ptr -= 1
                else:
                    current_vol += hi_val
                    hi_ptr += 1

            # lo_ptr+1 is the lowest included bin; hi_ptr-1 is the highest included bin
            # VA lower fence = bins[lo_ptr + 1]; VA upper fence = bins[hi_ptr]
            va_low_arr[idx] = bins[lo_ptr + 1] if (lo_ptr + 1) < len(bins) else bin_min
            va_high_arr[idx] = bins[hi_ptr] if hi_ptr < len(bins) else bin_max

    return poc, va_high_arr, va_low_arr


class Strategy(BaseStrategy):
    name = "market_profile_poc_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — 15 min after open for profile development
    session_end_minutes = 920     # 15:20 IST — no new entries in last 10 min
    max_trades_per_day = 6
    max_lookback = 180            # 15 min warmup (180 × 5s = 900s)

    def tunable_params(self) -> list:
        return [
            # Minimum bps distance from POC before entry (ensures room for reversion)
            TunableParam("poc_distance_bps", 8.0, 3.0, 20.0),
            # VIX threshold above which POC gravity breaks down (trend days)
            TunableParam("vix_threshold", 20.0, 14.0, 26.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Extract spot arrays — forward-fill NaN before converting
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(float)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(float)
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        poc_distance_bps = params.get("poc_distance_bps", 8.0)
        vix_threshold = params.get("vix_threshold", 20.0)

        # Compute developing Market Profile (accumulates from 09:15 = 555 min)
        poc, va_high, va_low = _compute_market_profile(
            close, high, low, volume, time_min, day_id,
            session_start=555,          # accumulate from market open
            bin_size=5.0,               # 5-point NIFTY bins
            va_pct=0.70,                # Value Area covers 70% of session volume
            min_bars_for_profile=60,    # minimum 5 min of profile (60 × 5s)
        )

        # VIX filter — aligned to spot bars via backward join
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # Probe detection: rolling 6-bar (30s) min/max of close
        # Detects if price probed outside VA within the last 30 seconds
        probe_bars = 6
        roll_min = close.copy()   # default: no probe detected
        roll_max = close.copy()
        for i in range(probe_bars, n):
            roll_min[i] = float(np.min(close[i - probe_bars:i]))
            roll_max[i] = float(np.max(close[i - probe_bars:i]))

        # POC distance in bps — ensures sufficient room for reversion
        poc_dist_bps = np.abs(close - poc) / (poc + 1e-10) * 10000.0

        # VA width validation — degenerate profile (VA collapsed to a single point) filtered out
        va_width_bps = (va_high - va_low) / (poc + 1e-10) * 10000.0
        va_valid = va_width_bps > 10.0

        # Session filter
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Bullish entry: price probed below VA_low then re-entered ──
        # POC gravity pulls price upward from below-POC position inside VA
        probed_below = roll_min < va_low          # had a close below VA_low recently
        back_in_va_bull = close > va_low           # now back inside VA (failed breakdown)
        below_poc = close < poc                    # below POC — directional pull is upward

        buy_ce = (
            in_session
            & va_valid
            & probed_below
            & back_in_va_bull
            & below_poc
            & (poc_dist_bps > poc_distance_bps)
            & (vix_close < vix_threshold)
        )

        # ── Bearish entry: price probed above VA_high then re-entered ──
        # POC gravity pulls price downward from above-POC position inside VA
        probed_above = roll_max > va_high          # had a close above VA_high recently
        back_in_va_bear = close < va_high          # now back inside VA (failed breakout)
        above_poc = close > poc                    # above POC — directional pull is downward

        buy_pe = (
            in_session
            & va_valid
            & probed_above
            & back_in_va_bear
            & above_poc
            & (poc_dist_bps > poc_distance_bps)
            & (vix_close < vix_threshold)
        )

        # Resolve simultaneous signals (shouldn't happen, but be safe)
        both = buy_ce & buy_pe
        buy_ce = buy_ce & ~both
        buy_pe = buy_pe & ~both

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # Stop: 4 pts = ~8 NIFTY spot pts; if price extends further from POC, thesis failed
            # Target: 7 pts = ~14 NIFTY spot pts; captures 50-60% of expected POC reversion (1:1.75 R:R)
            stop_points=np.full(n, 4.0),
            target_points=np.full(n, 7.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,              # 90 seconds — POC reversion must manifest quickly
            max_trades_per_day=self.max_trades_per_day,
        )
