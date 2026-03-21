"""Order Block + Fair Value Gap entry strategy for NIFTY 5-second options.

Mechanism: Strong 15-second NIFTY impulses (3 × 5s bars, >= impulse_pts spot pts)
reveal Order Blocks — the last opposite-direction candle before the impulse — where
residual institutional resting orders remain. When NIFTY retraces to these zones within
6 minutes, resting orders re-engage and produce fresh 15-25 spot pt continuation.
Fair Value Gaps (3-bar price discontinuities) capture the same dynamic at sub-minute
scale. Trades are only taken in the direction of VWAP trend.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


def _compute_vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Session-cumulative VWAP, reset each trading day."""
    n = len(close)
    vwap = np.empty(n)
    tp = (high + low + close) / 3.0
    cum_tpv = 0.0
    cum_v = 0.0
    prev_day = -1
    for i in range(n):
        if day_id[i] != prev_day:
            cum_tpv = 0.0
            cum_v = 0.0
            prev_day = day_id[i]
        cum_tpv += tp[i] * volume[i]
        cum_v += volume[i]
        vwap[i] = cum_tpv / cum_v if cum_v > 0 else tp[i]
    return vwap


def _compute_signals(
    open_a: np.ndarray,
    high_a: np.ndarray,
    low_a: np.ndarray,
    close_a: np.ndarray,
    vwap: np.ndarray,
    time_min: np.ndarray,
    impulse_pts: float,
    max_zone_age: int,
    session_start: int,
    session_end: int,
) -> tuple:
    """Detect OB/FVG zones and signal on first-touch revisit.

    Returns (buy_ce, buy_pe) boolean arrays of length n.
    """
    n = len(close_a)
    buy_ce = np.zeros(n, dtype=bool)
    buy_pe = np.zeros(n, dtype=bool)

    # Circular zone buffer
    MAX_ZONES = 80
    z_lo = np.zeros(MAX_ZONES)
    z_hi = np.zeros(MAX_ZONES)
    z_dir = np.zeros(MAX_ZONES, dtype=np.int8)    # +1 bullish, -1 bearish
    z_born = np.full(MAX_ZONES, -9999, dtype=np.int32)
    z_used = np.zeros(MAX_ZONES, dtype=bool)
    zp = 0  # next write index (wraps via zp % MAX_ZONES)

    OB_BARS = 3  # 15-second impulse window for Order Block detection

    for i in range(OB_BARS + 2, n):
        t = time_min[i]
        in_sess = session_start <= t < session_end

        # ── 1. Detect Fair Value Gaps (3-bar price discontinuity at bar i) ──
        if in_sess and i >= 2:
            # Bullish FVG: high[i-2] < low[i] — ascending price gap
            if high_a[i - 2] < low_a[i]:
                lo = high_a[i - 2]
                hi = low_a[i]
                if hi > lo:
                    slot = zp % MAX_ZONES
                    z_lo[slot] = lo
                    z_hi[slot] = hi
                    z_dir[slot] = 1
                    z_born[slot] = i
                    z_used[slot] = False
                    zp += 1

            # Bearish FVG: low[i-2] > high[i] — descending price gap
            if low_a[i - 2] > high_a[i]:
                lo = high_a[i]
                hi = low_a[i - 2]
                if hi > lo:
                    slot = zp % MAX_ZONES
                    z_lo[slot] = lo
                    z_hi[slot] = hi
                    z_dir[slot] = -1
                    z_born[slot] = i
                    z_used[slot] = False
                    zp += 1

        # ── 2. Detect Order Blocks (last opposite candle before 15s impulse) ──
        if in_sess and i >= OB_BARS:
            net = close_a[i] - close_a[i - OB_BARS]
            if abs(net) >= impulse_pts:
                d = 1 if net > 0 else -1
                ob_lo = 0.0
                ob_hi = 0.0
                found = False
                # Scan backward: find last opposite candle in impulse window
                for k in range(i - 1, i - OB_BARS - 1, -1):
                    is_bull = close_a[k] > open_a[k]
                    if d == 1 and not is_bull:    # bearish candle before bullish impulse
                        ob_lo = low_a[k]
                        ob_hi = high_a[k]
                        found = True
                        break
                    elif d == -1 and is_bull:     # bullish candle before bearish impulse
                        ob_lo = low_a[k]
                        ob_hi = high_a[k]
                        found = True
                        break

                if found and ob_hi > ob_lo:
                    # Dedup: skip if nearly identical active zone already exists
                    dup = False
                    for s in range(MAX_ZONES):
                        if (not z_used[s] and z_born[s] > 0 and z_dir[s] == d
                                and abs(z_lo[s] - ob_lo) < 3.0
                                and abs(z_hi[s] - ob_hi) < 3.0):
                            dup = True
                            break
                    if not dup:
                        slot = zp % MAX_ZONES
                        z_lo[slot] = ob_lo
                        z_hi[slot] = ob_hi
                        z_dir[slot] = d
                        z_born[slot] = i
                        z_used[slot] = False
                        zp += 1

        # ── 3. Check all active zones for first-touch revisit at bar i ──
        if not in_sess:
            continue

        c = close_a[i]
        for s in range(MAX_ZONES):
            if z_used[s] or z_born[s] < 0:
                continue
            age = i - z_born[s]
            if age <= 0:
                continue
            if age > max_zone_age:
                z_born[s] = -1  # expire zone
                continue
            # Price enters zone → first-touch signal
            if z_lo[s] <= c <= z_hi[s]:
                z_used[s] = True
                if z_dir[s] == 1 and c >= vwap[i]:    # bullish zone + above VWAP
                    buy_ce[i] = True
                elif z_dir[s] == -1 and c <= vwap[i]: # bearish zone + below VWAP
                    buy_pe[i] = True

    return buy_ce, buy_pe


class Strategy(BaseStrategy):
    name = "order_block_entry_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min, let zones populate
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 144            # 12-min warmup (720s) to detect initial zones before trading

    def tunable_params(self) -> list:
        return [
            TunableParam("impulse_pts", 8.0, 4.0, 15.0),
            TunableParam("stop_pts",    4.0, 2.0,  8.0),
            TunableParam("target_pts",  7.0, 4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN before converting to numpy
        open_a  = spot_df["open"].fill_null(strategy="forward").to_numpy()
        high_a  = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low_a   = spot_df["low"].fill_null(strategy="forward").to_numpy()
        close_a = spot_df["close"].fill_null(strategy="forward").to_numpy()
        vol_a   = spot_df["volume"].fill_null(0).to_numpy()
        day_id  = spot_df["day_id"].to_numpy()
        time_m  = spot_df["time_minutes"].to_numpy()

        impulse_pts = float(params.get("impulse_pts", 8.0))
        stop_pts    = float(params.get("stop_pts", 4.0))
        target_pts  = float(params.get("target_pts", 7.0))
        max_zone_age = 72  # 6 minutes — fixed (not tunable: a structural feature of OB decay)

        vwap = _compute_vwap(high_a, low_a, close_a, vol_a, day_id)

        buy_ce, buy_pe = _compute_signals(
            open_a, high_a, low_a, close_a, vwap, time_m,
            impulse_pts, max_zone_age,
            self.session_start_minutes, self.session_end_minutes,
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
