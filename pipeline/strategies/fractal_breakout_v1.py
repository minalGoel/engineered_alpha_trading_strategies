"""fractal_breakout_v1 — Williams Fractal swing-point breakout on NIFTY 5s bars.

Mechanism:
    5-bar Williams Fractals identify micro swing highs/lows on NIFTY (25-second windows).
    These fractal levels represent price-discovery peaks where short-sellers cluster limit
    sell orders. When NIFTY breaks above a recent fractal high with all three Alligator
    SMAs (5/8/13 bars = 25s/40s/65s) in bullish alignment and price above session VWAP,
    resting supply at the swing level has been absorbed and stop-loss cascades on shorts
    drive a secondary push. 2-bar persistence required to filter NIFTY constituent lag spikes.

Converted from: trading_strategies/unique_strategies_all/Strategy_284.json
Original: Williams Fractals + Alligator on 1-min equity bars, hold 5-15 min.
"""
from __future__ import annotations
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "fractal_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST — skip first 5 min of open noise
    session_end_minutes = 920     # 15:20 IST — skip last 10 min of EOD squaring
    max_lookback = 120            # 10 min warmup (13-bar SMA + fractal tracking)
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_max", 24.0, 16.0, 30.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill NaN before numpy) ──────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ───────────────────────────────────────────────────────
        vix_max = params.get("vix_max", 24.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)
        # Fractal freshness: fractals older than this many bars are stale (60 bars = 5 min)
        fractal_staleness = 60

        # ── Alligator: 3 raw SMAs without the original's forward shifts ──────
        # Shifts are removed: at 5s frequency they add 15-40s of lag that hurts more than it
        # helps. Raw SMAs detect current alignment; we rely on 2-bar persistence instead.
        sma5 = _rolling_sma(close, 5)    # lips: 25s — fastest trend component
        sma8 = _rolling_sma(close, 8)    # teeth: 40s — medium trend component
        sma13 = _rolling_sma(close, 13)  # jaw: 65s — context/direction filter

        # ── Session VWAP (cumulative, resets per day) ─────────────────────────
        vwap = _compute_vwap(close, volume, day_id)

        # ── India VIX (aligned to spot bars) ────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Williams Fractal tracking ─────────────────────────────────────────
        # Fractal up at bar t: high[t-2] is the swing peak (confirmed at t when t, t-1 are lower)
        # Fractal dn at bar t: low[t-2] is the swing trough (confirmed at t when t, t-1 are higher)
        # We track the most recent confirmed fractal level within `fractal_staleness` bars.
        last_fractal_up = np.full(n, np.nan)   # most recent swing high level
        last_fractal_dn = np.full(n, np.nan)   # most recent swing low level

        up_level = np.nan
        up_age = 99999
        dn_level = np.nan
        dn_age = 99999

        for t in range(4, n):
            up_age += 1
            dn_age += 1

            # Check if bar t-2 is a confirmed fractal high
            if (high[t - 2] > high[t - 4] and high[t - 2] > high[t - 3] and
                    high[t - 2] > high[t - 1] and high[t - 2] > high[t]):
                up_level = high[t - 2]
                up_age = 2  # peak was 2 bars ago; confirmation delay is 2 bars

            # Check if bar t-2 is a confirmed fractal low
            if (low[t - 2] < low[t - 4] and low[t - 2] < low[t - 3] and
                    low[t - 2] < low[t - 1] and low[t - 2] < low[t]):
                dn_level = low[t - 2]
                dn_age = 2

            if up_age <= fractal_staleness and not np.isnan(up_level):
                last_fractal_up[t] = up_level
            if dn_age <= fractal_staleness and not np.isnan(dn_level):
                last_fractal_dn[t] = dn_level

        # ── Primary conditions ────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = vix_close <= vix_max

        # Alligator full alignment: all 3 SMAs fanning in same direction
        alligator_bullish = (sma5 > sma8) & (sma8 > sma13)
        alligator_bearish = (sma5 < sma8) & (sma8 < sma13)

        # Fractal breakout/breakdown (single bar)
        has_up = ~np.isnan(last_fractal_up)
        has_dn = ~np.isnan(last_fractal_dn)
        above_fractal = has_up & (close > last_fractal_up)
        below_fractal = has_dn & (close < last_fractal_dn)

        # 2-bar persistence: this bar AND the previous bar both exceed fractal level
        # Filters single-tick NIFTY constituent lag spikes
        prev_above = np.roll(above_fractal, 1)
        prev_above[0] = False
        prev_below = np.roll(below_fractal, 1)
        prev_below[0] = False

        # VWAP context: only trade in the direction consistent with where we are relative to fair value
        above_vwap = close > vwap
        below_vwap = close < vwap

        # ── Final signals ─────────────────────────────────────────────────────
        buy_ce = in_session & vix_ok & alligator_bullish & above_fractal & prev_above & above_vwap
        buy_pe = in_session & vix_ok & alligator_bearish & below_fractal & prev_below & below_vwap

        return OptionSignals(
            buy_ce=buy_ce.astype(bool),
            buy_pe=buy_pe.astype(bool),
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,          # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rolling_sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Rolling simple moving average. Warmup bars filled with arr[0]."""
    n = len(arr)
    out = np.empty(n)
    init = arr[0] if (n > 0 and not np.isnan(arr[0])) else 0.0
    out[:period - 1] = init
    cs = np.cumsum(arr)
    out[period - 1] = cs[period - 1] / period
    for i in range(period, n):
        out[i] = (cs[i] - cs[i - period]) / period
    return out


def _compute_vwap(
    close: np.ndarray,
    volume: np.ndarray,
    day_id: np.ndarray,
) -> np.ndarray:
    """Session VWAP, resets at each new day_id."""
    n = len(close)
    vwap = np.empty(n)
    cum_pv = 0.0
    cum_v = 0.0
    cur_day = -1
    for i in range(n):
        if day_id[i] != cur_day:
            cur_day = day_id[i]
            cum_pv = 0.0
            cum_v = 0.0
        v = max(volume[i], 0.0)
        cum_pv += close[i] * v
        cum_v += v
        vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]
    return vwap
