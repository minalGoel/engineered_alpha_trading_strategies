"""orb_momentum_v32 — Rolling Range Breakout on NIFTY (5-second bars)

Original: 5-min ORB on nifty200 stocks (Strategy_104.json) using 21-bar rolling
highest-high / lowest-low (105-min window) with VWAP + volume confirmation.

Conversion thesis: NIFTY rolling-range breakouts throughout the day, NOT the
fixed opening range (orb_v22). Captures afternoon consolidation-breakouts where
TWAP/VWAP algorithms have established a 30-min balance zone and new directional
orders push through the range ceiling/floor.

Key differences from orb_momentum_v22:
- v22: Fixed 09:15-09:35 opening range (static, morning only)
- v32: Rolling 30-min range (dynamic, all day, captures afternoon breakouts)
"""

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "orb_momentum_v32"
    underlying = "NIFTY"
    session_start_minutes = 575    # 09:35 IST — after 360 bars (30 min) are populated
    session_end_minutes = 870      # 14:30 IST — avoid late-session thin liquidity
    max_trades_per_day = 5
    max_lookback = 360             # 30-min warmup for rolling range

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_ratio_threshold", 1.2, 0.8, 2.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        vol_ratio_threshold = params.get("vol_ratio_threshold", 1.2)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # ── Compute session VWAP (cumulative, reset per day) ─────────────────
        typical = (high + low + close) / 3.0
        vwap = np.zeros(n)
        tpv_cum = 0.0
        vol_cum = 0.0
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                tpv_cum = typical[i] * volume[i]
                vol_cum = volume[i]
            else:
                tpv_cum += typical[i] * volume[i]
                vol_cum += volume[i]
            vwap[i] = tpv_cum / vol_cum if vol_cum > 0 else close[i]

        # ── Rolling 30-min (360-bar) highest high and lowest low ─────────────
        WIN = 360
        rolling_high = np.full(n, np.nan)
        rolling_low = np.full(n, np.nan)
        for i in range(WIN - 1, n):
            rolling_high[i] = np.max(high[i - WIN + 1: i + 1])
            rolling_low[i] = np.min(low[i - WIN + 1: i + 1])

        # ── Volume ratio: current bar vs 10-min (120-bar) rolling mean ───────
        VOL_WIN = 120
        vol_mean = np.full(n, np.nan)
        for i in range(VOL_WIN - 1, n):
            w = volume[i - VOL_WIN + 1: i + 1]
            m = np.mean(w)
            vol_mean[i] = m if m > 0 else 1.0
        # Where vol_mean is NaN → ratio = 0.0 (won't fire)
        vol_ratio = np.where(
            (~np.isnan(vol_mean)) & (vol_mean > 0),
            volume / vol_mean,
            0.0,
        )

        # ── 3-bar persistence: 15-second confirmation filter ─────────────────
        # NaN comparisons produce False in numpy → safe default
        above_high = close > rolling_high   # NaN rolling_high → False
        below_low = close < rolling_low     # NaN rolling_low → False

        persist_above = np.zeros(n, dtype=bool)
        persist_below = np.zeros(n, dtype=bool)
        for i in range(2, n):
            persist_above[i] = above_high[i] and above_high[i - 1] and above_high[i - 2]
            persist_below[i] = below_low[i] and below_low[i - 1] and below_low[i - 2]

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Entry signals ─────────────────────────────────────────────────────
        range_valid = ~np.isnan(rolling_high)

        buy_ce = (
            in_session
            & range_valid
            & persist_above
            & (close > vwap)
            & (vol_ratio > vol_ratio_threshold)
        )

        buy_pe = (
            in_session
            & range_valid
            & persist_below
            & (close < vwap)
            & (vol_ratio > vol_ratio_threshold)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,
            max_trades_per_day=self.max_trades_per_day,
        )
