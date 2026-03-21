"""ORB Index Aligned — NIFTY Opening Range Breakout with VWAP Confirmation

Original: orb_index_aligned_v1 (equity stock ORB aligned with NIFTY ORB direction)
Conversion: Trade NIFTY options directly when NIFTY breaks its own 15-min ORB,
confirmed by 3 consecutive 5-second closes and VWAP alignment.

Mechanism: NIFTY's 09:15–09:30 opening range is the overnight information absorption
window. A breakout above range high with 3+ consecutive 5s bar confirmation signals
institutional TWAP/VWAP algorithms have exhausted supply at the ORB ceiling and are
continuing to accumulate. VWAP filter ensures we're aligned with session bias.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "orb_index_aligned_v1"
    underlying = "NIFTY"
    session_start_minutes = 570    # 09:30 IST — after ORB fully forms at 09:30
    session_end_minutes = 920      # 15:20 IST
    max_trades_per_day = 5
    max_lookback = 240             # 20-min warmup covers full ORB formation period

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("confirm_bars", 3.0, 1.0, 6.0),
            TunableParam("vix_low", 13.0, 10.0, 17.0),
            TunableParam("vix_high", 22.0, 18.0, 28.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        confirm_bars = max(1, int(params.get("confirm_bars", 3)))
        vix_low = params.get("vix_low", 13.0)
        vix_high = params.get("vix_high", 22.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # VIX — align to spot bars via backward asof join
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # Per-day: compute ORB levels (09:15–09:30) and VWAP
        orb_high = np.zeros(n)
        orb_low = np.full(n, np.inf)
        orb_formed = np.zeros(n, dtype=bool)
        vwap = np.zeros(n)

        for d in np.unique(day_id):
            day_mask = day_id == d
            idx = np.where(day_mask)[0]

            # ORB window: 09:15–09:30 = time_min [555, 570)
            orb_mask = day_mask & (time_min >= 555) & (time_min < 570)
            orb_idx = np.where(orb_mask)[0]

            if len(orb_idx) > 0:
                d_orb_high = np.max(high[orb_idx])
                d_orb_low = np.min(low[orb_idx])
                orb_high[day_mask] = d_orb_high
                orb_low[day_mask] = d_orb_low
                orb_formed[day_mask] = True

            # Cumulative VWAP from session open
            tp = (high[idx] + low[idx] + close[idx]) / 3.0
            cum_vol = np.cumsum(volume[idx])
            cum_tp_vol = np.cumsum(tp * volume[idx])
            with np.errstate(divide="ignore", invalid="ignore"):
                day_vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, close[idx])
            vwap[idx] = day_vwap

        # Consecutive-bar persistence counter (resets at day boundary and on break)
        above_count = np.zeros(n, dtype=int)
        below_count = np.zeros(n, dtype=int)
        for i in range(1, n):
            same_day = day_id[i] == day_id[i - 1]
            above_count[i] = (above_count[i - 1] + 1 if same_day else 1) if close[i] > orb_high[i] else 0
            below_count[i] = (below_count[i - 1] + 1 if same_day else 1) if close[i] < orb_low[i] else 0

        # Session and VIX regime filters
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = (vix_close >= vix_low) & (vix_close <= vix_high)

        # Fire exactly on the confirm_bars-th consecutive close above/below ORB
        # (fires once per breakout event; re-fires if price dips back and breaks again)
        buy_ce = (
            in_session
            & vix_ok
            & orb_formed
            & (above_count == confirm_bars)
            & (close > vwap)
        )

        buy_pe = (
            in_session
            & vix_ok
            & orb_formed
            & (below_count == confirm_bars)
            & (close < vwap)
        )

        # Prevent simultaneous CE + PE signals
        both = buy_ce & buy_pe
        buy_ce = buy_ce & ~both
        buy_pe = buy_pe & ~both

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
