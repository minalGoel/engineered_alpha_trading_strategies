"""ORB Pullback Entry — NIFTY 5-second options strategy.

After NIFTY breaks the 09:15-09:30 opening range, institutional VWAP algorithms
temporarily exhaust their order queue and price pulls back to the ORB level.
A 5-second bar that touches the ORB level on its low but closes above it signals
resting institutional buy orders absorbing the pullback. The subsequent continuation
leg extends 12-25 spot points within 30-90 seconds.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "orb_pullback_entry_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — after ORB window closes
    session_end_minutes = 660     # 11:00 IST — pullbacks later in day less reliable
    max_trades_per_day = 3
    max_lookback = 240            # 20 min warmup (ensures ORB window is fully formed)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("pullback_tolerance", 0.0005, 0.0001, 0.0015),
            TunableParam("max_pullback_bars", 150.0, 60.0, 300.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        pullback_tol = params.get("pullback_tolerance", 0.0005)
        max_pb_bars = int(params.get("max_pullback_bars", 150))

        # ORB window: 09:15–09:30 IST (minutes 555–569 inclusive)
        ORB_START = 555
        ORB_END = 570

        buy_ce = np.zeros(n, dtype=bool)
        buy_pe = np.zeros(n, dtype=bool)

        unique_days = np.unique(day_id)

        for d in unique_days:
            day_mask = day_id == d

            # ── Build ORB for this day ──────────────────────────────────────
            orb_mask = day_mask & (time_min >= ORB_START) & (time_min < ORB_END)
            orb_indices = np.where(orb_mask)[0]
            if len(orb_indices) == 0:
                continue

            orb_high = np.max(high[orb_indices])
            orb_low = np.min(low[orb_indices])

            # ── Scan post-ORB bars for breakout then pullback ───────────────
            post_orb_mask = day_mask & (time_min >= ORB_END) & (time_min < self.session_end_minutes)
            post_orb_indices = np.where(post_orb_mask)[0]
            if len(post_orb_indices) < 4:
                continue

            bull_consecutive = 0
            bear_consecutive = 0
            bull_breakout_bar = -1   # global index where bull breakout was confirmed
            bear_breakout_bar = -1

            for idx in post_orb_indices:
                c = close[idx]
                h = high[idx]
                lo = low[idx]

                # ── Track consecutive closes for breakout confirmation ──────
                if c > orb_high:
                    bull_consecutive += 1
                    bear_consecutive = 0
                elif c < orb_low:
                    bear_consecutive += 1
                    bull_consecutive = 0
                else:
                    bull_consecutive = 0
                    bear_consecutive = 0

                # Confirm breakout on the 3rd consecutive bar
                if bull_consecutive == 3 and bull_breakout_bar < 0:
                    bull_breakout_bar = idx
                if bear_consecutive == 3 and bear_breakout_bar < 0:
                    bear_breakout_bar = idx

                # ── Pullback detection: bull setup (buy CE) ─────────────────
                if bull_breakout_bar >= 0 and (idx - bull_breakout_bar) <= max_pb_bars:
                    # Low touches ORB high (within tolerance) AND close holds above
                    if abs(lo - orb_high) / orb_high <= pullback_tol and c > orb_high:
                        buy_ce[idx] = True

                # ── Pullback detection: bear setup (buy PE) ─────────────────
                if bear_breakout_bar >= 0 and (idx - bear_breakout_bar) <= max_pb_bars:
                    # High touches ORB low (within tolerance) AND close holds below
                    if abs(h - orb_low) / orb_low <= pullback_tol and c < orb_low:
                        buy_pe[idx] = True

        # ── Session filter ──────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        buy_ce = buy_ce & in_session
        buy_pe = buy_pe & in_session

        # ── VIX filter: stable regime only (12–22) ──────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        vix_ok = (vix_close >= 12.0) & (vix_close <= 22.0)
        buy_ce = buy_ce & vix_ok
        buy_pe = buy_pe & vix_ok

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # stop: 3 pts ≈ 6 NIFTY spot pts — ORB level failure threshold
            stop_points=np.full(n, 3.0),
            # target: 6 pts ≈ 12 NIFTY spot pts — lower half of typical 15-30pt continuation leg
            target_points=np.full(n, 6.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,             # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
