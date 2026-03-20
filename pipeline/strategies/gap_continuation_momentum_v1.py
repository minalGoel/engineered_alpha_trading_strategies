"""gap_continuation_momentum_v1 — Gap-continuation regime filter with 5s micro-momentum entries.

Mechanism: When NIFTY opens with a gap >0.3% driven by overnight global cues, FII and
institutional VWAP algorithms are directionally aligned. If the gap holds past 09:35 (close
still above session open for gap-up), we enter on 1-minute positive micro-momentum signals
in the gap direction, capturing 10-20 spot point continuation pulses as 5-10 option points.

Converted from: trading_strategies/unique_strategies_all/Strategy_169.json
Original: Gap continuation on NIFTY200 stocks, 1-min bars, 30-90 min hold.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "gap_continuation_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 575   # 09:35 IST — after 20-min gap confirmation window
    session_end_minutes = 780     # 13:00 IST — avoid afternoon reversal sessions
    max_trades_per_day = 5
    max_lookback = 180            # 15 minutes (confirmation window warmup)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_threshold", 0.003, 0.002, 0.008),   # 0.3% NIFTY gap minimum
            TunableParam("mom_threshold", 0.0001, 0.00005, 0.0003),  # 1-min micro-momentum trigger
            TunableParam("vix_max", 22.0, 16.0, 28.0),
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            TunableParam("target_pts", 5.0, 3.0, 10.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        gap_threshold = params.get("gap_threshold", 0.003)
        mom_threshold = params.get("mom_threshold", 0.0001)
        vix_max = params.get("vix_max", 22.0)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 5.0)

        # ── Build per-day session open and last close ──────────────────────────
        day_open = {}       # day_id -> first bar's open (session open)
        day_last_close = {} # day_id -> last seen close (updated each bar)

        for i in range(n):
            d = int(day_id[i])
            if d not in day_open:
                day_open[d] = open_[i]
            day_last_close[d] = close[i]  # last bar of each day wins

        sorted_days = sorted(day_open.keys())

        # Map each day to the prior day's last close
        day_prev_close = {}
        for k in range(1, len(sorted_days)):
            d = sorted_days[k]
            prev_d = sorted_days[k - 1]
            day_prev_close[d] = day_last_close[prev_d]

        # ── Per-bar gap direction and holding confirmation ─────────────────────
        # gap_dir: +1 = gap-up day, -1 = gap-down day, 0 = no meaningful gap
        # gap_confirmed: True if past 09:35 AND gap still holding (not filled)
        gap_dir = np.zeros(n)
        gap_confirmed = np.zeros(n, dtype=bool)

        for i in range(n):
            d = int(day_id[i])
            if d not in day_prev_close:
                continue  # first day of the dataset: no prior close available

            pc = day_prev_close[d]
            so = day_open[d]
            gp = (so - pc) / pc  # gap as fraction of prev close

            if gp > gap_threshold:
                gap_dir[i] = 1.0
            elif gp < -gap_threshold:
                gap_dir[i] = -1.0

            # Confirmation: past 09:35 IST (575 min) AND gap not filled
            if time_min[i] >= 575:
                if gap_dir[i] > 0 and close[i] > so:
                    gap_confirmed[i] = True
                elif gap_dir[i] < 0 and close[i] < so:
                    gap_confirmed[i] = True

        # ── 1-minute micro-momentum trigger (12 bars × 5s = 60s) ─────────────
        # Positive momentum = price is rising over the past minute → continuation onset
        mom_12 = np.zeros(n)
        for i in range(12, n):
            base = close[i - 12]
            if base != 0.0:
                mom_12[i] = (close[i] - base) / base

        # ── VIX regime filter ──────────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Entry signals ──────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        low_vix = vix_close < vix_max

        # Buy CE: gap-up regime confirmed + fresh positive micro-momentum
        buy_ce = in_session & low_vix & (gap_dir > 0) & gap_confirmed & (mom_12 > mom_threshold)

        # Buy PE: gap-down regime confirmed + fresh negative micro-momentum
        buy_pe = in_session & low_vix & (gap_dir < 0) & gap_confirmed & (mom_12 < -mom_threshold)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,             # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
