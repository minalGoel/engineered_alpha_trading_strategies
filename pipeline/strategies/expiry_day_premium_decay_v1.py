"""
expiry_day_premium_decay_v1 — NIFTY Expiry-Day Gamma Overshoot Reversion

On NIFTY weekly expiry Thursdays, option market makers net-short gamma must
delta-hedge directionally as the index deviates from VWAP, amplifying the
move. Once VIX begins declining (gamma pressure easing), these overshoots
revert sharply. We enter when VWAP z-score is extreme AND starting to recover,
with VIX declining from session open to confirm gamma unwind.

Hold: 30-180 seconds. Only fires on Thursdays (~52 days/year).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "expiry_day_premium_decay_v1"
    underlying = "NIFTY"
    session_start_minutes = 600   # 10:00 IST — post-first-hour reversion window
    session_end_minutes = 840     # 14:00 IST — before chaotic expiry pinning period
    max_trades_per_day = 3        # selective: only on Thursdays, high-conviction only
    max_lookback = 300            # 25 min warmup to initialise rolling std (240-bar window)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_threshold", 1.5, 1.0, 2.5),
            TunableParam("zscore_delta_threshold", 0.10, 0.05, 0.30),
            TunableParam("stop_pts", 5.0, 3.0, 8.0),
            TunableParam("target_pts", 9.0, 5.0, 15.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill before numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Parameters ──
        zscore_threshold = params.get("zscore_threshold", 1.5)
        zscore_delta_threshold = params.get("zscore_delta_threshold", 0.10)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 9.0)

        # ── Is Expiry Day (Thursday) ──
        # Polars dt.weekday() follows ISO 8601: Monday=1 ... Thursday=4 ... Sunday=7
        weekday = spot_df.select(
            pl.col("session_date").dt.weekday().alias("wd")
        )["wd"].to_numpy()
        is_thursday = (weekday == 4)  # Thursday = 4 in ISO weekday

        # ── Session VWAP (cumulative from day start, reset each day) ──
        typical_price = (high + low + close) / 3.0
        cum_tp_vol = np.zeros(n)
        cum_vol = np.zeros(n)
        vwap = np.zeros(n)

        for i in range(n):
            is_new_day = (i == 0) or (day_id[i] != day_id[i - 1])
            if is_new_day:
                cum_tp_vol[i] = typical_price[i] * volume[i]
                cum_vol[i] = max(volume[i], 1e-9)
            else:
                cum_tp_vol[i] = cum_tp_vol[i - 1] + typical_price[i] * volume[i]
                cum_vol[i] = cum_vol[i - 1] + volume[i]
            vwap[i] = cum_tp_vol[i] / max(cum_vol[i], 1e-9)

        # ── VWAP Deviation Z-score (20-min rolling std = 240 bars) ──
        # 240 bars: captures current intraday vol regime without stale pre-10am noise
        ZSCORE_WINDOW = 240
        dev = close - vwap
        zscore = np.zeros(n)

        for i in range(ZSCORE_WINDOW, n):
            window = dev[i - ZSCORE_WINDOW:i]
            std = np.std(window)
            if std > 0.5:  # guard against near-zero std in extremely quiet markets
                zscore[i] = dev[i] / std

        # ── Z-score Delta over 60s (12 bars) — reversal confirmation ──
        # 12-bar window chosen so 5s noise doesn't trigger false reversals;
        # a genuine gamma-unwind reversion produces sustained directional z-score recovery
        DELTA_WINDOW = 12
        zscore_delta = np.zeros(n)
        zscore_delta[DELTA_WINDOW:] = zscore[DELTA_WINDOW:] - zscore[:-DELTA_WINDOW]

        # ── VIX Change from Session Open ──
        # Declining VIX = gamma pressure easing = overshoot reversion viable
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(
                    ["datetime", pl.col("close").alias("vix_close")]
                ).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # VIX at session open: carry forward the first VIX reading each day
        vix_at_open = np.full(n, 15.0)
        for i in range(n):
            is_new_day = (i == 0) or (day_id[i] != day_id[i - 1])
            if is_new_day:
                vix_at_open[i] = vix_close[i]
            else:
                vix_at_open[i] = vix_at_open[i - 1]

        vix_change = vix_close - vix_at_open  # negative = VIX declining from open

        # ── Session Filter ──
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Entry Signals ──
        # buy_ce: expiry day + NIFTY oversold vs VWAP + z-score recovering + VIX declining
        buy_ce = (
            is_thursday
            & in_session
            & (zscore < -zscore_threshold)
            & (zscore_delta > zscore_delta_threshold)   # z-score moving back toward 0
            & (vix_change < 0.0)                        # VIX declining from open
        )

        # buy_pe: expiry day + NIFTY overbought vs VWAP + z-score reverting + VIX declining
        buy_pe = (
            is_thursday
            & in_session
            & (zscore > zscore_threshold)
            & (zscore_delta < -zscore_delta_threshold)  # z-score moving back toward 0
            & (vix_change < 0.0)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=36,          # 180s = 3 min; reversion completes or thesis is wrong
            max_trades_per_day=self.max_trades_per_day,
        )
