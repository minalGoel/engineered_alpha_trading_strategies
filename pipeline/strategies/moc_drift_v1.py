"""
moc_drift_v1 — End-of-Day MOC Drift on NIFTY

Mechanism:
    In the 14:30–15:20 IST window, NIFTY experiences systematic directional drift
    driven by ETF rebalancers, passive-fund TWAP algos, and institutional MOC programs
    that are pre-committed to executing at or near the close price. When NIFTY is above
    session VWAP and up since 14:00, pre-committed buy programs create 15–30 second
    momentum bursts as they absorb resting sell liquidity faster than new sellers
    replenish it. A 1-minute momentum confirmation signals an active burst is underway.

Converted from: trading_strategies/unique_strategies_all/Strategy_82.json
Original: MOC drift on NIFTY50 stocks (1-min bars, 15-55 min hold)
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "moc_drift_v1"
    underlying = "NIFTY"
    session_start_minutes = 870   # 14:30 IST — MOC window open
    session_end_minutes = 920     # 15:20 IST — stop new entries before EOD flatten
    max_trades_per_day = 4
    max_lookback = 360            # 30 min warmup (360 × 5s); strategy only signals after 14:30

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_threshold", 0.10, 0.05, 0.25),    # % NIFTY above/below VWAP
            TunableParam("pm_return_threshold", 0.08, 0.03, 0.20), # % return since 14:00
            TunableParam("mom_threshold", 0.0002, 0.0001, 0.0008), # 1-min momentum threshold
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays (forward-fill before numpy conversion) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ──
        vwap_threshold = params.get("vwap_threshold", 0.10)
        pm_return_threshold = params.get("pm_return_threshold", 0.08)
        mom_threshold = params.get("mom_threshold", 0.0002)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # ── 1. Session VWAP (cumulative per day, reset at each new day_id) ──
        vwap = np.zeros(n)
        cum_vol = 0.0
        cum_pv = 0.0
        prev_day = -999999
        for i in range(n):
            if day_id[i] != prev_day:
                cum_vol = 0.0
                cum_pv = 0.0
                prev_day = day_id[i]
            v = volume[i] if volume[i] > 0 else 1.0
            cum_vol += v
            cum_pv += close[i] * v
            vwap[i] = cum_pv / cum_vol

        # ── 2. VWAP deviation (%) ──
        vwap_dev = np.where(vwap > 0, (close - vwap) / vwap * 100.0, 0.0)

        # ── 3. Return since 14:00 (840 min) — fixed daily anchor ──
        # Lock in the first bar at time_minutes >= 840 for each day as reference price
        return_since_1400 = np.zeros(n)
        ref_price_by_day: dict[int, float] = {}
        for i in range(n):
            d = int(day_id[i])
            if d not in ref_price_by_day and time_min[i] >= 840:
                ref_price_by_day[d] = close[i]
            ref = ref_price_by_day.get(d, 0.0)
            if ref > 0:
                return_since_1400[i] = (close[i] - ref) / ref * 100.0
            # bars before 14:00 remain 0.0 — no signal fires there anyway (time filter)

        # ── 4. 1-minute momentum (12 bars × 5s = 60s) — entry trigger ──
        momentum_12 = np.zeros(n)
        for i in range(12, n):
            if day_id[i] == day_id[i - 12] and close[i - 12] > 0:
                momentum_12[i] = (close[i] - close[i - 12]) / close[i - 12]

        # ── 5. VIX filter: skip when VIX >= 22 (panic regime; MOC programs unreliable) ──
        vix_ok = np.ones(n, dtype=bool)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()
            vix_ok = vix_close < 22.0

        # ── 6. Session filter: 14:30–15:20 IST ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── 7. Entry signals ──
        buy_ce = (
            in_session
            & vix_ok
            & (vwap_dev > vwap_threshold)
            & (return_since_1400 > pm_return_threshold)
            & (momentum_12 > mom_threshold)
        )

        buy_pe = (
            in_session
            & vix_ok
            & (vwap_dev < -vwap_threshold)
            & (return_since_1400 < -pm_return_threshold)
            & (momentum_12 < -mom_threshold)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,       # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
