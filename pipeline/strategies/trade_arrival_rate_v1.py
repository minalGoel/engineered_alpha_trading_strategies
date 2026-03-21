"""
trade_arrival_rate_v1 — Volume Arrival Rate Anomaly on NIFTY

Mechanism:
On NIFTY, institutional TWAP/VWAP programs create volume bursts 2-3x above the
expected activity for that time of day (U-shaped intraday profile). When such a
burst coincides with directional price impact (burst bar closes above/below its
open) and NIFTY is positioned above/below session VWAP, the institutional order
is only partially filled and will continue pushing prices for 30-90 seconds.

Adapted from: trading_strategies/unique_strategies_all/Strategy_247.json
Original: trade count anomaly on NIFTY 50 stocks (1-min bars, 3-8 min hold).
Adaptation: volume replaces trade count; 15s burst window; 30-90s hold.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "trade_arrival_rate_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min (naturally elevated)
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 360            # 30 min warmup for profile stabilisation

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_ratio_threshold", 2.0, 1.5, 3.5),
            TunableParam("price_impact_threshold", 0.002, 0.001, 0.005),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays (forward-fill before .to_numpy()) ──────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        volume = (
            spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        )
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Parameters ────────────────────────────────────────────────────────
        vol_ratio_thr = float(params.get("vol_ratio_threshold", 2.0))
        price_impact_thr = float(params.get("price_impact_threshold", 0.002))
        stop_pts = float(params.get("stop_pts", 4.0))
        target_pts = float(params.get("target_pts", 7.0))

        # ── 1. 3-bar (15s) rolling volume sum ─────────────────────────────────
        vol_15s = np.zeros(n)
        for i in range(2, n):
            vol_15s[i] = volume[i] + volume[i - 1] + volume[i - 2]

        # ── 2. Time-of-day expected volume profile (U-shape normaliser) ───────
        # Median volume per time_minutes slot across all available days.
        # U-shape is very stable; using full dataset for profile is acceptable.
        unique_times = np.unique(time_min)
        time_to_expected: dict[int, float] = {}
        for t in unique_times:
            mask = time_min == t
            vals = volume[mask]
            if len(vals) > 0:
                med = float(np.median(vals))
                time_to_expected[int(t)] = med if med > 0 else 1.0
            else:
                time_to_expected[int(t)] = 1.0

        # Expected 15s-window volume = 3 × per-bar expected at that time slot
        expected_vol_15s = np.ones(n)
        for i in range(n):
            t = int(time_min[i])
            expected_vol_15s[i] = max(time_to_expected.get(t, 1.0) * 3.0, 1.0)

        # ── 3. Volume ratio (anomaly score) ───────────────────────────────────
        vol_ratio = np.zeros(n)
        for i in range(2, n):
            vol_ratio[i] = vol_15s[i] / expected_vol_15s[i]

        # ── 4. Bar directional return (price impact) ──────────────────────────
        safe_open = np.where(open_ > 0, open_, 1.0)
        bar_return = (close - open_) / safe_open

        # ── 5. Session VWAP (cumulative from daily open) ──────────────────────
        vwap = np.zeros(n)
        cum_vol = 0.0
        cum_vp = 0.0
        cur_day = int(day_id[0]) if n > 0 else -1

        for i in range(n):
            d = int(day_id[i])
            if d != cur_day:
                cur_day = d
                cum_vol = volume[i]
                cum_vp = close[i] * volume[i]
            else:
                cum_vol += volume[i]
                cum_vp += close[i] * volume[i]
            vwap[i] = cum_vp / max(cum_vol, 1.0)

        # ── 6. Session filter ─────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── 7. Entry signals ──────────────────────────────────────────────────
        # Bullish: vol burst + positive directional impact + above VWAP
        buy_ce = (
            in_session
            & (vol_ratio >= vol_ratio_thr)
            & (bar_return > price_impact_thr)
            & (close > vwap)
        )

        # Bearish: vol burst + negative directional impact + below VWAP
        buy_pe = (
            in_session
            & (vol_ratio >= vol_ratio_thr)
            & (bar_return < -price_impact_thr)
            & (close < vwap)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
