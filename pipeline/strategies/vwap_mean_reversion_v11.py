"""vwap_mean_reversion_v11 — Forced-Liquidation Exhaustion at Extreme VWAP Deviation.

Mechanism:
    On NIFTY, when the z-score relative to the 20-minute rolling std drops below -2.5
    (~60-90 spot pts below VWAP), the selling is dominated by forced margin liquidations
    of leveraged long positions. Once this forced-sell pool is depleted, price snaps back
    25-35 spot pts within 60 seconds. The exhaustion is detected via range_position_6:
    the close reclaiming >30% of the 6-bar (30-second) high-low range from below while
    z-score is still extreme signals that forced sellers have been absorbed.

Differentiation from other VWAP versions:
    - v10: z=-2.3, VIX band 12-22, rel_volume (VWAP-urgency programs in normal VIX)
    - v18: z=-1.7, dev_momentum_24 (z-score rate change, deep deviation with momentum turn)
    - v19: moderate z, range-contraction microstructure (market-maker gamma hedging)
    - v11: z=-2.5 (most extreme), range_position_6 exhaustion, VIX<25 upper cap only

Converted from: trading_strategies/unique_strategies_all/Strategy_96.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "vwap_mean_reversion_v11"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 240            # 20-min warmup (240 bars x 5s)
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_threshold", 2.5, 2.0, 3.0),
            TunableParam("range_thresh", 0.3, 0.1, 0.6),
            TunableParam("vix_max", 25.0, 18.0, 35.0),
            TunableParam("stop_pts", 5.0, 3.0, 9.0),
            TunableParam("target_pts", 9.0, 5.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Pull spot arrays — forward-fill NaN in Polars before converting
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # Tunable parameters
        zscore_thresh = float(params.get("zscore_threshold", 2.5))
        range_thresh = float(params.get("range_thresh", 0.3))
        vix_max = float(params.get("vix_max", 25.0))
        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 9.0))

        # ── VIX alignment (backward asof join) ──────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Session VWAP (cumulative, resets per day) ────────────────────────
        vwap = np.zeros(n)
        cum_pv = 0.0
        cum_v = 0.0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_pv = 0.0
                cum_v = 0.0
                prev_day = day_id[i]
            cum_pv += close[i] * volume[i]
            cum_v += volume[i]
            vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]

        # ── 20-min rolling std (240 bars) — z-score denominator ─────────────
        std_240 = np.full(n, 1.0)
        for i in range(240, n):
            s = np.std(close[i - 240:i])
            std_240[i] = s if s > 0.5 else 0.5

        # ── VWAP z-score ─────────────────────────────────────────────────────
        vwap_zscore = (close - vwap) / std_240

        # ── 30-second range position (6-bar window) ──────────────────────────
        # range_pos[i] = (close[i] - min(close[i-5:i+1])) / range(close[i-5:i+1])
        # Value near 1 = close near 6-bar high; near 0 = close near 6-bar low.
        # For buy_ce: need range_pos > range_thresh (price lifted off the extreme low).
        # For buy_pe: need (1 - range_pos) > range_thresh (price fell from extreme high).
        range_pos = np.full(n, 0.5)
        for i in range(5, n):
            window = close[i - 5:i + 1]
            lo = window.min()
            hi = window.max()
            rng = hi - lo
            if rng > 0.1:
                range_pos[i] = (close[i] - lo) / rng
            # else leave at 0.5 (neutral)

        # ── Session & warmup filters ─────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= 240
        vix_ok = vix_close < vix_max

        # ── Entry signals ────────────────────────────────────────────────────
        # buy_ce: extreme negative z-score + price reclaiming from 30s low + VIX ok
        buy_ce = (
            in_session
            & warmed
            & vix_ok
            & (vwap_zscore < -zscore_thresh)
            & (range_pos > range_thresh)
        )

        # buy_pe: extreme positive z-score + price falling from 30s high + VIX ok
        buy_pe = (
            in_session
            & warmed
            & vix_ok
            & (vwap_zscore > zscore_thresh)
            & ((1.0 - range_pos) > range_thresh)
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
