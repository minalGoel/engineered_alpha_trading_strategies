"""vwap_mean_reversion_v10 — VWAP Mid-Depth Reversion in Normal VIX Regime (Band 12-22).

Converted from: trading_strategies/unique_strategies_all/Strategy_95.json
Original: 5-min VWAP z-score mean reversion on top-50 FnO stocks, z<-2.3,
rel_volume > 1.4, VIX band 12-22.

Mechanism: On NIFTY, z-score deviations of -2.3 standard deviations from session
VWAP (~50-70 spot points below the institutional benchmark) represent the mid-level
dislocation class — too deep for passive gamma-hedger fades but not deep enough to
trigger the multi-buyer convergence of capitulation events. The VIX band 12-22 is
the defining differentiator: VIX<12 (ultra-calm, no VWAP urgency) and VIX>22
(crisis, sustained directional flow overpowers reversion) are both excluded.
In the 12-22 band, VWAP-benchmarked buy programs activate predictably at z=-2.3,
and the 1.4x volume confirmation detects the moment they shift from passive to
active program buying.

Differentiation from other VWAP versions:
- v18: dev_momentum trigger, no VIX filter, stop=6/target=10
- v19: microstructure basing (range contraction), shallow deviations, no VIX filter
- v1:  RSI(36) dual-confirmation, VIX < 20 (upper-only bound), z=-1.5
- v3:  panic capitulation, z=-2.7, 1.7x volume, no VIX filter, stop=7/target=12
- v5:  VIX > 15 (lower-only bound), z=-2.7, 1.2x volume, 4-min baseline
- v10: VIX BAND 12-22 (both sides), z=-2.3, 1.4x volume, 3-min baseline, ends 14:30
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "vwap_mean_reversion_v10"
    underlying = "NIFTY"
    session_start_minutes = 570    # 09:30 IST — 15 min from open for VWAP/std stabilization
    session_end_minutes = 870      # 14:30 IST — original strategy's earlier cutoff
    max_trades_per_day = 6
    max_lookback = 180             # 15 min warmup for 180-bar rolling std

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_threshold", 2.3, 1.8, 3.0),
            TunableParam("rel_vol_threshold", 1.4, 1.1, 2.0),
            TunableParam("vix_min", 12.0, 10.0, 16.0),
            TunableParam("vix_max", 22.0, 18.0, 28.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill before numpy conversion
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        zscore_threshold = params.get("zscore_threshold", 2.3)
        rel_vol_threshold = params.get("rel_vol_threshold", 1.4)
        vix_min = params.get("vix_min", 12.0)
        vix_max = params.get("vix_max", 22.0)

        # ── VIX close aligned backward to spot bars ───────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Session VWAP — cumulative, reset each day ─────────────────────────
        cum_tpv = np.zeros(n)
        cum_vol = np.zeros(n)
        vwap = np.zeros(n)

        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                cum_tpv[i] = close[i] * volume[i]
                cum_vol[i] = volume[i]
            else:
                cum_tpv[i] = cum_tpv[i - 1] + close[i] * volume[i]
                cum_vol[i] = cum_vol[i - 1] + volume[i]

            if cum_vol[i] > 0:
                vwap[i] = cum_tpv[i] / cum_vol[i]
            else:
                vwap[i] = close[i]

        # ── Rolling 180-bar (15 min) std of close for z-score denominator ─────
        # 15 min chosen: more responsive than v5's 240 bars (20 min) while being
        # more stable than a 5-min window; suited to z=-2.3 mid-depth threshold.
        std_180 = np.zeros(n)
        for i in range(180, n):
            std_180[i] = np.std(close[i - 180:i])

        # ── VWAP z-score ──────────────────────────────────────────────────────
        vwap_zscore = np.zeros(n)
        for i in range(n):
            if std_180[i] > 0:
                vwap_zscore[i] = (close[i] - vwap[i]) / std_180[i]
            # else stays 0.0 (neutral — no signal during warmup)

        # ── Relative volume: 3-minute (36 bar) rolling mean baseline ─────────
        # 36 bars = 3 min. Between v3's 24-bar (2 min) and v5's 48-bar (4 min).
        # Normal-VIX program buying builds over 2-4 min; 3-min baseline is optimal.
        rel_volume = np.ones(n)
        for i in range(36, n):
            vol_mean = np.mean(volume[i - 36:i])
            if vol_mean > 0:
                rel_volume[i] = volume[i] / vol_mean
            # else stays 1.0 (neutral)

        # ── Session filter and warmup gate ────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        has_warmup = np.arange(n) >= 180  # need 180 bars for rolling std to be valid

        # ── VIX band filter (dual-sided — unique to v10) ──────────────────────
        # Excludes ultra-calm VIX<12 (no VWAP urgency) and crisis VIX>22
        # (sustained directional flow overwhelms reversion).
        vix_in_band = (vix_close >= vix_min) & (vix_close <= vix_max)

        # ── Entry signals ─────────────────────────────────────────────────────
        # buy_ce: mid-depth VWAP dislocation below + moderate volume + normal VIX
        buy_ce = (
            in_session
            & has_warmup
            & vix_in_band
            & (vwap_zscore < -zscore_threshold)
            & (rel_volume > rel_vol_threshold)
        )

        # buy_pe: mid-depth VWAP overshoot above + moderate volume + normal VIX
        buy_pe = (
            in_session
            & has_warmup
            & vix_in_band
            & (vwap_zscore > zscore_threshold)
            & (rel_volume > rel_vol_threshold)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # Stop: 5 pts (~10 NIFTY spot pts at delta 0.5).
            # At z=-2.3 in VIX 12-22, a further 10-spot-pt decline post-entry
            # signals structural move not overshoot; thesis clearly wrong.
            # Intermediate between v1 (4 pts, shallower z) and v3 (7 pts, deeper z).
            stop_points=np.full(n, 5.0),
            # Target: 8 pts (~16 NIFTY spot pts at delta 0.5).
            # First-leg reversion from z=-2.3 to z~=-0.8 covers 20-30 spot pts
            # in normal-VIX sessions; 8 pts captures ~50-60% of expected first leg.
            # R:R = 1:1.6.
            target_points=np.full(n, 8.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,           # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
