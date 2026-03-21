"""vwap_mean_reversion_v5 — VWAP Extreme Reversion in Elevated VIX Regime.

Converted from: trading_strategies/unique_strategies_all/Strategy_94.json
Original: 15-min VWAP z-score mean reversion on Nifty50 stocks with VIX > 15 filter.

Mechanism: On NIFTY, when India VIX exceeds 15, market maker quote widening and elevated
institutional hedging urgency amplify intraday moves — even moderate-volume selling (1.2x
the 4-minute baseline, not panic levels) can drive NIFTY 80-100 spot points below session
VWAP, creating z-scores below -2.7. VWAP-benchmarked algorithms accelerate accumulation
urgency in elevated-VIX environments because benchmark slippage compounds faster.

Differentiation from other VWAP versions:
- v18: dev_momentum trigger, no VIX filter
- v19: microstructure basing, shallow deviations, no VIX filter
- v1: RSI dual-confirmation, VIX < 20 (low-VIX regime)
- v3: panic capitulation, 1.7x volume required, no VIX filter
- v5 (this): elevated VIX regime gate (>15), 1.2x volume bar, VIX does the quality filtering
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "vwap_mean_reversion_v5"
    underlying = "NIFTY"
    session_start_minutes = 570    # 09:30 IST — allow 15 min from open for VWAP stabilization
    session_end_minutes = 920      # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 240             # 20 min warmup for 240-bar rolling std

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_threshold", 2.7, 2.0, 3.5),
            TunableParam("rel_vol_threshold", 1.2, 1.0, 2.0),
            TunableParam("vix_min", 15.0, 12.0, 20.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        zscore_threshold = params.get("zscore_threshold", 2.7)
        rel_vol_threshold = params.get("rel_vol_threshold", 1.2)
        vix_min = params.get("vix_min", 15.0)

        # ── VIX close aligned backward to spot bars ──────────────────────────
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

        # ── Rolling 240-bar (20 min) std of close for z-score denominator ─────
        std_240 = np.zeros(n)
        for i in range(240, n):
            std_240[i] = np.std(close[i - 240:i])

        # ── VWAP z-score ──────────────────────────────────────────────────────
        vwap_zscore = np.zeros(n)
        for i in range(n):
            if std_240[i] > 0:
                vwap_zscore[i] = (close[i] - vwap[i]) / std_240[i]
            # else stays 0 (neutral)

        # ── Relative volume: 4-minute (48 bar) rolling mean baseline ─────────
        rel_volume = np.ones(n)
        for i in range(48, n):
            vol_mean = np.mean(volume[i - 48:i])
            if vol_mean > 0:
                rel_volume[i] = volume[i] / vol_mean
            # else stays 1.0 (neutral)

        # ── Session filter and warmup gate ────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        has_warmup = np.arange(n) >= 240  # need 240 bars for rolling std

        # ── Entry signals ─────────────────────────────────────────────────────
        # buy_ce: extreme VWAP dislocation below + moderate volume + elevated VIX
        buy_ce = (
            in_session
            & has_warmup
            & (vwap_zscore < -zscore_threshold)
            & (rel_volume > rel_vol_threshold)
            & (vix_close > vix_min)
        )

        # buy_pe: extreme VWAP overshoot above + moderate volume + elevated VIX
        buy_pe = (
            in_session
            & has_warmup
            & (vwap_zscore > zscore_threshold)
            & (rel_volume > rel_vol_threshold)
            & (vix_close > vix_min)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # Stop: 5 pts (~10 NIFTY spot pts at delta 0.5). Further 10-pt decline
            # post-entry in VIX>15 regime signals structural breakdown, not overshoot.
            # Target: 9 pts (~18 spot pts). First-leg snap-back in elevated-VIX
            # VWAP reversion covers 20-30 spot pts in 30-90 seconds. R:R = 1:1.8.
            stop_points=np.full(n, 5.0),
            target_points=np.full(n, 9.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,           # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
