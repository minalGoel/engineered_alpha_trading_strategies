"""price_impact_reversion_v1 — Fade NIFTY volume-spike impact bars.

When a 5-second NIFTY bar shows 2.5x+ the rolling volume average AND a sharp
price move (±15 bps) that has pushed the index away from session VWAP, we
fade the move. Institutional block executions that overshoot fair value attract
VWAP-benchmarked algorithms and market makers who provide liquidity in the
opposite direction, causing 40-60% reversion within 30-90 seconds.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "price_impact_reversion_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip opening auction distortion
    session_end_minutes = 915     # 15:15 IST — avoid EOD illiquidity
    max_trades_per_day = 8
    max_lookback = 120            # 10-min warmup for rolling volume mean

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_spike_threshold", 2.5, 1.5, 4.0),
            TunableParam("impact_bps_threshold", 15.0, 8.0, 25.0),
            TunableParam("vwap_dev_threshold", 12.0, 6.0, 20.0),
            TunableParam("vix_max", 22.0, 15.0, 28.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays (forward-fill NaN in Polars before converting) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        vol_spike_thr = params.get("vol_spike_threshold", 2.5)
        impact_bps_thr = params.get("impact_bps_threshold", 15.0)
        vwap_dev_thr = params.get("vwap_dev_threshold", 12.0)
        vix_max = params.get("vix_max", 22.0)

        # ── VIX filter ──
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Rolling mean volume over 60 bars (5 min) ──
        # Uses a running cumsum approach for efficiency — O(n)
        vol_mean_60 = np.zeros(n)
        running_sum = 0.0
        window = 60
        for i in range(n):
            running_sum += volume[i]
            if i >= window:
                running_sum -= volume[i - window]
                vol_mean_60[i] = running_sum / window
            elif i > 0:
                vol_mean_60[i] = running_sum / i
            else:
                vol_mean_60[i] = volume[i] if volume[i] > 0 else 1.0

        # ── Volume spike ratio ──
        safe_vol_mean = np.where(vol_mean_60 > 0, vol_mean_60, 1.0)
        vol_spike = volume / safe_vol_mean

        # ── Bar return in bps (open-to-close within the 5s bar) ──
        safe_open = np.where(open_ > 0, open_, close)
        bar_return_bps = (close - safe_open) / safe_open * 10000.0

        # ── Session VWAP (cumulative, reset each day) ──
        # Uses typical price = (high + low + close) / 3
        vwap = np.zeros(n)
        cum_pv = 0.0
        cum_vol = 0.0
        current_day = -1
        for i in range(n):
            d = day_id[i]
            if d != current_day:
                cum_pv = 0.0
                cum_vol = 0.0
                current_day = d
            typical_price = (high[i] + low[i] + close[i]) / 3.0
            cum_pv += typical_price * volume[i]
            cum_vol += volume[i]
            vwap[i] = cum_pv / cum_vol if cum_vol > 0 else close[i]

        # ── VWAP deviation in bps ──
        safe_vwap = np.where(vwap > 0, vwap, close)
        vwap_dev_bps = (close - safe_vwap) / safe_vwap * 10000.0

        # ── Session filter ──
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── VIX regime filter ──
        vix_ok = vix_close < vix_max

        # ── Signals ──
        # Large SELL impact bar (sharp down + volume spike + below VWAP) → fade → buy CE
        sell_impact = (
            in_session
            & vix_ok
            & (vol_spike >= vol_spike_thr)
            & (bar_return_bps <= -impact_bps_thr)
            & (vwap_dev_bps <= -vwap_dev_thr)
        )

        # Large BUY impact bar (sharp up + volume spike + above VWAP) → fade → buy PE
        buy_impact = (
            in_session
            & vix_ok
            & (vol_spike >= vol_spike_thr)
            & (bar_return_bps >= impact_bps_thr)
            & (vwap_dev_bps >= vwap_dev_thr)
        )

        return OptionSignals(
            buy_ce=sell_impact,                          # fade the sell impact — bullish
            buy_pe=buy_impact,                           # fade the buy impact — bearish
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 4.0),                # 4 pts = ~8 spot pts at delta 0.5
            target_points=np.full(n, 6.0),              # 6 pts = ~12 spot pts reversion target
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,                           # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
