"""volume_spike_exhaustion_fade — NIFTY index options, 5-second bars.

Thesis: When a 5-second bar shows a volume spike (4x+ recent 5-minute average)
but the price body is tiny relative to the bar's range (doji), large institutional
order flow was absorbed by opposing resting limit orders. The exhaustion unwinds
as the aggressive participants are now flat, producing a 15-90 second reversion.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "volume_spike_exhaustion_fade"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip opening auction noise
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 60             # 5-min warmup for rolling volume SMA

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_spike_mult", 4.0, 2.5, 6.0),
            TunableParam("range_body_ratio_thresh", 0.3, 0.15, 0.5),
            TunableParam("stop_pts", 4.0, 3.0, 8.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Parameters ────────────────────────────────────────────────────────
        vol_spike_mult = params.get("vol_spike_mult", 4.0)
        range_body_thresh = params.get("range_body_ratio_thresh", 0.3)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # ── Raw arrays (forward-fill NaN in Polars before numpy) ──────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = (
            spot_df["volume"]
            .cast(pl.Float64)
            .fill_null(0.0)
            .to_numpy()
        )
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Indicator 1: 60-bar rolling volume SMA (5-min baseline) ──────────
        vol_sma = np.zeros(n)
        for i in range(60, n):
            vol_sma[i] = np.mean(volume[i - 60:i])

        # ── Indicator 2: Volume spike flag ────────────────────────────────────
        vol_spike = np.zeros(n, dtype=bool)
        for i in range(60, n):
            if vol_sma[i] > 0:
                vol_spike[i] = volume[i] > vol_spike_mult * vol_sma[i]

        # ── Indicator 3: Range-body ratio (doji check) ────────────────────────
        # Guard against zero-range bars (high == low) which can appear in thin periods
        bar_range = high - low
        body = np.abs(close - open_)
        range_body_ratio = np.where(bar_range > 0.5, body / bar_range, 1.0)
        is_doji = range_body_ratio < range_body_thresh

        # ── Indicator 4: 1-bar return for directional assignment ───────────────
        ret_1 = np.zeros(n)
        ret_1[1:] = close[1:] - close[:-1]

        # ── Session filter ─────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Signals ────────────────────────────────────────────────────────────
        # Down move + volume climax + doji → seller exhaustion → buy CE (fade the drop)
        buy_ce = in_session & vol_spike & is_doji & (ret_1 < 0)
        # Up move + volume climax + doji → buyer exhaustion → buy PE (fade the rally)
        buy_pe = in_session & vol_spike & is_doji & (ret_1 > 0)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
