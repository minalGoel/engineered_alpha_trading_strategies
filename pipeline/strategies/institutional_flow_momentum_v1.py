"""Institutional Flow Momentum Strategy (5-second NIFTY/BANKNIFTY index options).

Detects large volume spikes (4-8x 10-minute rolling average) that are also directionally
impactful (bar return > 0.03%) and confirm VWAP direction. On NIFTY/BANKNIFTY, these
spikes signal institutional block execution or algo momentum ignition, producing 15-45
second follow-through. We enter on the bar immediately after the spike bar.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "institutional_flow_momentum_v1"
    underlying = "BOTH"
    session_start_minutes = 560    # 09:20 IST — skip opening auction noise
    session_end_minutes = 920      # 15:20 IST — avoid EOD thin-book spikes
    max_trades_per_day = 8
    max_lookback = 120             # 10 min warmup for volume SMA

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_spike_threshold", 4.0, 2.5, 8.0),
            TunableParam("bar_return_threshold", 0.03, 0.01, 0.08),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
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

        # ── Extract arrays ──
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
        day_id = spot_df["day_id"].to_numpy()

        vol_spike_thr = params.get("vol_spike_threshold", 4.0)
        bar_ret_thr = params.get("bar_return_threshold", 0.03)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # ── VWAP (cumulative typical price × volume from session open, per day) ──
        typical_price = (high + low + close) / 3.0
        vwap = np.zeros(n)
        running_tpv = 0.0
        running_vol = 0.0
        prev_day = day_id[0] - 1
        for i in range(n):
            if day_id[i] != prev_day:
                running_tpv = 0.0
                running_vol = 0.0
                prev_day = day_id[i]
            running_tpv += typical_price[i] * volume[i]
            running_vol += volume[i]
            vwap[i] = running_tpv / running_vol if running_vol > 0.0 else close[i]

        # ── Volume spike ratio: volume / rolling 120-bar (10-min) SMA ──
        vol_sma_lookback = 120
        vol_sma = np.zeros(n)
        for i in range(vol_sma_lookback, n):
            window_vol = volume[i - vol_sma_lookback:i]
            mean_vol = np.mean(window_vol)
            vol_sma[i] = mean_vol if mean_vol > 0.0 else 1.0
        # Before warmup, use a short expanding mean to avoid zeros
        for i in range(1, min(vol_sma_lookback, n)):
            m = np.mean(volume[:i])
            vol_sma[i] = m if m > 0.0 else 1.0
        vol_sma[0] = volume[0] if volume[0] > 0.0 else 1.0

        vol_spike_ratio = np.where(vol_sma > 0.0, volume / vol_sma, 0.0)

        # ── Bar return (directional price impact within the spike bar) ──
        # 0.03% threshold ≈ 6 NIFTY pts or 13 BANKNIFTY pts in 5 seconds
        bar_return_pct = np.where(open_ > 0.0, (close - open_) / open_ * 100.0, 0.0)

        # ── Spike flag on bar i; signal fires on bar i+1 (next bar) ──
        spike_bull = (vol_spike_ratio > vol_spike_thr) & (bar_return_pct > bar_ret_thr) & (close > vwap)
        spike_bear = (vol_spike_ratio > vol_spike_thr) & (bar_return_pct < -bar_ret_thr) & (close < vwap)

        # Shift by 1: entry on the bar AFTER the spike bar
        buy_ce_raw = np.zeros(n, dtype=bool)
        buy_pe_raw = np.zeros(n, dtype=bool)
        buy_ce_raw[1:] = spike_bull[:-1]
        buy_pe_raw[1:] = spike_bear[:-1]

        # ── Session filter ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed_up = np.arange(n) >= vol_sma_lookback

        buy_ce = buy_ce_raw & in_session & warmed_up
        buy_pe = buy_pe_raw & in_session & warmed_up

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=12,        # 60 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
