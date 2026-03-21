"""rbi_policy_reaction_v1 — BANKNIFTY 5s index options strategy.

Thesis (Strategy_260): RBI monetary policy announcements (6 per year, at 10:00
IST) create a predictable intraday pattern on BANKNIFTY:
  1. Pre-announcement compression (09:15–10:00): narrow range, low volume
  2. Knee-jerk reaction (10:00–10:15): sharp 50-200 bps directional move
  3. Fade/continuation (10:15–11:00): varies by surprise vs expectation

Since we lack RBI calendar data, we replicate the TIME-OF-DAY pattern:
BANKNIFTY around 10:00 IST regularly shows a volatility regime shift. We
trade the 10:00 momentum burst by entering on a confirmed 3-bar momentum
move starting at 10:00–10:10 IST, which captures both RBI days and the
general liquidity surge that occurs daily around 10:00 IST.

Source: trading_strategies/unique_strategies_all/Strategy_260.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam

_BURST_START = 600   # 10:00 IST
_BURST_END   = 610   # 10:10 IST


class Strategy(BaseStrategy):
    name = "rbi_policy_reaction_v1"
    underlying = "BANKNIFTY"
    session_start_minutes = _BURST_START
    session_end_minutes = 660     # 11:00 IST
    max_trades_per_day = 2
    max_lookback = 300            # 25-min warmup to compute pre-compression range

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("compression_bars", 180.0, 120.0, 300.0),  # pre-announcement window in bars
            TunableParam("momentum_bars", 3.0, 2.0, 6.0),
            TunableParam("vol_contraction_ratio", 0.7, 0.4, 0.9),
            TunableParam("stop_pts", 8.0, 5.0, 15.0),
            TunableParam("target_pts", 14.0, 8.0, 25.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        compression_bars = int(params.get("compression_bars", 180))
        momentum_bars = int(params.get("momentum_bars", 3))
        vol_contraction_ratio = float(params.get("vol_contraction_ratio", 0.7))
        stop_pts = float(params.get("stop_pts", 8.0))
        target_pts = float(params.get("target_pts", 14.0))

        # ── Rolling ATR-like range (high - low) ───────────────────────────────
        bar_range = high - low

        # Rolling average range over compression_bars window
        avg_range = np.zeros(n)
        for i in range(compression_bars, n):
            avg_range[i] = np.mean(bar_range[i - compression_bars:i])
        for i in range(min(compression_bars, n)):
            avg_range[i] = bar_range[i]

        # ── Compression detection: current range < vol_contraction_ratio * avg ─
        compressed = bar_range < (vol_contraction_ratio * avg_range)

        # ── N-bar momentum ────────────────────────────────────────────────────
        mom_up = np.zeros(n, dtype=bool)
        mom_down = np.zeros(n, dtype=bool)
        mb = max(1, momentum_bars)
        for i in range(mb, n):
            if day_id[i] == day_id[i - mb]:
                mom_up[i] = close[i] > close[i - mb]
                mom_down[i] = close[i] < close[i - mb]

        # ── Pre-burst compression: was price compressed in the N bars before burst?
        was_compressed = np.zeros(n, dtype=bool)
        for i in range(compression_bars, n):
            # Check if most bars in last compression_bars were compressed
            seg = compressed[i - compression_bars:i]
            was_compressed[i] = np.sum(seg) > 0.5 * len(seg)

        in_burst = (time_min >= _BURST_START) & (time_min < _BURST_END)

        buy_ce = in_burst & was_compressed & mom_up
        buy_pe = in_burst & was_compressed & mom_down

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=36,
            max_trades_per_day=self.max_trades_per_day,
        )
