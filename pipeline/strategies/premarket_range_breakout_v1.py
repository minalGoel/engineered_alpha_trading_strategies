"""premarket_range_breakout_v1 — NIFTY 5s index options strategy.

Thesis (Strategy_185): The NSE pre-open call auction (09:00-09:08 IST) sets an
indicative price range. Since we don't have pre-open data in the 5s feed, we
use the first 90-second window after session open (09:15-09:16:30, 18 bars)
as a proxy for the initial price discovery range. Breakouts from this
micro-range with volume confirmation indicate committed directional flow.

Source: trading_strategies/unique_strategies_all/Strategy_185.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam

# Pre-open proxy: 09:15:00 to 09:16:30 (18 bars × 5s)
_PROXY_START = 555   # 09:15 IST
_PROXY_END   = 557   # 09:16:40 — 2 minutes = 24 bars, use time < 557


class Strategy(BaseStrategy):
    name = "premarket_range_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 557   # Enter after 2-min proxy range forms
    session_end_minutes = 660     # 11:00 IST
    max_trades_per_day = 3
    max_lookback = 30

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("breakout_buffer", 0.5, 0.2, 1.5),
            TunableParam("stop_pts", 5.0, 3.0, 9.0),
            TunableParam("target_pts", 10.0, 6.0, 18.0),
            TunableParam("volume_ratio", 1.2, 0.8, 2.5),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        breakout_buffer = float(params.get("breakout_buffer", 0.5))
        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 10.0))
        vol_ratio_thresh = float(params.get("volume_ratio", 1.2))

        # ── Build proxy pre-open range (first 2 min of session) ─────────────
        proxy_high = np.full(n, np.nan)
        proxy_low = np.full(n, np.nan)

        for d in np.unique(day_id):
            day_mask = day_id == d
            day_idx = np.where(day_mask)[0]
            proxy_mask = day_mask & (time_min >= _PROXY_START) & (time_min < _PROXY_END)
            proxy_idx = np.where(proxy_mask)[0]
            if len(proxy_idx) == 0:
                # Fall back to first bar of the day
                if len(day_idx) > 0:
                    proxy_high[day_idx] = high[day_idx[0]]
                    proxy_low[day_idx] = low[day_idx[0]]
                continue
            proxy_high[day_idx] = np.nanmax(high[proxy_idx])
            proxy_low[day_idx] = np.nanmin(low[proxy_idx])

        last_h = close[0] if not np.isnan(close[0]) else 0.0
        last_l = last_h
        for i in range(n):
            if not np.isnan(proxy_high[i]):
                last_h = proxy_high[i]
                last_l = proxy_low[i]
            else:
                proxy_high[i] = last_h
                proxy_low[i] = last_l

        # ── Rolling 60-bar volume average ────────────────────────────────────
        vol_avg = np.ones(n)
        for i in range(1, n):
            start = max(0, i - 60)
            seg = volume[start:i]
            pos = seg[seg > 0]
            vol_avg[i] = np.mean(pos) if len(pos) > 0 else 1.0
        vol_above = volume > (vol_ratio_thresh * vol_avg)

        # ── Breakout signals (with small buffer to avoid noise) ───────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        buy_ce = in_session & (close > proxy_high + breakout_buffer) & vol_above
        buy_pe = in_session & (close < proxy_low - breakout_buffer) & vol_above

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
