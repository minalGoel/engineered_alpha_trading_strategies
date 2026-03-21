"""orb_multi_stock_v1 — NIFTY 5s index options strategy.

Thesis (Strategy_235): Individual stock ORB breakouts have ~52% win rate, but
a portfolio approach across 5-8 uncorrelated stocks improves the daily win rate
to ~70%+ through diversification. When multiple stocks break their ORBs in the
same direction, it signals a market-wide directional bias.

Adaptation: Without individual stock data, we proxy the "multi-stock ORB
consensus" using a multi-timeframe ORB confirmation on NIFTY itself. We compute
three ORBs at different windows (5-min, 10-min, 15-min) and only enter when all
three are broken in the same direction — analogous to multiple stocks confirming
the same direction.

Source: trading_strategies/unique_strategies_all/Strategy_235.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "orb_multi_stock_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — after all three ORBs formed
    session_end_minutes = 840     # 14:00 IST
    max_trades_per_day = 3
    max_lookback = 200

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("stop_pts", 5.0, 3.0, 10.0),
            TunableParam("target_pts", 10.0, 6.0, 18.0),
            TunableParam("volume_ratio", 1.2, 0.8, 2.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 10.0))
        vol_ratio_thresh = float(params.get("volume_ratio", 1.2))

        # ── Three ORBs: 5-min (60 bars), 10-min (120 bars), 15-min (180 bars) ─
        # Window end times: 09:20, 09:25, 09:30
        orb_windows = [
            (555, 560),   # 5-min: 09:15–09:20
            (555, 565),   # 10-min: 09:15–09:25
            (555, 570),   # 15-min: 09:15–09:30
        ]

        orb_highs = []
        orb_lows = []
        for orb_start, orb_end in orb_windows:
            orb_h = np.full(n, np.nan)
            orb_l = np.full(n, np.nan)
            for d in np.unique(day_id):
                day_mask = day_id == d
                day_idx = np.where(day_mask)[0]
                orb_mask = day_mask & (time_min >= orb_start) & (time_min < orb_end)
                orb_idx = np.where(orb_mask)[0]
                if len(orb_idx) == 0:
                    continue
                orb_h[day_idx] = np.nanmax(high[orb_idx])
                orb_l[day_idx] = np.nanmin(low[orb_idx])
            # Forward-fill
            last_h = close[0] if not np.isnan(close[0]) else 0.0
            last_l = last_h
            for i in range(n):
                if not np.isnan(orb_h[i]):
                    last_h = orb_h[i]
                    last_l = orb_l[i]
                else:
                    orb_h[i] = last_h
                    orb_l[i] = last_l
            orb_highs.append(orb_h)
            orb_lows.append(orb_l)

        # ── All three ORBs broken in the same direction ───────────────────────
        all_above = np.ones(n, dtype=bool)
        all_below = np.ones(n, dtype=bool)
        for orb_h, orb_l in zip(orb_highs, orb_lows):
            all_above &= (close > orb_h)
            all_below &= (close < orb_l)

        # ── Volume filter ─────────────────────────────────────────────────────
        vol_avg = np.ones(n)
        for i in range(1, n):
            start = max(0, i - 60)
            seg = volume[start:i]
            pos = seg[seg > 0]
            vol_avg[i] = np.mean(pos) if len(pos) > 0 else 1.0
        vol_above = volume > (vol_ratio_thresh * vol_avg)

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        buy_ce = in_session & all_above & vol_above
        buy_pe = in_session & all_below & vol_above

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
