"""banknifty_lead_lag_v1 — BANKNIFTY 5s index options strategy.

Thesis (Strategy_75): BANK NIFTY leads NIFTY 50 by 1-5 minutes during
directional moves because banking stocks are highest-weight and institutional
flows hit banks first. Adapted for 5s bars: when BANKNIFTY makes a sharp
1-minute (12-bar) move with strong autocorrelation (current momentum aligns
with recent trend), it signals committed institutional flow that continues
for 30-90 seconds.

At 5-second resolution, the "lead" effect becomes: BANKNIFTY's short-term
momentum burst (strong 12-bar return with persistent direction) predicts
continuation for the next 24-36 bars before the broader market catches up.

Source: trading_strategies/unique_strategies_all/Strategy_75.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "banknifty_lead_lag_v1"
    underlying = "BANKNIFTY"
    session_start_minutes = 570   # 09:30 IST
    session_end_minutes = 900     # 15:00 IST
    max_trades_per_day = 6
    max_lookback = 120

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("momentum_threshold", 0.0012, 0.0006, 0.003),
            TunableParam("lookback_bars", 12.0, 6.0, 24.0),
            TunableParam("persistence_bars", 3.0, 2.0, 6.0),
            TunableParam("stop_pts", 10.0, 6.0, 18.0),
            TunableParam("target_pts", 15.0, 9.0, 25.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        mom_thresh = float(params.get("momentum_threshold", 0.0012))
        lb = int(params.get("lookback_bars", 12))
        pb = int(params.get("persistence_bars", 3))
        stop_pts = float(params.get("stop_pts", 10.0))
        target_pts = float(params.get("target_pts", 15.0))

        # ── N-bar return (proxy for "sharp BANKNIFTY move") ──────────────────
        roc = np.zeros(n)
        for i in range(lb, n):
            if day_id[i] == day_id[i - lb] and close[i - lb] > 0:
                roc[i] = (close[i] - close[i - lb]) / close[i - lb]

        # ── Persistence filter: last pb bars all in same direction ───────────
        persist_up = np.zeros(n, dtype=bool)
        persist_down = np.zeros(n, dtype=bool)
        for i in range(pb, n):
            same_day = all(day_id[i - j] == day_id[i] for j in range(pb))
            if same_day:
                persist_up[i] = all(close[i - j] >= close[i - j - 1] for j in range(pb - 1))
                persist_down[i] = all(close[i - j] <= close[i - j - 1] for j in range(pb - 1))

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # Strong upward burst + persistent → buy CE (momentum continuation)
        buy_ce = in_session & (roc > mom_thresh) & persist_up
        # Strong downward burst + persistent → buy PE
        buy_pe = in_session & (roc < -mom_thresh) & persist_down

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=30,
            max_trades_per_day=self.max_trades_per_day,
        )
