"""Index Momentum Continuation: 5-minute NIFTY return > threshold predicts 30-120s continuation.

When NIFTY's 5-minute return exceeds 0.2%, institutional TWAP/VWAP algorithms are typically
40-60% filled — remaining unfilled lots continue pushing price for another 30-90 seconds.
A 1-minute confirmation filter removes stale entries where momentum has already reversed.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "index_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST — skip pre-open noise
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 10
    max_lookback = 72             # warmup: 60 bars (5-min signal) + 12 bars buffer = 72 bars (6 min)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("return_threshold", 0.20, 0.10, 0.40),  # % — 5-min return trigger
            TunableParam("stop_pts", 4.0, 2.0, 7.0),
            TunableParam("target_pts", 7.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)
        day_id = spot_df["day_id"].to_numpy().astype(np.int32)

        return_threshold = params.get("return_threshold", 0.20)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # ── 5-minute return (60 bars × 5s = 5 min): primary trigger ──
        # Same time window as original strategy — not scaled, it's a FIXED TIME WINDOW.
        ret_60 = np.zeros(n)
        for i in range(60, n):
            if day_id[i] == day_id[i - 60] and close[i - 60] > 0:
                ret_60[i] = (close[i] - close[i - 60]) / close[i - 60] * 100.0

        # ── 1-minute return (12 bars × 5s = 60s): confirmation filter ──
        # Removes stale entries where the recent 1-minute direction contradicts the 5-min trend.
        ret_12 = np.zeros(n)
        for i in range(12, n):
            if day_id[i] == day_id[i - 12] and close[i - 12] > 0:
                ret_12[i] = close[i] - close[i - 12]

        # ── Session and warmup filters ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= self.max_lookback

        # ── Entry signals ──
        # Bullish: 5-min return > threshold AND last 60s still moving up
        buy_ce = in_session & warmed & (ret_60 > return_threshold) & (ret_12 > 0)

        # Bearish: 5-min return < -threshold AND last 60s still moving down
        buy_pe = in_session & warmed & (ret_60 < -return_threshold) & (ret_12 < 0)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,           # 24 × 5s = 120s max hold
            max_trades_per_day=self.max_trades_per_day,
        )
