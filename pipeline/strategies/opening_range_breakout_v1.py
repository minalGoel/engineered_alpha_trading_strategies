"""Opening Range Breakout — 5-second NIFTY index options strategy.

Mechanism:
  On NIFTY, the 09:15-09:30 window is the overnight information absorption phase where
  FII pre-open imbalances, SGX cues, and domestic institutional adjustments settle into
  resting limit orders that define the opening range. When NIFTY sustains 3 consecutive
  5-second closes above the OR high (15 seconds of unbroken price action), buy-side
  institutional flow has overwhelmed the OR supply cluster, triggering stop-loss cascades
  from range-bound players and initiating a mechanical momentum burst that we capture
  before 1-minute-bar participants can react.

Converted from: trading_strategies/unique_strategies_all/Strategy_3.json
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "opening_range_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — after 15-min OR is fully formed
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 2
    max_lookback = 200            # 180 bars (15-min OR) + 20-bar buffer

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_max", 25.0, 15.0, 35.0),
            TunableParam("persist_bars", 3.0, 2.0, 6.0),
            TunableParam("stop_pts", 5.0, 3.0, 10.0),
            TunableParam("target_pts", 8.0, 5.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill then extract arrays
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        vix_max = params.get("vix_max", 25.0)
        persist_bars = int(params.get("persist_bars", 3))
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        # ── VIX filter ─────────────────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Per-day Opening Range (09:15-09:30, time_minutes 555-569) ──────────
        # Fixed time window — 15 calendar minutes = 180 bars at 5s. NOT scaled.
        OR_START = 555   # 09:15 IST
        OR_END = 570     # 09:30 IST (exclusive)

        or_high = np.zeros(n)
        or_low = np.zeros(n)

        for d in np.unique(day_id):
            day_mask = day_id == d
            day_indices = np.where(day_mask)[0]

            or_bar_indices = np.where(day_mask & (time_min >= OR_START) & (time_min < OR_END))[0]
            if len(or_bar_indices) == 0:
                continue

            day_or_high = np.max(high[or_bar_indices])
            day_or_low = np.min(low[or_bar_indices])

            or_high[day_indices] = day_or_high
            or_low[day_indices] = day_or_low

        # OR validity: both levels set and range is positive
        or_valid = (or_high > 0) & (or_low > 0) & (or_high > or_low)

        # ── Breakout conditions ────────────────────────────────────────────────
        above_or = close > or_high
        below_or = close < or_low

        # ── Persistence filter: N consecutive bars above/below OR level ────────
        # Replaces the original's 1-min bar body check at 5-second resolution.
        persist_above = np.zeros(n, dtype=bool)
        persist_below = np.zeros(n, dtype=bool)
        consec_above = np.zeros(n, dtype=np.int32)
        consec_below = np.zeros(n, dtype=np.int32)

        for i in range(1, n):
            consec_above[i] = consec_above[i - 1] + 1 if above_or[i] else 0
            consec_below[i] = consec_below[i - 1] + 1 if below_or[i] else 0

        persist_above = consec_above >= persist_bars
        persist_below = consec_below >= persist_bars

        # ── Session and composite filters ──────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = vix_close < vix_max

        buy_ce = in_session & or_valid & vix_ok & persist_above
        buy_pe = in_session & or_valid & vix_ok & persist_below

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,          # 120 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
