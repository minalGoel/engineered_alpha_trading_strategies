"""
gap_fade_reversion_v116 — NIFTY 5s index options strategy.

When NIFTY gaps >0.5% overnight, waits for a 1-minute reversal candle
(close in upper 40% of 1-min range + net positive 1-min return on gap-downs)
before entering the fade. Targets the post-absorption impulse leg (60-120s hold).

Differentiated from v103: v103 catches the FIRST fade tick within 09:15-09:35
using 3-bar momentum; v116 catches the CONFIRMED REVERSAL CANDLE within 09:20-09:45
using 1-minute range-position absorption structure.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "gap_fade_reversion_v116"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST — skip initial noisy open
    session_end_minutes = 585     # 09:45 IST — gap fades resolve by then
    max_trades_per_day = 3
    max_lookback = 24             # 12 bars for indicators + buffer

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_threshold", 0.005, 0.003, 0.010),
            TunableParam("reversal_threshold", 0.0003, 0.0001, 0.0008),
            TunableParam("range_pos_bull", 0.60, 0.50, 0.75),
            TunableParam("range_pos_bear", 0.40, 0.25, 0.50),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # --- Extract arrays (forward-fill NaN before numpy conversion) ---
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        open_arr = spot_df["open"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # --- Parameters ---
        gap_threshold = params.get("gap_threshold", 0.005)
        reversal_threshold = params.get("reversal_threshold", 0.0003)
        range_pos_bull = params.get("range_pos_bull", 0.60)
        range_pos_bear = params.get("range_pos_bear", 0.40)

        # --- VIX filter ---
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # --- Day-level: compute day_open and prev_day_close per bar ---
        # day_open = first bar's open of each day
        # prev_day_close = last bar's close of the previous day
        day_open = np.zeros(n)
        prev_day_close = np.zeros(n)

        day_open_map: dict[int, float] = {}
        day_last_close_map: dict[int, float] = {}

        # Single pass: record first open and last close per day
        for i in range(n):
            d = int(day_id[i])
            if d not in day_open_map:
                day_open_map[d] = float(open_arr[i])
            day_last_close_map[d] = float(close[i])

        unique_days = sorted(day_open_map.keys())

        for i in range(n):
            d = int(day_id[i])
            day_open[i] = day_open_map[d]
            # Find the previous day
            idx = unique_days.index(d)
            if idx > 0:
                prev_d = unique_days[idx - 1]
                prev_day_close[i] = day_last_close_map[prev_d]
            else:
                prev_day_close[i] = np.nan

        # gap_pct = (day_open - prev_day_close) / prev_day_close
        with np.errstate(invalid="ignore", divide="ignore"):
            gap_pct = np.where(
                prev_day_close > 0,
                (day_open - prev_day_close) / prev_day_close,
                0.0,
            )

        # --- 1-minute (12-bar) net return — reversal candle signal ---
        net_return_12 = np.zeros(n)
        for i in range(12, n):
            if close[i - 12] > 0:
                net_return_12[i] = (close[i] - close[i - 12]) / close[i - 12]

        # --- Range position within last 12 bars ---
        # (close - min_low_12) / (max_high_12 - min_low_12)
        # 0 = at 1-min low, 1 = at 1-min high; >0.6 = bullish absorption
        range_position = np.full(n, 0.5)
        for i in range(12, n):
            lo = np.min(low[i - 11 : i + 1])
            hi = np.max(high[i - 11 : i + 1])
            rng = hi - lo
            if rng > 0:
                range_position[i] = (close[i] - lo) / rng
            # else stays 0.5 (neutral)

        # --- Session and VIX filter ---
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )
        vix_ok = (vix_close >= 10.0) & (vix_close <= 25.0)

        # --- Entry signals ---
        # buy_ce: gap-down exists AND 1-min reversal candle confirmed (bullish absorption)
        buy_ce = (
            in_session
            & vix_ok
            & (gap_pct < -gap_threshold)
            & (net_return_12 > reversal_threshold)
            & (range_position > range_pos_bull)
        )

        # buy_pe: gap-up exists AND 1-min reversal candle confirmed (bearish distribution)
        buy_pe = (
            in_session
            & vix_ok
            & (gap_pct > gap_threshold)
            & (net_return_12 < -reversal_threshold)
            & (range_position < range_pos_bear)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 5.0),
            target_points=np.full(n, 8.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,
            max_trades_per_day=self.max_trades_per_day,
        )
