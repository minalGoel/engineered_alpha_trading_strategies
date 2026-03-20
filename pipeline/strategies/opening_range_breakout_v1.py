"""Opening Range Breakout — 5-second NIFTY index options strategy.

Mechanism:
  On NIFTY, the 09:15-09:30 window is the information-absorption phase where FII
  pre-positioning, overnight SGX cues, and gap-filling orders compete. Once the OR is
  established at 09:30, NIFTY's large TWAP/VWAP algorithms release directional order flow.
  A breach of OR_high/OR_low with elevated 5-second volume (1.5x OR average) confirmed over
  3 consecutive bars signals institutional conviction, capturing the initial 15-40 spot point
  institutional drive in the first 30-90 seconds after breakout.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "opening_range_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — after OR is established
    session_end_minutes = 720     # 12:00 IST — ORB edge dissipates by midday
    max_trades_per_day = 4
    max_lookback = 200            # covers 15-min OR (180 bars at 5s) + buffer

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_mult", 1.5, 1.0, 3.0),    # volume surge multiplier vs OR avg
            TunableParam("vix_max", 25.0, 18.0, 32.0),  # max India VIX for valid ORB
            TunableParam("stop_pts", 5.0, 3.0, 10.0),
            TunableParam("target_pts", 8.0, 5.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill then extract arrays
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        vol_mult = params.get("vol_mult", 1.5)
        vix_max = params.get("vix_max", 25.0)
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
        # OR is a FIXED TIME WINDOW — 180 bars at 5s regardless of bar frequency.
        OR_START = 555   # 09:15 IST in minutes from midnight
        OR_END = 570     # 09:30 IST (exclusive)

        or_high = np.zeros(n)
        or_low = np.zeros(n)
        or_vol_avg = np.zeros(n)

        for d in np.unique(day_id):
            day_mask = day_id == d
            day_indices = np.where(day_mask)[0]

            or_bar_mask = day_mask & (time_min >= OR_START) & (time_min < OR_END)
            or_indices = np.where(or_bar_mask)[0]

            if len(or_indices) == 0:
                continue

            day_or_high = np.max(high[or_indices])
            day_or_low = np.min(low[or_indices])
            day_or_vol_avg = np.mean(volume[or_indices]) if len(or_indices) > 0 else 0.0

            or_high[day_indices] = day_or_high
            or_low[day_indices] = day_or_low
            or_vol_avg[day_indices] = day_or_vol_avg

        # ── OR range validity (0.1% to 2.0%) ───────────────────────────────────
        or_range_pct = np.where(or_low > 0, (or_high - or_low) / or_low * 100.0, 0.0)
        or_range_ok = (or_range_pct > 0.1) & (or_range_pct < 2.0)

        # ── Rolling 3-bar (15s) volume average ─────────────────────────────────
        vol_3 = np.zeros(n)
        for i in range(3, n):
            vol_3[i] = np.mean(volume[i - 3:i])

        # ── 3-bar persistence filter (replaces 1-min body check from original) ─
        # Three consecutive 5s closes above/below OR boundary confirms breakout.
        above_or = close > or_high
        below_or = close < or_low

        persist_above = np.zeros(n, dtype=bool)
        persist_below = np.zeros(n, dtype=bool)
        for i in range(3, n):
            persist_above[i] = above_or[i] and above_or[i - 1] and above_or[i - 2]
            persist_below[i] = below_or[i] and below_or[i - 1] and below_or[i - 2]

        # ── Volume confirmation ─────────────────────────────────────────────────
        vol_confirm = (vol_3 > vol_mult * or_vol_avg) & (or_vol_avg > 0)

        # ── Combined filters ───────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = vix_close < vix_max

        buy_ce = in_session & persist_above & vol_confirm & vix_ok & or_range_ok
        buy_pe = in_session & persist_below & vol_confirm & vix_ok & or_range_ok

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
