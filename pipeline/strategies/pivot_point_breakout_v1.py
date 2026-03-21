"""Pivot Point Breakout — NIFTY 5-second index options strategy.

Mechanism: Previous-session R1/S1 pivot levels are pre-loaded by retail platforms and
institutional execution algos. When NIFTY crosses R1 upward with VWAP above and a volume
surge, the collective algo+retail response creates a 15-25 spot point burst within 30-90s.
We enter at the exact crossover bar and exit after capturing the first leg of continuation.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "pivot_point_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 560    # 09:20 IST
    session_end_minutes = 920      # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 240             # 20 min warmup

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_surge_mult", 1.5, 1.0, 3.0),
            TunableParam("stop_pts", 5.0, 2.0, 8.0),
            TunableParam("target_pts", 8.0, 4.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        vol_surge_mult = params.get("vol_surge_mult", 1.5)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high_arr = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low_arr = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Compute daily OHLC for pivot calculation ──
        daily_df = (
            spot_df.group_by("day_id")
            .agg([
                pl.col("high").max().alias("day_high"),
                pl.col("low").min().alias("day_low"),
                pl.col("close").last().alias("day_close"),
            ])
            .sort("day_id")
        )

        day_ids_sorted = daily_df["day_id"].to_numpy()
        day_highs = daily_df["day_high"].fill_null(strategy="forward").to_numpy()
        day_lows = daily_df["day_low"].fill_null(strategy="forward").to_numpy()
        day_closes = daily_df["day_close"].fill_null(strategy="forward").to_numpy()

        # Assign prev-day pivot R1/S1 to each bar
        r1 = np.full(n, np.nan)
        s1 = np.full(n, np.nan)

        for i in range(len(day_ids_sorted)):
            if i == 0:
                continue  # No previous day available
            prev_h = day_highs[i - 1]
            prev_l = day_lows[i - 1]
            prev_c = day_closes[i - 1]
            pivot = (prev_h + prev_l + prev_c) / 3.0
            r1_val = 2.0 * pivot - prev_l
            s1_val = 2.0 * pivot - prev_h
            mask = day_id == day_ids_sorted[i]
            r1[mask] = r1_val
            s1[mask] = s1_val

        # ── VWAP (cumulative per session day) ──
        typical = (high_arr + low_arr + close) / 3.0
        vwap = np.empty(n)
        vwap[:] = close  # fallback

        unique_days = np.unique(day_id)
        for did in unique_days:
            idxs = np.where(day_id == did)[0]
            if len(idxs) == 0:
                continue
            tp = typical[idxs]
            vol = volume[idxs]
            cum_tp_vol = np.cumsum(tp * vol)
            cum_vol = np.cumsum(vol)
            with np.errstate(divide="ignore", invalid="ignore"):
                vwap_day = np.where(cum_vol > 0, cum_tp_vol / cum_vol, tp)
            vwap[idxs] = vwap_day

        # ── Rolling 60-bar (5-min) volume moving average ──
        vol_ma = np.zeros(n)
        for i in range(1, n):
            start = max(0, i - 60)
            vol_ma[i] = np.mean(volume[start:i])

        # ── Entry conditions ──
        # Pivot levels: treat NaN as no-trade (set to extreme values)
        r1_clean = np.where(np.isnan(r1), np.inf, r1)
        s1_clean = np.where(np.isnan(s1), -np.inf, s1)
        has_r1 = ~np.isnan(r1)
        has_s1 = ~np.isnan(s1)

        # Previous bar close for crossover detection
        prev_close = np.empty(n)
        prev_close[0] = close[0]
        prev_close[1:] = close[:-1]

        # Session mask
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # Volume surge
        vol_surge = (vol_ma > 0) & (volume > vol_ma * vol_surge_mult)

        # R1 crossover: prev bar at or below R1, current bar above R1
        r1_crossover = (prev_close <= r1_clean) & (close > r1_clean)
        # S1 crossover: prev bar at or above S1, current bar below S1
        s1_crossover = (prev_close >= s1_clean) & (close < s1_clean)

        buy_ce = in_session & has_r1 & r1_crossover & (close > vwap) & vol_surge
        buy_pe = in_session & has_s1 & s1_crossover & (close < vwap) & vol_surge

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
