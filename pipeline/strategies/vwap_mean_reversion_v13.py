"""VWAP Mean Reversion v13 — Deep z-score with 2-bar persistence and volume surge.

Mechanism: On NIFTY, when the index sustains a z-score deviation below -2.4 relative to
the 60-minute session standard deviation for at least two consecutive 5-second bars AND
volume is elevated above the 5-minute average, this marks multi-wave institutional forced
selling. The 2-bar persistence distinguishes sustained cascades from momentary spikes.
Every VWAP-benchmarked program on NIFTY starts absorbing supply simultaneously, creating
a snap-back within 30-120 seconds.

Differentiation from vwap_mean_reversion family:
  v10: single-bar trigger, 15-min stddev, 3-min volume baseline
  v11: single-bar, range-reclaim confirmation, 20-min stddev
  v12: single-bar, 3-min stddev (short), bar_return confirmation
  v13 (this): 2-bar persistence, 60-min stddev (longer session context), 5-min volume SMA
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "vwap_mean_reversion_v13"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 720            # 60-min warmup for stddev_720

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_threshold", 2.4, 1.8, 3.2),
            TunableParam("rel_vol_threshold", 1.3, 1.0, 2.0),
            TunableParam("vix_max", 30.0, 20.0, 40.0),
            TunableParam("stop_pts", 6.0, 3.0, 10.0),
            TunableParam("target_pts", 10.0, 5.0, 18.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill NaN before to_numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high  = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low   = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        day_id = spot_df["day_id"].fill_null(0).to_numpy()
        time_min = spot_df["time_minutes"].fill_null(0).to_numpy()

        zscore_threshold = params.get("zscore_threshold", 2.4)
        rel_vol_threshold = params.get("rel_vol_threshold", 1.3)
        vix_max = params.get("vix_max", 30.0)
        stop_pts = params.get("stop_pts", 6.0)
        target_pts = params.get("target_pts", 10.0)

        # ── Cumulative session VWAP (reset per day_id) ──
        typical = (high + low + close) / 3.0
        vwap = np.zeros(n)
        cum_tv = 0.0
        cum_v = 0.0
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                cum_tv = typical[i] * volume[i]
                cum_v = volume[i]
            else:
                cum_tv += typical[i] * volume[i]
                cum_v += volume[i]
            vwap[i] = cum_tv / cum_v if cum_v > 0.0 else close[i]

        # ── Rolling 60-min (720-bar) standard deviation of close ──
        STDDEV_WINDOW = 720
        stddev_720 = np.zeros(n)
        for i in range(STDDEV_WINDOW, n):
            stddev_720[i] = np.std(close[i - STDDEV_WINDOW:i])
        # Backfill early bars with first valid value (or fallback)
        first_valid_std = 5.0
        for i in range(STDDEV_WINDOW, n):
            if stddev_720[i] > 0.0:
                first_valid_std = stddev_720[i]
                break
        for i in range(STDDEV_WINDOW):
            stddev_720[i] = first_valid_std

        # ── VWAP z-score ──
        safe_std = np.where(stddev_720 > 0.5, stddev_720, 0.5)
        zscore = (close - vwap) / safe_std

        # ── Relative volume: volume / 5-min (60-bar) SMA ──
        VOL_WINDOW = 60
        vol_sma = np.zeros(n)
        for i in range(VOL_WINDOW, n):
            vol_sma[i] = np.mean(volume[i - VOL_WINDOW:i])
        # Backfill early bars
        first_valid_vol = 1.0
        for i in range(VOL_WINDOW, n):
            if vol_sma[i] > 0.0:
                first_valid_vol = vol_sma[i]
                break
        for i in range(VOL_WINDOW):
            vol_sma[i] = first_valid_vol
        safe_vol_sma = np.where(vol_sma > 0.0, vol_sma, 1.0)
        rel_vol = volume / safe_vol_sma

        # ── VIX close aligned to spot bars ──
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── 2-bar persistence: both current and prior bar must be extreme ──
        # zscore_prev[i] = zscore[i-1], neutral at 0 for bar 0
        zscore_prev = np.zeros(n)
        zscore_prev[1:] = zscore[:-1]

        # ── Session filter ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Entry signals ──
        # Bullish: deep negative z-score sustained 2 bars + volume surge → buy CE
        extreme_neg_now  = zscore < -zscore_threshold
        extreme_neg_prev = zscore_prev < -zscore_threshold
        buy_ce = (
            in_session
            & extreme_neg_now
            & extreme_neg_prev
            & (rel_vol > rel_vol_threshold)
            & (vix_close < vix_max)
        )

        # Bearish: deep positive z-score sustained 2 bars + volume surge → buy PE
        extreme_pos_now  = zscore > zscore_threshold
        extreme_pos_prev = zscore_prev > zscore_threshold
        buy_pe = (
            in_session
            & extreme_pos_now
            & extreme_pos_prev
            & (rel_vol > rel_vol_threshold)
            & (vix_close < vix_max)
        )

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
