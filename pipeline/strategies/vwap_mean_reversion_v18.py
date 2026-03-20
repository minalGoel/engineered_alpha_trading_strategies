"""VWAP Mean Reversion v18: deep session-VWAP deviations on NIFTY.

On NIFTY, sharp intraday selloffs push the index 40-80 spot points below session
VWAP. At this deviation level, every VWAP-benchmarked institutional algorithm starts
accumulating aggressively. We detect the entry point using dev_momentum_24 (the 2-min
rate of change of the deviation turning positive) as confirmation that the flush is
exhausting before entering the CE.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "vwap_mean_reversion_v18"
    underlying = "NIFTY"
    session_start_minutes = 560    # 09:20 IST
    session_end_minutes = 925      # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 360             # 30 min warmup for vwap_stddev_360

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_threshold", 1.7, 1.2, 2.5),
            TunableParam("stop_pts", 6.0, 4.0, 10.0),
            TunableParam("target_pts", 10.0, 6.0, 18.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)
        day_id = spot_df["day_id"].to_numpy().astype(np.int32)

        zscore_threshold = params.get("zscore_threshold", 1.7)
        stop_pts = params.get("stop_pts", 6.0)
        target_pts = params.get("target_pts", 10.0)

        typical = (high + low + close) / 3.0

        # ── Session VWAP (cumulative from each day's open) ──────────────
        session_vwap = np.zeros(n, dtype=np.float64)
        cum_tpv = 0.0   # cumulative typical_price * volume
        cum_vol = 0.0   # cumulative volume
        prev_day = -1

        for i in range(n):
            d = day_id[i]
            if d != prev_day:
                cum_tpv = 0.0
                cum_vol = 0.0
                prev_day = d
            cum_tpv += typical[i] * volume[i]
            cum_vol += volume[i]
            if cum_vol > 0:
                session_vwap[i] = cum_tpv / cum_vol
            else:
                session_vwap[i] = close[i]

        # ── VWAP deviation (spot points) ──────────────────────────────
        vwap_dev = close - session_vwap

        # ── 30-min rolling stddev of vwap_dev (360 bars) ──────────────
        lookback_std = 360
        vwap_stddev = np.ones(n, dtype=np.float64)  # default 1.0 to avoid div/0
        for i in range(lookback_std, n):
            # Only compute within same day if possible; cross-day is fine for stddev
            window = vwap_dev[i - lookback_std:i]
            s = np.std(window)
            vwap_stddev[i] = s if s > 0.5 else 0.5   # floor at 0.5 pts

        # ── VWAP z-score ────────────────────────────────────────────
        vwap_zscore = vwap_dev / vwap_stddev

        # ── dev_momentum_24: 2-min change in VWAP deviation ─────────
        dev_momentum = np.zeros(n, dtype=np.float64)
        for i in range(24, n):
            dev_momentum[i] = vwap_dev[i] - vwap_dev[i - 24]

        # ── Session and warmup filters ───────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= self.max_lookback

        # ── Signals ─────────────────────────────────────────────────
        # Buy CE: price deeply below VWAP AND deviation decelerating (bounce starting)
        buy_ce = (
            in_session & warmed
            & (vwap_zscore < -zscore_threshold)
            & (dev_momentum > 0.0)
        )

        # Buy PE: price deeply above VWAP AND deviation decelerating (fade starting)
        buy_pe = (
            in_session & warmed
            & (vwap_zscore > zscore_threshold)
            & (dev_momentum < 0.0)
        )

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
