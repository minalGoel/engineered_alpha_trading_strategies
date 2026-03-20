"""zscore_reversion_v1 — NIFTY 5-second z-score mean-reversion strategy.

When NIFTY price deviates more than 2 standard deviations from its 10-minute
rolling mean AND the z-score velocity (30-second rate of change) turns back
toward zero, we enter ATM options in the reversion direction.

Original: Strategy_151.json — pure z-score reversion on NIFTY100 stocks, 1-min bars.
Adaptation: compressed lookback from 60-min to 10-min, velocity from 5-min to 30s.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "zscore_reversion_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — ensures 15 min of warmup data after open
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 120            # 10 min warmup (120 × 5s = 600s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_threshold", 2.0, 1.5, 3.0),
            TunableParam("vix_max", 25.0, 18.0, 30.0),
            TunableParam("stop_pts", 5.0, 3.0, 8.0),
            TunableParam("target_pts", 8.0, 5.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        zscore_threshold = params.get("zscore_threshold", 2.0)
        vix_max = params.get("vix_max", 25.0)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        # ── VIX regime filter ────────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Rolling 10-min (120-bar) mean and standard deviation ─────────────
        window = 120
        rolling_mean = np.full(n, np.nan)
        rolling_std = np.full(n, np.nan)
        for i in range(window, n):
            w = close[i - window:i]
            mu = np.mean(w)
            sigma = np.std(w)
            rolling_mean[i] = mu
            rolling_std[i] = sigma if sigma > 0.0 else np.nan

        # ── Z-score: normalized distance from 10-min mean ────────────────────
        valid = np.isfinite(rolling_mean) & np.isfinite(rolling_std) & (rolling_std > 0)
        zscore = np.where(valid, (close - rolling_mean) / rolling_std, 0.0)

        # ── Z-score velocity: 30-second (6-bar) rate of change ───────────────
        zscore_vel = np.zeros(n)
        zscore_vel[6:] = zscore[6:] - zscore[:-6]

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── VIX regime: only trade when VIX is not in high-vol panic mode ─────
        low_vix = vix_close < vix_max

        # Warmup guard: no signal until rolling window is populated
        warmed_up = np.zeros(n, dtype=bool)
        warmed_up[window:] = True

        # ── Entry signals ─────────────────────────────────────────────────────
        # Buy CE: z-score deeply negative (oversold) AND turning upward → bullish reversion
        buy_ce = (
            warmed_up
            & in_session
            & low_vix
            & (zscore < -zscore_threshold)
            & (zscore_vel > 0.0)
        )

        # Buy PE: z-score deeply positive (overbought) AND turning downward → bearish reversion
        buy_pe = (
            warmed_up
            & in_session
            & low_vix
            & (zscore > zscore_threshold)
            & (zscore_vel < 0.0)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,           # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
